import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torchdiffeq import odeint
import os
import sympy as sp
import joblib
from tqdm import tqdm
import time
from typing import Tuple

# ----------------------------------------------------------------------
# 神经网络动力学模型定义 (已修正时间缩放)
# ----------------------------------------------------------------------
class ODEDynamics(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim=256):
        super(ODEDynamics, self).__init__()
        # 神经网络用于逼近 f
        self.net = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, t, state_and_control):
        x, u = state_and_control
        xu = torch.cat([x, u], dim=1)
        return self.net(xu)



class NeuralODE(nn.Module):
    def __init__(self, dynamics_func, solver='dopri5', rtol=1e-5, atol=1e-5):
        super(NeuralODE, self).__init__()
        self.dynamics_func = dynamics_func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x0, u, t_span):
        def wrapped_dynamics(t, x):
            return self.dynamics_func(t, (x, u))
        pred_x = odeint(wrapped_dynamics, x0, t_span,
                        method=self.solver, rtol=self.rtol, atol=self.atol)
        return pred_x[-1]
# ----------------------------------------------------------------------
# 辅助函数和物理模型
# ----------------------------------------------------------------------
def get_future_xref(full_ref_trajectory, k, N, state_dim, output_indices):
    if k + N + 1 > len(full_ref_trajectory):
        future_refs = full_ref_trajectory[k+1:, :]
        padding_needed = N - len(future_refs)
        if padding_needed > 0:
            padding = np.tile(future_refs[-1, :], (padding_needed, 1))
            future_refs = np.vstack([future_refs, padding])
    else:
        future_refs = full_ref_trajectory[k+1:k+N+1, :]
    
    X_ref_future = np.zeros((N, state_dim))
    X_ref_future[:, output_indices[0]] = future_refs[:, 0]
    X_ref_future[:, output_indices[1]] = future_refs[:, 1]
    
    if len(future_refs) > 1:
        dq_ref = np.diff(future_refs, axis=0, append=future_refs[-1:, :]) / 0.02
        X_ref_future[:, 5] = dq_ref[:, 0]
        X_ref_future[:, 6] = dq_ref[:, 1]
    
    return X_ref_future

class CMGPhysicsModel:
    def __init__(self):
        print("generating physics model...")
        # ... (Symbolic model definition remains the same)
        Ka, Ib, Jb, Kb, Ic, Jc, Kc, Id, Jd, Kd = sp.symbols('Ka Ib Jb Kb Ic Jc Kc Id Jd Kd'); T1max, T2max = sp.symbols('T1max T2max')
        b1 = Jd; b2 = Id - Jc - Jd + Kc; b3 = Ic + Id; b4 = Jb + Jc + Jd; b5 = Ib + Ic - Kb - Kc; b6 = Id + Ka + Kb + Kc; b7 = Id - Jd; b8 = Ic - Jc + Kc + Id; b9 = Kc - Jc; b10 = Jc - Kc - Id + Jd
        b11 = Jc - Ic - Id - Ib + Jd + Kb; b12 = Kc - Jc - Jd - Ic; b16 = T1max; b17 = T2max
        q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4'); dq1, dq2, dq3, dq4 = sp.symbols('dq1 dq2 dq3 dq4'); u1, u2 = sp.symbols('u1 u2'); f1, f2, f3 = sp.symbols('f1 f2 f3')
        q = sp.Matrix([q1, q2, q3, q4]); dq = sp.Matrix([dq1, dq2, dq3, dq4]); u = sp.Matrix([u1, u2]); f = sp.Matrix([f1, f2, f3]); b13, b14, b15 = f[0], f[1], f[2]
        s2, c2 = sp.sin(q2), sp.cos(q2); s3, c3 = sp.sin(q3), sp.cos(q3)
        M = sp.Matrix([[b1, 0, b1*c2, b1*s2*c3], [0, b3, 0, -b3*s3], [b1*c2, 0, b2*s2**2 + b4, -b2*s2*c2*c3],[b1*s2*c3, -b3*s3, -b2*s2*c2*c3, -b2*s2**2*c3**2 + b5*s3**2 + b6]])
        D = sp.diag(b13, b14, b15, b15)
        k1_tilde = b1*(c2*c3*dq2*dq4 - s2*dq2*dq3 - s2*s3*dq3*dq4)
        k2_tilde = b1*(s2*dq1*dq3 - c2*c3*dq1*dq4) + b2*(c2*c3**2*s2*dq4**2 - c2*s2*dq3**2) - b8*c3*dq3*dq4 + b7*(1-2*s2**2)*c3*dq3*dq4 + 2*b9*c2**2*c3*dq3*dq4
        k3_tilde = b1*(s2*s3*dq1*dq4 - s2*dq1*dq2) + (b8 + b7)*c3*dq2*dq4 + b11*s3*c3*dq4**2 + b10*(2*c2**2*c3*dq2*dq4 - 2*s2*c2*dq2*dq3 - s3*c2**2*c3*dq4**2)
        k4_tilde = b1*(c2*c3*dq1*dq2 - s2*s3*dq1*dq3) + b2*s2*s3*c2*dq3**2 - 2*b11*s3*c3*dq3*dq4 + 2*b10*(c2**2*c3*dq2*dq3 + s2*c2*c3**2*dq2*dq4 + s3*c2**2*c3*dq3*dq4) + b12*c3*dq2*dq3
        K = sp.Matrix([k1_tilde, k2_tilde, k3_tilde, k4_tilde]); T = sp.Matrix([[b16, 0], [0, b17], [0, 0], [0, 0]])
        M_inv = M.inv(); ddq = M_inv * (T * u - D * dq - K); fx = sp.Matrix([dq, ddq])
        constants = {'Ka': 6.70e-2, 'Ib': 1.19e-2, 'Jb': 1.78e-2, 'Kb': 2.97e-2, 'Ic': 0.92e-2, 'Jc': 2.30e-2, 'Kc': 2.20e-2, 'Id': 1.48e-2, 'Jd': 2.73e-2, 'Kd': 1.48e-2, 'T1max': 66.60e-2, 'T2max': 244.0e-2}
        self.input_vars = [q1, q2, q3, q4, dq1, dq2, dq3, dq4, u1, u2, f1, f2, f3]
        self.fx_with_values = fx.subs(constants)
        self.generated_func = sp.lambdify(self.input_vars, self.fx_with_values, 'numpy')
        print("finished!")
    def __call__(self, t, x, u, f): return self.generated_func(*np.concatenate([x, u, f])).flatten()
    def __getstate__(self): state = self.__dict__.copy(); del state['generated_func']; return state
    def __setstate__(self, state): self.__dict__.update(state); self.generated_func = sp.lambdify(self.input_vars, self.fx_with_values, 'numpy')

def get_or_create_cmg_model():
    cache_file = 'cmg_model_instance.joblib'
    if os.path.exists(cache_file):
        try: model_instance = joblib.load(cache_file); print("Loaded physics model from cache."); return model_instance
        except Exception as e: print(f"Failed to load cached model: {e}. Recreating."); os.remove(cache_file)
    model_instance = CMGPhysicsModel(); joblib.dump(model_instance, cache_file); print(f"New physics model created and saved: {cache_file}"); return model_instance


# ----------------------------------------------------------------------
# MPC 控制器定义 (最终修正版 2：修正维度错误)
# ----------------------------------------------------------------------
class ODEMPC:
    def __init__(self, plant_model, model_stepper, 
                 state_dim, input_dim, N_horizon, Ts, Q, R, 
                 u_min, u_max, scaler_x, scaler_u, optim_iter=10):
        self.plant_model = plant_model
        self.model = model_stepper
        self.model.eval()
        self.n = state_dim; self.m = input_dim; self.N = N_horizon; self.Ts = Ts; self.optim_iter = optim_iter
        self.device = next(self.model.parameters()).device; self.dtype = next(self.model.parameters()).dtype
        self.Q = torch.tensor(Q, dtype=self.dtype, device=self.device)
        self.R = torch.tensor(R, dtype=self.dtype, device=self.device)
        self.u_min = torch.tensor(u_min, dtype=self.dtype, device=self.device)
        self.u_max = torch.tensor(u_max, dtype=self.dtype, device=self.device)
        self.t_span = torch.tensor([0.0, 1.0]).to(self.device)  # 归一化时间跨度

        # 从scaler对象中提取均值和标准差，并存为张量
        self.x_mean = torch.tensor(scaler_x.mean_, dtype=self.dtype, device=self.device)
        self.x_scale = torch.tensor(scaler_x.scale_, dtype=self.dtype, device=self.device)
        self.u_mean = torch.tensor(scaler_u.mean_, dtype=self.dtype, device=self.device)
        self.u_scale = torch.tensor(scaler_u.scale_, dtype=self.dtype, device=self.device)

        print("ODEMPC initialized with PyTorch-based scalers for gradient flow.")

        self.U = nn.Parameter(torch.zeros(self.N, self.m, dtype=self.dtype, device=self.device))
        self.optimizer = torch.optim.Adam([self.U], lr=0.05)

    # 用纯张量运算重写缩放函数
    def _scale_x(self, x: torch.Tensor) -> torch.Tensor:
        # This function now correctly returns a 2D tensor [1, 7]
        return ((x - self.x_mean) / self.x_scale).unsqueeze(0) if x.ndim == 1 else (x - self.x_mean) / self.x_scale

    def _unscale_x(self, x_s: torch.Tensor) -> torch.Tensor:
        # This function also returns a 2D tensor [1, 7]
        return x_s * self.x_scale + self.x_mean

    def _scale_u(self, u: torch.Tensor) -> torch.Tensor:
        # This function returns a 2D tensor [1, 2]
        return ((u - self.u_mean) / self.u_scale).unsqueeze(0) if u.ndim == 1 else (u - self.u_mean) / self.u_scale
    
    def _sim_open_loop_model(self, x0, U_seq):
        X_pred = [x0.unsqueeze(0)]; 
        x_current = x0
        for k in range(self.N):
            u_k = U_seq[k]
            x_current_scaled = self._scale_x(x_current)
            u_k_scaled = self._scale_u(u_k)
            x_next_scaled = self.model(x_current_scaled, u_k_scaled, self.t_span)
            x_next = self._unscale_x(x_next_scaled)
            
            # 关键修正 1: x_next 已经是 2D [1, 7]，直接 append
            X_pred.append(x_next)
            
            # 关键修正 2: 将 x_next 变回 1D 以用于下一次循环
            x_current = x_next.squeeze(0)
            
        return torch.cat(X_pred, dim=0)

    def _cost(self, X_pred, X_ref_future, U_seq):
        error = X_pred[1:, :] - X_ref_future 
        state_cost = torch.sum(torch.einsum('bi,ij,bj->b', error, self.Q, error))
        control_cost = torch.sum(torch.einsum('bi,ij,bj->b', U_seq, self.R, U_seq))
        return state_cost + control_cost

    def _warm_start(self):
        if self.U.grad is not None: self.U.grad.zero_()
        with torch.no_grad():
            rolled_U = torch.roll(self.U.data, shifts=-1, dims=0)
            rolled_U[-1].zero_()
            self.U.data.copy_(rolled_U)

    def solve_ocp(self, x0_nn, X_ref_future_np):
        x0_tensor = torch.tensor(x0_nn, dtype=self.dtype, device=self.device)
        xref_tensor = torch.tensor(X_ref_future_np, dtype=self.dtype, device=self.device)
        for _ in range(self.optim_iter):
            self.optimizer.zero_grad()
            X_pred = self._sim_open_loop_model(x0_tensor, self.U)
            loss = self._cost(X_pred, xref_tensor, self.U)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.U.data.clamp_(self.u_min, self.u_max)
                
        print(f'Optimal cost: {loss.item():.4f}')
        return self.U.detach().clone()
    

if __name__ == '__main__':
    # --- 基本参数设置 ---
    Ts = 0.02
    sim_time = 50.0
    total_steps = int(sim_time / Ts)
    physics_state_dim = 8
    odenn_state_dim = 7
    input_dim = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 加载参考轨迹 ---
    try:
        mat_data = scipy.io.loadmat('Ref_truncated.mat')
        ref_vector_stacked = mat_data.get('Ref', mat_data.get('X_ref')).flatten()
        output_dim_ref = 2; ref_output_trajectory = ref_vector_stacked.reshape(-1, output_dim_ref)
        print(f"Loaded ref shape: {ref_output_trajectory.shape}")
    except Exception as e: print(f"load Ref.mat failed: {e}"); raise

    # --- 初始化物理和神经网络模型 ---
    cmg_physics_model = get_or_create_cmg_model()
    dynamics_func = ODEDynamics(x_dim=odenn_state_dim, u_dim=input_dim).to(DEVICE)
    neural_ode_model = NeuralODE(dynamics_func).to(DEVICE)
    neural_ode_model.load_state_dict(torch.load("NeuralODE_best.pth", map_location=DEVICE))
    neural_ode_model.eval()


    try:
        scaler_x = joblib.load('scaler_x_7.joblib')
        scaler_u = joblib.load('scaler_u_7.joblib')
        print("Loaded scalers from training.")
    except FileNotFoundError:
        print("ERROR: scaler_x.joblib or scaler_u.joblib not found.")
        raise
        
    # --- MPC 控制器参数设置 ---
    physics_to_nn_indices = [2, 3, 1, 4, 5, 6, 7]
    output_indices_in_nn_state = [0, 1]
    
    N_horizon = 10
    R_cost = np.diag([20.0, 20.0])
    Q_cost = np.diag([100.0, 50.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    u_min_constraint = [-0.5, -0.5]
    u_max_constraint = [0.5, 0.5]

    mpc_controller = ODEMPC(
        plant_model=cmg_physics_model, model_stepper=neural_ode_model,
        state_dim=odenn_state_dim, input_dim=input_dim,
        N_horizon=N_horizon, Ts=Ts, Q=Q_cost, R=R_cost,
        u_min=u_min_constraint, u_max=u_max_constraint,
        scaler_x=scaler_x, scaler_u=scaler_u,
        optim_iter=30)
    
    # --- 仿真循环 ---
    x0_phys = np.zeros(physics_state_dim)
    history_x_phys = [x0_phys.copy()]; 
    x_real_current = x0_phys.copy()
    history_u = [np.zeros(mpc_controller.m)]
    f_sim = np.array([0.000, 0.0, 0.0])

    # --- 1. 修改：开启交互模式并创建图形窗口 ---
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.tight_layout(pad=3.0)
    
    # 准备参考轨迹数据
    ref_time_axis = np.arange(ref_output_trajectory.shape[0]) * Ts
    ref_q3_deg = ref_output_trajectory[:, 0] * 180/np.pi
    ref_q4_deg = ref_output_trajectory[:, 1] * 180/np.pi

    # 初始化绘图元素
    # 子图1: 角度跟踪
    axes[0].set_title('MPC Tracking Performance (Live)')
    axes[0].set_ylabel('Angle [deg]')
    axes[0].grid(True)
    line_q3_ref, = axes[0].plot(ref_time_axis, ref_q3_deg, 'k-', label='q3_ref', alpha=0.5)
    line_q4_ref, = axes[0].plot(ref_time_axis, ref_q4_deg, 'b-', label='q4_ref', alpha=0.5)
    line_q3_act, = axes[0].plot([], [], 'r--', label='q3_actual')
    line_q4_act, = axes[0].plot([], [], 'g--', label='q4_actual')
    axes[0].legend()
    axes[0].set_xlim(0, sim_time)
    
    # 子图2: 控制输入
    axes[1].set_ylabel('Control Input [Nm]')
    axes[1].grid(True)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
    line_u1, = axes[1].step([], [], where='post', label='u1 (tau_1)')
    line_u2, = axes[1].step([], [], where='post', label='u2 (tau_2)')
    axes[1].legend()
    axes[1].set_xlim(0, sim_time)
    axes[1].set_ylim(-0.6, 0.6)

    # 子图3: 角速度
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Angular Velocities [rad/s]')
    axes[2].grid(True)
    line_dq3, = axes[2].plot([], [], label='dq3_actual')
    line_dq4, = axes[2].plot([], [], label='dq4_actual')
    axes[2].legend()
    axes[2].set_xlim(0, sim_time)
    
    print("\n--- Starting MPC Simulation with Live Plotting ---")
    start_time = time.time()
    
    # 使用tqdm但禁用默认输出，手动更新
    progress_bar = tqdm(range(total_steps - 1))
    
    for k in progress_bar:
        x_nn_current = x_real_current[physics_to_nn_indices].copy()
        X_ref_future_np = get_future_xref(ref_output_trajectory, k, N_horizon, odenn_state_dim, output_indices_in_nn_state)
        
        # 求解MPC
        U_optimal_sequence = mpc_controller.solve_ocp(x_nn_current, X_ref_future_np)
        u_apply = U_optimal_sequence[0, :].cpu().numpy()
        
        # 应用控制并仿真一步
        sol = solve_ivp(
            lambda t, x: cmg_physics_model(t, x, u_apply, f_sim),
            [0, Ts], x_real_current, method="RK45", rtol=1e-6, atol=1e-8)
        
        x_real_next = sol.y[:, -1]
        history_x_phys.append(x_real_next.copy())
        history_u.append(u_apply.copy())
        x_real_current = x_real_next
        mpc_controller._warm_start()
        
        # --- 2. 修改：在循环中更新绘图 (每50步更新一次以提高效率) ---
        if k % 50 == 0:
            # 准备当前的数据
            current_history_x = np.array(history_x_phys)
            current_history_u = np.array(history_u)
            current_time_axis = np.arange(current_history_x.shape[0]) * Ts
            
            # 更新角度图
            q3_deg = current_history_x[:, 2] * 180/np.pi
            q4_deg = current_history_x[:, 3] * 180/np.pi
            line_q3_act.set_data(current_time_axis, q3_deg)
            line_q4_act.set_data(current_time_axis, q4_deg)
            
            # 更新控制输入图
            line_u1.set_data(current_time_axis, current_history_u[:, 0])
            line_u2.set_data(current_time_axis, current_history_u[:, 1])

            # 更新角速度图 (dq3是第7个状态，dq4是第8个)
            line_dq3.set_data(current_time_axis, current_history_x[:, 6])
            line_dq4.set_data(current_time_axis, current_history_x[:, 7])
            
            # 自动调整Y轴范围
            for ax in axes:
                ax.relim()
                ax.autoscale_view()
            
            # 刷新画布
            fig.canvas.draw()
            fig.canvas.flush_events()

    end_time = time.time()
    print(f"Finish simulation in {end_time - start_time:.2f}s")

    # --- 3. 修改：仿真结束后保持图形显示 ---
    plt.ioff()
    # 最终再画一次，确保显示的是完整数据
    final_history_x = np.array(history_x_phys)
    final_history_u = np.array(history_u)
    final_time_axis = np.arange(final_history_x.shape[0]) * Ts
    line_q3_act.set_data(final_time_axis, final_history_x[:, 2] * 180/np.pi)
    line_q4_act.set_data(final_time_axis, final_history_x[:, 3] * 180/np.pi)
    line_u1.set_data(final_time_axis, final_history_u[:, 0])
    line_u2.set_data(final_time_axis, final_history_u[:, 1])
    line_dq3.set_data(final_time_axis, final_history_x[:, 6])
    line_dq4.set_data(final_time_axis, final_history_x[:, 7])
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    
    axes[0].set_title('MPC Tracking Performance (Final Result)')
    plt.show()