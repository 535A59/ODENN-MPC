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
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm
import time
from typing import Tuple

class ODEDynamics(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, t: float, state_and_control: Tuple[torch.Tensor, torch.Tensor]):
        x = state_and_control[0]
        u = state_and_control[1]
        if x.dim() > 1 and u.dim() == 1:
            u = u.unsqueeze(0).expand(x.shape[0], -1)
        combined = torch.cat([x, u], dim=-1)
        return self.net(combined)

class SubstepODEIntStepper(nn.Module):
    def __init__(self, dynamics: nn.Module, default_dt: float, inner_dt: float, method: str = "rk4"):
        super().__init__()
        self.f = dynamics
        self.default_dt = default_dt
        self.inner_dt = inner_dt
        self.method = method
        self.n_sub = max(1, int(np.ceil(self.default_dt / self.inner_dt)))
        
    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float = None) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        if u.ndim == 1: u = u.unsqueeze(0)
        H = self.default_dt if dt is None else dt
        t_span = torch.linspace(0.0, H, steps=self.n_sub + 1, device=x.device, dtype=x.dtype)
        def dyn(t, x_state):
            return self.f(t, (x_state, u))
        options = {'step_size': self.inner_dt} if self.method == 'rk4' else None
        x_traj = odeint(dyn, x, t_span, method=self.method, options=options)
        return x_traj[-1]

def get_future_xref(full_ref_trajectory, k, N, state_dim, output_indices):

    if k + N > len(full_ref_trajectory):
        future_refs = full_ref_trajectory[k:, :]
        padding_needed = N - len(future_refs)
        padding = np.tile(future_refs[-1, :], (padding_needed, 1))
        future_refs = np.vstack([future_refs, padding])
    else:
        future_refs = full_ref_trajectory[k:k+N, :]
    
    X_ref_future = np.zeros((N, state_dim))

    X_ref_future[:, output_indices[0]] = future_refs[:, 0]
    X_ref_future[:, output_indices[1]] = future_refs[:, 1]

    dq3_ref = np.gradient(future_refs[:, 0], 1)
    dq4_ref = np.gradient(future_refs[:, 1], 1)
    X_ref_future[:, 5] = dq3_ref
    X_ref_future[:, 6] = dq4_ref

    return X_ref_future

class CMGPhysicsModel:
    def __init__(self):
        print("generate physics model...")
        Ka, Ib, Jb, Kb, Ic, Jc, Kc, Id, Jd, Kd = sp.symbols('Ka Ib Jb Kb Ic Jc Kc Id Jd Kd')
        T1max, T2max = sp.symbols('T1max T2max')
        b1 = Jd; b2 = Id - Jc - Jd + Kc; b3 = Ic + Id; b4 = Jb + Jc + Jd; b5 = Ib + Ic - Kb - Kc
        b6 = Id + Ka + Kb + Kc; b7 = Id - Jd; b8 = Ic - Jc + Kc + Id; b9 = Kc - Jc; b10 = Jc - Kc - Id + Jd
        b11 = Jc - Ic - Id - Ib + Jd + Kb; b12 = Kc - Jc - Jd - Ic; b16 = T1max; b17 = T2max
        q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4'); dq1, dq2, dq3, dq4 = sp.symbols('dq1 dq2 dq3 dq4')
        u1, u2 = sp.symbols('u1 u2'); f1, f2, f3 = sp.symbols('f1 f2 f3')
        q = sp.Matrix([q1, q2, q3, q4]); dq = sp.Matrix([dq1, dq2, dq3, dq4]); u = sp.Matrix([u1, u2])
        f = sp.Matrix([f1, f2, f3]); b13, b14, b15 = f[0], f[1], f[2]
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

    def __call__(self, t, x, u, f):
        return self.generated_func(*np.concatenate([x, u, f])).flatten()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['generated_func']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.generated_func = sp.lambdify(self.input_vars, self.fx_with_values, 'numpy')

def get_or_create_cmg_model():
    cache_file = 'cmg_model_instance.joblib'
    if os.path.exists(cache_file):
        try:
            model_instance = joblib.load(cache_file)
            return model_instance
        except Exception as e:
            os.remove(cache_file)
    model_instance = CMGPhysicsModel()
    joblib.dump(model_instance, cache_file)
    print(f"new physics model saved: {cache_file}")
    return model_instance


class ODEMPC:
    def __init__(self, plant_model, model_stepper, scaler_x_path, scaler_u_path,
                 state_dim, input_dim, N, Ts, Q, R,
                 optimizer="adam", lr=0.05, optim_iter=50, device="cpu"):
        self.plant_model = plant_model           
        self.model = model_stepper.to(device)    
        self.model.eval()
        self.N = N
        self.Ts = Ts
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._init_scalers(scaler_x_path, scaler_u_path)
        self.Q = torch.tensor(Q, dtype=self.dtype, device=self.device)
        self.R = torch.tensor(R, dtype=self.dtype, device=self.device)
        self.U = nn.Parameter(torch.zeros(self.N, self.input_dim, dtype=self.dtype, device=self.device))
        self.optim_iter = optim_iter
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam([self.U], lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not supported yet.")

    def _init_scalers(self, scaler_x_path, scaler_u_path):
        scaler_x = joblib.load(scaler_x_path)
        scaler_u = joblib.load(scaler_u_path)
        self.scale_x = torch.tensor(scaler_x.scale_, dtype=self.dtype, device=self.device)
        self.scale_u = torch.tensor(scaler_u.scale_, dtype=self.dtype, device=self.device)

    def _scale_x(self, x): return x / self.scale_x
    def _unscale_x(self, x_s): return x_s * self.scale_x
    def _scale_u(self, u): return u / self.scale_u

    def _cost(self, X_pred, X_ref_future):
        error = X_pred - X_ref_future
        cost = torch.einsum('ni,ij,nj->', error, self.Q, error)
        cost += torch.einsum('mi,ij,mj->', self.U, self.R, self.U)
        return cost

    def _sim_open_loop_model(self, x0_torch):
        X_pred = []
        x_current = x0_torch.clone()
        for i in range(self.N):
            x_scaled = self._scale_x(x_current)
            u_scaled = self._scale_u(self.U[i])
            x_next_scaled = self.model.step(x_scaled, u_scaled, dt=self.Ts)
            x_next = self._unscale_x(x_next_scaled.squeeze(0))
            X_pred.append(x_next)
            x_current = x_next
        return torch.stack(X_pred)

    @torch.no_grad()
    def _warm_start(self):
        if self.U is not None:
            rolled = torch.roll(self.U.data, shifts=-1, dims=0)
            self.U.data.copy_(rolled)
            self.U.data[-1].zero_()

    def solve_ocp(self, x_current_np, X_ref_future_np):
        x0_torch = torch.tensor(x_current_np, dtype=self.dtype, device=self.device)
        X_ref_future_torch = torch.tensor(X_ref_future_np, dtype=self.dtype, device=self.device)
        for i in range(self.optim_iter):
            self.optimizer.zero_grad()
            X_pred = self._sim_open_loop_model(x0_torch)
            loss = self._cost(X_pred, X_ref_future_torch)
            loss.backward()
            self.optimizer.step()
        print(f"loss: {loss.item()}")
        if not torch.isfinite(loss):
            print("Warning: loss non-finite")
        return self.U.detach().cpu().numpy()

    def simulate_physical_plant_one_step(self, x_current_phys, u_applied):

        # f_sim = np.array([0.000187, 0.0118, 0.0027])
        f_sim = np.array([0.0, 0.0, 0.0])
        sol = solve_ivp(
            lambda t, x: self.plant_model(t, x, u_applied,f_sim),
            [0, self.Ts],
            x_current_phys,
            method="RK45", rtol=1e-6, atol=1e-8
        )
        return sol.y[:, -1]

    @torch.no_grad()
    def simulate_nn_one_step(self, x_current_nn, u_applied):
        x_torch = torch.tensor(x_current_nn, dtype=self.dtype, device=self.device)
        u_torch = torch.tensor(u_applied, dtype=self.dtype, device=self.device)
        x_scaled = self._scale_x(x_torch)
        u_scaled = self._scale_u(u_torch)
        x_next_scaled = self.model.step(x_scaled, u_scaled, dt=self.Ts)
        x_next = self._unscale_x(x_next_scaled.squeeze(0))
        return x_next.cpu().numpy()

if __name__ == '__main__':
    Ts = 0.02
    sim_time = 100.0   
    total_steps = int(sim_time / Ts)
    N_horizon = 5

    physics_state_dim = 8
    odenn_state_dim = 7
    input_dim = 2
    DEVICE = "cpu"
    
    

    if not os.path.exists('scaler_x.joblib') or not os.path.exists('scaler_u.joblib'):
        placeholder_scaler_x = MaxAbsScaler().fit(np.random.rand(10, odenn_state_dim))
        placeholder_scaler_u = MaxAbsScaler().fit(np.random.rand(10, input_dim))
        joblib.dump(placeholder_scaler_x, 'scaler_x.joblib')
        joblib.dump(placeholder_scaler_u, 'scaler_u.joblib')

    try:
        mat_data = scipy.io.loadmat('Ref.mat')
        ref_vector_stacked = mat_data.get('Ref', mat_data.get('X_ref')).flatten()
        output_dim_ref = 2
        ref_output_trajectory = ref_vector_stacked.reshape(-1, output_dim_ref)
        print(f"Loaded ref shape: {ref_output_trajectory.shape}")
    except Exception as e:
        print(f"load Ref.mat failed: {e}")
        raise

    cmg_physics_model = get_or_create_cmg_model()

    dynamics_func = ODEDynamics(x_dim=odenn_state_dim, u_dim=input_dim)
    # state = 7, u = 2

    state_dict = torch.load('NeuralODE_best.pth', map_location=DEVICE)
    new_state_dict = {k.replace('dynamics_func.', ''): v for k, v in state_dict.items()}
    dynamics_func.load_state_dict(new_state_dict)
    dynamics_func.eval()
    dynamics_func_scripted = torch.jit.script(dynamics_func)
    print("✅ load ODENN model successfully.")

    stepper = SubstepODEIntStepper(dynamics=dynamics_func_scripted, default_dt=Ts, inner_dt=0.005, method="rk4")

    physics_to_nn_indices = [2, 3, 1, 4, 5, 6, 7]  
    output_indices_in_nn_state = [0, 1]

    Q_cost = np.zeros((odenn_state_dim, odenn_state_dim))
   
    Q_cost[0, 0] = 10.0  #q3
    Q_cost[1, 1] = 10.0  #q4

    Q_cost[2, 2] = 0.0  #q2
    Q_cost[3, 3] = 0.0   #dq1
    Q_cost[4, 4] = 0.0   #dq2

    Q_cost[5, 5] = 0.0   #dq3
    Q_cost[6, 6] = 0.0   #dq4

    R_cost = np.diag([100,100])

    mpc_controller = ODEMPC(
        plant_model=cmg_physics_model, 
        model_stepper=stepper,
        scaler_x_path='scaler_x.joblib', 
        scaler_u_path='scaler_u.joblib',
        state_dim=odenn_state_dim, 
        input_dim=input_dim, 
        N=N_horizon, 
        Ts=Ts,
        Q=Q_cost, 
        R=R_cost, 
        optimizer="adam", 
        lr=0.01, 
        optim_iter=10, 
        device=DEVICE
    )

    x_real_current = np.zeros(physics_state_dim) 

    x_nn_current = x_real_current[physics_to_nn_indices].copy()

    history_x_phys = [x_real_current.copy()]
    history_x_nn = [x_nn_current.copy()]
    history_u = []
    history_ref = [ref_output_trajectory[0, :].copy()]

    print("--- start MPC ---")
    start_time = time.time()

    # x_test = torch.tensor(x_nn_current, dtype=torch.float32)
    # u_test = torch.tensor([0.1, 0.1], dtype=torch.float32)
    # x_pred = mpc_controller.simulate_nn_one_step(x_test, u_test)
    # x_real = mpc_controller.simulate_physical_plant_one_step(
    #     np.pad(x_nn_current, (1,0)), 
    #     [0.1, 0.1]
    # )[physics_to_nn_indices]  
    # print("预测误差:", np.linalg.norm(x_pred - x_real))
    # error_per_dim = np.abs(x_pred - x_real)
    # print("各维度误差:", error_per_dim)
    # print("最大误差维度:", np.argmax(error_per_dim), 
    #     "值:", np.max(error_per_dim))

    # 可视化误差分布
    # plt.figure(figsize=(10, 5))
    # plt.bar(range(7), error_per_dim)
    # plt.xticks(range(7), ['q3', 'q4', 'q2', 'dq1', 'dq2', 'dq3', 'dq4'])
    # plt.title('ODENN单步预测各维度误差')
    # plt.ylabel('绝对误差')
    # plt.grid(axis='y')
    # plt.show()

   
    for k in tqdm(range(total_steps - 1)):
        X_ref_future_np = get_future_xref(ref_output_trajectory, k, N_horizon, odenn_state_dim, output_indices_in_nn_state)
        U_optimal_sequence = mpc_controller.solve_ocp(x_nn_current, X_ref_future_np)
        u_apply = U_optimal_sequence[0, :].copy()
        x_real_next = mpc_controller.simulate_physical_plant_one_step(x_real_current, u_apply)
        x_nn_next = x_real_next[physics_to_nn_indices].copy()
        history_x_phys.append(x_real_next.copy())
        history_x_nn.append(x_nn_next.copy())
        history_u.append(u_apply.copy())
        history_ref.append(ref_output_trajectory[k+1, :].copy())

        x_real_current = x_real_next
        x_nn_current = x_nn_next

        mpc_controller._warm_start()

    end_time = time.time()
    print(f"finish simulation  {end_time - start_time:.2f}s，average time cost {(end_time - start_time)/ (total_steps-1) * 1000:.2f} ms")

    history_x_phys = np.array(history_x_phys)
    history_u = np.array(history_u)
    history_ref = np.array(history_ref)
    time_axis = np.arange(history_x_phys.shape[0]) * Ts

    q3_deg = history_x_phys[:, 2] * 180/np.pi   
    q4_deg = history_x_phys[:, 3] * 180/np.pi  
    ref_q3_deg = history_ref[:, 0] * 180/np.pi
    ref_q4_deg = history_ref[:, 1] * 180/np.pi

    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(time_axis, ref_q3_deg, 'k-', label='q3_ref')
    plt.plot(time_axis, q3_deg, 'r--', label='q3_phys')
    plt.legend(); plt.grid(); plt.ylabel('q3 [deg]')

    plt.subplot(3,1,2)
    plt.plot(time_axis, ref_q4_deg, 'k-', label='q4_ref')
    plt.plot(time_axis, q4_deg, 'r--', label='q4_phys')
    plt.legend(); plt.grid(); plt.ylabel('q4 [deg]')

    plt.subplot(3,1,3)
    tu = np.arange(len(history_u)) * Ts
    if len(history_u)>0:
        plt.plot(tu, history_u[:,0], label='u1 [Nm]')
        plt.plot(tu, history_u[:,1], label='u2 [Nm]')
    plt.legend(); plt.grid(); plt.xlabel('Time [s]'); plt.ylabel('Torque')

    plt.tight_layout()
    plt.show()