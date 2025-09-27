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
        x, u = state_and_control
        if x.dim() > 1 and u.dim() == 1:
            u = u.unsqueeze(0).expand(x.shape[0], -1)
        combined = torch.cat([x, u], dim=-1)
        return self.net(combined)


class ODEDynamicsWrapper(nn.Module):
    def __init__(self, dynamics_func: ODEDynamics, u: torch.Tensor):
        super().__init__()
        self.f = dynamics_func
        self.u = u

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.f(t, (x, self.u))


class SubstepODEIntStepper(nn.Module):
    def __init__(self, dynamics: ODEDynamics, default_dt: float, inner_dt: float, method: str = "rk4"):
        super().__init__()
        self.f = dynamics
        self.default_dt = default_dt
        self.inner_dt = inner_dt
        self.method = method
        self.n_sub = int(np.ceil(self.default_dt / self.inner_dt))
        if self.n_sub == 0:
            self.n_sub = 1
        
    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if u.ndim == 1:
            u = u.unsqueeze(0)
            
        H = self.default_dt
        if dt != 0.0:
            H = dt
            
        t_span = torch.linspace(0.0, H, steps=self.n_sub + 1, device=x.device, dtype=x.dtype)
        
        dyn_wrapper = ODEDynamicsWrapper(self.f, u)
        
        options = {'step_size': self.inner_dt} if self.method == 'rk4' else None
        x_traj = odeint(dyn_wrapper, x, t_span, method=self.method, options=options)
        return x_traj[-1]

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
        print("generate physics model...")
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


class ODEMPC:
    def __init__(self, plant_model, model_stepper: SubstepODEIntStepper, 
                 state_dim, input_dim, N_horizon, Ts, Q, R, optim_iter=10):
        self.plant_model = plant_model
        self.model = model_stepper
        self.model.eval()
        self.n = state_dim; self.m = input_dim; self.N = N_horizon; self.Ts = Ts; self.optim_iter = optim_iter
        self.device = next(self.model.parameters()).device; self.dtype = next(self.model.parameters()).dtype
        self.Q = torch.tensor(Q, dtype=self.dtype, device=self.device)
        self.R = torch.tensor(R, dtype=self.dtype, device=self.device)
        self.U = nn.Parameter(torch.zeros(self.N, self.m, dtype=self.dtype, device=self.device))
        self.optimizer = torch.optim.Adam([self.U], lr=0.1)
        self._init_scalers_from_notebook()

    def _init_scalers_from_notebook(self):
        print("Initializing scalers from hard-coded values found in the notebook...")
        x_max_abs = np.array([np.pi/2, np.pi/2, np.pi/2, 2.0, 3.0, 4.0, 4.0])
        u_max_abs = np.array([0.6, 2.4])
        self.scale_x = torch.tensor(x_max_abs, dtype=self.dtype, device=self.device)
        self.scale_u = torch.tensor(u_max_abs, dtype=self.dtype, device=self.device)
        print("Scalers initialized successfully.")

    def _scale_x(self, x): return x / self.scale_x
    def _unscale_x(self, x_s): return x_s * self.scale_x
    def _scale_u(self, u): return u / self.scale_u

    def _sim_open_loop_model(self, x0, U_seq):
        X_pred = [x0.unsqueeze(0)]; x_current = x0
        for k in range(self.N):
            u_k = U_seq[k]
            x_current_scaled = self._scale_x(x_current)
            u_k_scaled = self._scale_u(u_k)
            x_next_scaled = self.model.step(x_current_scaled, u_k_scaled, dt=self.Ts)
            x_next = self._unscale_x(x_next_scaled)
            X_pred.append(x_next)
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
        return self.U.detach().clone()

if __name__ == '__main__':
    Ts = 0.02; sim_time = 50.0; total_steps = int(sim_time / Ts); N_horizon = 20
    physics_state_dim = 8; odenn_state_dim = 7; input_dim = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        mat_data = scipy.io.loadmat('Ref.mat')
        ref_vector_stacked = mat_data.get('Ref', mat_data.get('X_ref')).flatten()
        output_dim_ref = 2; ref_output_trajectory = ref_vector_stacked.reshape(-1, output_dim_ref)
        print(f"Loaded ref shape: {ref_output_trajectory.shape}")
    except Exception as e: print(f"load Ref.mat failed: {e}"); raise

    cmg_physics_model = get_or_create_cmg_model()
    dynamics_func = ODEDynamics(x_dim=odenn_state_dim, u_dim=input_dim).to(DEVICE)
    
    try:
        state_dict = torch.load('NeuralODE_best.pth', map_location=DEVICE)
        new_state_dict = {k.replace('dynamics_func.', ''): v for k, v in state_dict.items()}
        dynamics_func.load_state_dict(new_state_dict)
        print("load ODENN model successfully.")
    except Exception as e: print(f"Failed to load model weights: {e}")

    stepper = SubstepODEIntStepper(dynamics=dynamics_func, default_dt=Ts, inner_dt=0.005, method="rk4")
    
    physics_to_nn_indices = [2, 3, 1, 4, 5, 6, 7]
    output_indices_in_nn_state = [0, 1]
    Q_cost = np.diag([10.0, 10.0, 0.1, 0.0, 0.1, 1.0, 1.0])
    R_cost = np.diag([0.1, 0.1])

    mpc_controller = ODEMPC(
        plant_model=cmg_physics_model, model_stepper=stepper,
        state_dim=odenn_state_dim, input_dim=input_dim,
        N_horizon=N_horizon, Ts=Ts, Q=Q_cost, R=R_cost,
        optim_iter=20)
    
    x0_phys = np.zeros(physics_state_dim)
    history_x_phys = [x0_phys.copy()]; x_real_current = x0_phys.copy()
    history_u = [np.zeros(mpc_controller.m)]
    f_sim = np.array([0.000187, 0.0118, 0.0027])

    print("--- Starting MPC Simulation ---")
    start_time = time.time()
    for k in tqdm(range(total_steps - 1)):
        x_nn_current = x_real_current[physics_to_nn_indices].copy()
        X_ref_future_np = get_future_xref(ref_output_trajectory, k, N_horizon, odenn_state_dim, output_indices_in_nn_state)
        U_optimal_sequence = mpc_controller.solve_ocp(x_nn_current, X_ref_future_np)
        u_apply = U_optimal_sequence[0, :].cpu().numpy()
        sol = solve_ivp(
            lambda t, x: cmg_physics_model(t, x, u_apply, f_sim),
            [0, Ts], x_real_current, method="RK45", rtol=1e-6, atol=1e-8)
        x_real_next = sol.y[:, -1]
        history_x_phys.append(x_real_next.copy()); history_u.append(u_apply.copy())
        x_real_current = x_real_next
        mpc_controller._warm_start()

    end_time = time.time()
    print(f"Finish simulation in {end_time - start_time:.2f}s")

    history_x_phys = np.array(history_x_phys); history_u = np.array(history_u)
    time_axis = np.arange(history_x_phys.shape[0]) * Ts
    ref_time_axis = np.arange(ref_output_trajectory.shape[0]) * Ts

    q3_deg = history_x_phys[:, 2] * 180/np.pi; q4_deg = history_x_phys[:, 3] * 180/np.pi
    ref_q3_deg = ref_output_trajectory[:, 0] * 180/np.pi; ref_q4_deg = ref_output_trajectory[:, 1] * 180/np.pi
    
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(ref_time_axis, ref_q3_deg, 'k-', label='q3_ref', alpha=0.7)
    plt.plot(time_axis, q3_deg, 'r--', label='q3_actual')
    plt.plot(ref_time_axis, ref_q4_deg, 'b-', label='q4_ref', alpha=0.7)
    plt.plot(time_axis, q4_deg, 'g--', label='q4_actual')
    plt.legend(); plt.grid(); plt.ylabel('Angle [deg]'); plt.title('MPC Tracking Performance')

    plt.subplot(2,1,2)
    tu = np.arange(len(history_u)) * Ts
    if len(history_u) > 0:
        plt.step(tu, history_u[:,0], label='u1 [Nm]')
        plt.step(tu, history_u[:,1], label='u2 [Nm]')
    plt.legend(); plt.grid(); plt.xlabel('Time [s]'); plt.ylabel('Control Input')
    plt.tight_layout()
    plt.show()