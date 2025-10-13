import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import os
import sympy as sp
import joblib
from tqdm import tqdm
import time

# --- Acados and l4casadi Imports ---
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
from l4casadi import L4CasADi

# ==============================================================================
# 1. MODEL AND UTILITY DEFINITIONS
# ==============================================================================

class ODEDynamics(nn.Module):
    """
    The original dynamics model class, trained using torchdiffeq.
    This class remains UNCHANGED.
    """
    def __init__(self, x_dim, u_dim, hidden_dim=256):
        super(ODEDynamics, self).__init__()
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

class L4CasADiWrapper(nn.Module):
    """
    A wrapper class to make the trained ODEDynamics model compatible with l4casadi.
    """
    def __init__(self, original_model, x_dim, u_dim):
        super().__init__()
        self.original_model = original_model
        self.x_dim = x_dim
        self.u_dim = u_dim

    def forward(self, xu):
        x = xu[:, :self.x_dim]
        u = xu[:, self.x_dim:]
        return self.original_model(0.0, (x, u))

class CMGPhysicsModel:
    """ The physics-based plant model for simulation. Unchanged. """
    def __init__(self):
        print("Generating physics model...")
        Ka, Ib, Jb, Kb, Ic, Jc, Kc, Id, Jd, Kd = sp.symbols('Ka Ib Jb Kb Ic Jc Kc Id Jd Kd'); T1max, T2max = sp.symbols('T1max T2max')
        b1 = Jd; b2 = Id - Jc - Jd + Kc; b3 = Ic + Id; b4 = Jb + Jc + Jd; b5 = Ib + Ic - Kb - Kc; b6 = Id + Ka + Kb + Kc; b7 = Id - Jd; b8 = Ic - Jc + Kc + Id; b9 = Kc - Jc; b10 = Jc - Kc - Id + Jd
        b11 = Jc - Ic - Id - Ib + Jd + Kb; b12 = Kc - Jc - Jd - Ic
        q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4'); dq1, dq2, dq3, dq4 = sp.symbols('dq1 dq2 dq3 dq4'); u1, u2 = sp.symbols('u1 u2'); f1, f2, f3 = sp.symbols('f1 f2 f3')
        q = sp.Matrix([q1, q2, q3, q4]); dq = sp.Matrix([dq1, dq2, dq3, dq4]); u = sp.Matrix([u1, u2]); f = sp.Matrix([f1, f2, f3]); b13, b14, b15 = f[0], f[1], f[2]
        s2, c2 = sp.sin(q2), sp.cos(q2); s3, c3 = sp.sin(q3), sp.cos(q3)
        M = sp.Matrix([[b1, 0, b1*c2, b1*s2*c3], [0, b3, 0, -b3*s3], [b1*c2, 0, b2*s2**2 + b4, -b2*s2*c2*c3],[b1*s2*c3, -b3*s3, -b2*s2*c2*c3, -b2*s2**2*c3**2 + b5*s3**2 + b6]])
        D = sp.diag(b13, b14, b15, b15)
        k1_tilde = b1*(c2*c3*dq2*dq4 - s2*dq2*dq3 - s2*s3*dq3*dq4); k2_tilde = b1*(s2*dq1*dq3 - c2*c3*dq1*dq4) + b2*(c2*c3**2*s2*dq4**2 - c2*s2*dq3**2) - b8*c3*dq3*dq4 + b7*(1-2*s2**2)*c3*dq3*dq4 + 2*b9*c2**2*c3*dq3*dq4; k3_tilde = b1*(s2*s3*dq1*dq4 - s2*dq1*dq2) + (b8 + b7)*c3*dq2*dq4 + b11*s3*c3*dq4**2 + b10*(2*c2**2*c3*dq2*dq4 - 2*s2*c2*dq2*dq3 - s3*c2**2*c3*dq4**2); k4_tilde = b1*(c2*c3*dq1*dq2 - s2*s3*dq1*dq3) + b2*s2*s3*c2*dq3**2 - 2*b11*s3*c3*dq3*dq4 + 2*b10*(c2**2*c3*dq2*dq3 + s2*c2*c3**2*dq2*dq4 + s3*c2**2*c3*dq3*dq4) + b12*c3*dq2*dq3
        K = sp.Matrix([k1_tilde, k2_tilde, k3_tilde, k4_tilde]); T = sp.Matrix([[66.60e-2, 0], [0, 244.0e-2], [0, 0], [0, 0]])
        M_inv = M.inv(); ddq = M_inv * (T * u - D * dq - K); fx = sp.Matrix([dq, ddq])
        constants = {'Ka': 6.70e-2, 'Ib': 1.19e-2, 'Jb': 1.78e-2, 'Kb': 2.97e-2, 'Ic': 0.92e-2, 'Jc': 2.30e-2, 'Kc': 2.20e-2, 'Id': 1.48e-2, 'Jd': 2.73e-2, 'Kd': 1.48e-2}
        self.input_vars = [q1, q2, q3, q4, dq1, dq2, dq3, dq4, u1, u2, f1, f2, f3]
        self.fx_with_values = fx.subs(constants)
        self.generated_func = sp.lambdify(self.input_vars, self.fx_with_values, 'numpy')
        print("Physics model generation finished!")
    def __call__(self, t, x, u, f): return self.generated_func(*np.concatenate([x, u, f])).flatten()
    def __getstate__(self): state = self.__dict__.copy(); del state['generated_func']; return state
    def __setstate__(self, state): self.__dict__.update(state); self.generated_func = sp.lambdify(self.input_vars, self.fx_with_values, 'numpy')

def get_or_create_cmg_model():
    cache_file = 'cmg_model_instance.joblib'
    if os.path.exists(cache_file):
        try: model_instance = joblib.load(cache_file); print("Loaded physics model from cache."); return model_instance
        except Exception as e: print(f"Failed to load cached model: {e}. Recreating."); os.remove(cache_file)
    model_instance = CMGPhysicsModel(); joblib.dump(model_instance, cache_file); print(f"New physics model created and saved: {cache_file}"); return model_instance

def get_future_xref(full_ref_trajectory, k, N, state_dim, output_indices, Ts):
    end_idx = k + N + 1
    if end_idx > len(full_ref_trajectory):
        future_refs = full_ref_trajectory[k:, :]
        padding_needed = end_idx - len(full_ref_trajectory)
        padding = np.tile(future_refs[-1, :], (padding_needed, 1))
        future_refs = np.vstack([future_refs, padding])
    else:
        future_refs = full_ref_trajectory[k:end_idx, :]
    X_ref_future = np.zeros((N + 1, state_dim))
    X_ref_future[:, output_indices[0]] = future_refs[:, 0]
    X_ref_future[:, output_indices[1]] = future_refs[:, 1]
    dq_ref = np.diff(future_refs, axis=0, append=future_refs[-1:, :]) / Ts
    X_ref_future[:, 5] = dq_ref[:, 0]
    X_ref_future[:, 6] = dq_ref[:, 1]
    return X_ref_future

# ==============================================================================
# 2. ACADOS OCP SETUP FUNCTION (Corrected Version)
# ==============================================================================

def setup_acados_ocp(scalers, torch_model, Q, R, u_min, u_max, N, Ts, odenn_state_dim, input_dim):
    import os
    import subprocess
    
    # 1. Create OCP object
    ocp = AcadosOcp()

    # 2. Set up the dynamic model
    model = AcadosModel()
    model_name = "cmg_odenn_mpc"
    
    # --- Define CasADi symbolic variables ---
    x_sym = ca.SX.sym('x', odenn_state_dim)
    u_sym = ca.SX.sym('u', input_dim)

    # --- Integrate ODEnn into CasADi graph using L4CasADi ---
    x_mean = ca.DM(scalers['x'].mean_)
    x_std = ca.DM(scalers['x'].scale_)
    u_mean = ca.DM(scalers['u'].mean_)
    u_std = ca.DM(scalers['u'].scale_)

    x_scaled = (x_sym - x_mean) / x_std
    u_scaled = (u_sym - u_mean) / u_std
    xu_scaled = ca.vcat([x_scaled, u_scaled])
    
    # Create L4CasADi model and FORCE compilation
    print("Creating L4CasADi model and generating C code...")
    l4casadi_model = L4CasADi(torch_model.cpu(), name=f'{model_name}_dynamics')
    
    # Force compilation with dummy data
    print("Forcing L4CasADi library compilation...")
    dummy_input = ca.DM.zeros(1, odenn_state_dim + input_dim)
    dummy_output = l4casadi_model(dummy_input)
    print(f"✓ L4CasADi compilation successful")
    print(f"  Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}")
    
    # Use in symbolic graph
    dxdt_scaled = l4casadi_model(xu_scaled.T)
    xdot = dxdt_scaled.T * x_std
    
    # Verify library exists
    shared_lib_dir = os.path.abspath(l4casadi_model.shared_lib_dir)
    shared_lib_name = l4casadi_model.name
    
    print(f"\n=== L4CasADi Library Info ===")
    print(f"Library directory: {shared_lib_dir}")
    print(f"Library name: {shared_lib_name}")
    
    lib_extension = '.dylib' if os.uname().sysname == 'Darwin' else '.so'
    lib_path = os.path.join(shared_lib_dir, f'lib{shared_lib_name}{lib_extension}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"L4CasADi library not found at: {lib_path}")
    
    print(f"✓ Found L4CasADi library at: {lib_path}")
    print(f"=== End Library Info ===\n")
    
    # Assign to Acados model
    model.f_expl_expr = xdot
    model.x = x_sym
    model.u = u_sym
    model.name = model_name
    model.external_shared_lib_dir = shared_lib_dir
    model.external_shared_lib_name = shared_lib_name
    ocp.model = model
    
    # 3. Set dimensions
    ocp.dims.N = N
    
    # 4. Set cost function
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS' 
    ny = odenn_state_dim + input_dim
    ny_e = odenn_state_dim
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q
    ocp.cost.Vx = np.zeros((ny, odenn_state_dim))
    ocp.cost.Vx[:odenn_state_dim, :] = np.eye(odenn_state_dim)
    ocp.cost.Vu = np.zeros((ny, input_dim))
    ocp.cost.Vu[odenn_state_dim:, :] = np.eye(input_dim)
    ocp.cost.Vx_e = np.eye(odenn_state_dim)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # 5. Set constraints
    ocp.constraints.lbu = np.array(u_min)
    ocp.constraints.ubu = np.array(u_max)
    ocp.constraints.idxbu = np.arange(input_dim)
    ocp.constraints.x0 = np.zeros(odenn_state_dim)

    # 6. Set solver options
    ocp.solver_options.tf = N * Ts
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode = 'ROBUST'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.print_level = 0
    ocp.solver_options.model_external_shared_lib_dir = shared_lib_dir
    ocp.solver_options.model_external_shared_lib_name = shared_lib_name
    ocp.code_export_directory = 'c_generated_code'
    
    # 7. Create solver with automatic Makefile patching
    print("Creating Acados OCP solver...")
    json_file = f"{model_name}.json"
    
    try:
        acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
        print("Acados solver created successfully!")
    except OSError as e:
        print(f"Initial compilation failed (expected). Patching Makefile...")
        
        makefile_path = os.path.join(ocp.code_export_directory, 'Makefile')
        
        if os.path.exists(makefile_path):
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()
            
            old_pattern = '-L -l'
            new_pattern = f'-L{shared_lib_dir} -l{shared_lib_name}'
            
            if old_pattern in makefile_content:
                makefile_content = makefile_content.replace(old_pattern, new_pattern)
                
                with open(makefile_path, 'w') as f:
                    f.write(makefile_content)
                
                print(f"✓ Patched Makefile with: {new_pattern}")
                
                print("Rebuilding Acados solver...")
                subprocess.run(['make', 'clean'], cwd=ocp.code_export_directory, 
                             capture_output=True)
                result = subprocess.run(['make', 'ocp_shared_lib'], 
                                      cwd=ocp.code_export_directory,
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✓ Rebuild successful!")
                    acados_solver = AcadosOcpSolver(ocp, json_file=json_file)
                    print("✓ Acados solver created successfully!")
                else:
                    print(f"✗ Rebuild failed: {result.stderr}")
                    raise RuntimeError("Failed to rebuild after patching")
            else:
                raise e
        else:
            raise e
    
    return acados_solver
# ==============================================================================
# 3. MAIN SIMULATION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # --- Setup & Parameters ---
    Ts = 0.01
    sim_time = 50.0
    total_steps = int(sim_time / Ts)
    physics_state_dim = 8
    odenn_state_dim = 7
    input_dim = 2
    DEVICE = "cpu"
    
    # --- Load Reference Trajectory ---
    try:
        mat_data = scipy.io.loadmat('Ref.mat')
        ref_vector_stacked = mat_data.get('Ref', mat_data.get('X_ref')).flatten()
        output_dim_ref = 2
        ref_output_trajectory = ref_vector_stacked.reshape(-1, output_dim_ref)
        print(f"Loaded ref shape: {ref_output_trajectory.shape}")
    except Exception as e:
        print(f"Load Ref—Sine-Cosine.mat failed: {e}")
        raise

    # --- Load Models & Scalers ---
    cmg_physics_model = get_or_create_cmg_model()
    
    trained_dynamics_func = ODEDynamics(x_dim=odenn_state_dim, u_dim=input_dim).to(DEVICE)
    original_state_dict = torch.load("NeuralODE_256.pth", map_location=DEVICE)
    new_state_dict = {key.replace("dynamics_func.", ""): value for key, value in original_state_dict.items()}
    trained_dynamics_func.load_state_dict(new_state_dict)
    trained_dynamics_func.eval()
    print("Successfully loaded original trained model weights.")

    l4casadi_compatible_model = L4CasADiWrapper(trained_dynamics_func, odenn_state_dim, input_dim)

    try:
        scaler_x = joblib.load('scaler_x.joblib')
        scaler_u = joblib.load('scaler_u.joblib')
        scalers = {'x': scaler_x, 'u': scaler_u}
        print("Loaded scalers from training.")
    except FileNotFoundError:
        print("ERROR: Scaler files not found.")
        raise
        
    # --- Setup Acados NMPC Controller ---
    physics_to_nn_indices = [2, 3, 1, 4, 5, 6, 7]
    output_indices_in_nn_state = [0, 1] 
    N_horizon = 30
    R_cost = np.diag([80.0, 40.0])
    Q_cost = np.diag([40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_min_constraint = [-0.5, -0.5]
    u_max_constraint = [0.5, 0.5]
    
    print("\nSetting up Acados OCP solver with L4CasADi...")
    acados_solver = setup_acados_ocp(
        scalers, l4casadi_compatible_model, Q=Q_cost, R=R_cost, 
        u_min=u_min_constraint, u_max=u_max_constraint,
        N=N_horizon, Ts=Ts,
        odenn_state_dim=odenn_state_dim, input_dim=input_dim
    )
    print("Acados solver setup complete.")

    # --- Simulation Setup ---
    x0_phys = np.zeros(physics_state_dim)
    history_x_phys = [x0_phys.copy()]
    x_real_current = x0_phys.copy()
    history_u = []
    f_sim = np.array([0.000187, 0.0118, 0.0027])
    u_guess = np.zeros((N_horizon, input_dim))

    # --- Live Plotting Setup ---
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.tight_layout(pad=3.0)
    ref_time_axis = np.arange(ref_output_trajectory.shape[0]) * Ts
    ref_q3_deg = ref_output_trajectory[:, 0] * 180/np.pi; ref_q4_deg = ref_output_trajectory[:, 1] * 180/np.pi
    axes[0].set_title('Acados NMPC Tracking Performance (Live)'); axes[0].set_ylabel('Angle [deg]'); axes[0].grid(True)
    line_q3_ref, = axes[0].plot(ref_time_axis, ref_q3_deg, 'k-', label='q3_ref', alpha=0.5); line_q4_ref, = axes[0].plot(ref_time_axis, ref_q4_deg, 'b-', label='q4_ref', alpha=0.5)
    line_q3_act, = axes[0].plot([], [], 'r--', label='q3_actual'); line_q4_act, = axes[0].plot([], [], 'g--', label='q4_actual')
    axes[0].legend(); axes[0].set_xlim(0, sim_time)
    axes[1].set_ylabel('Control Input [Nm]'); axes[1].grid(True); axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5); axes[1].axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
    line_u1, = axes[1].step([], [], where='post', label='u1 (tau_1)'); line_u2, = axes[1].step([], [], where='post', label='u2 (tau_2)')
    axes[1].legend(); axes[1].set_xlim(0, sim_time); axes[1].set_ylim(-0.6, 0.6)
    axes[2].set_xlabel('Time [s]'); axes[2].set_ylabel('Angular Velocities [rad/s]'); axes[2].grid(True)
    line_dq3, = axes[2].plot([], [], label='dq3_actual'); line_dq4, = axes[2].plot([], [], label='dq4_actual')
    axes[2].legend(); axes[2].set_xlim(0, sim_time)

    # --- Main Simulation Loop ---
    print("\n--- Starting Acados NMPC Simulation ---")
    start_time = time.time()
    
    solver_times = []
    
    for k in tqdm(range(total_steps)):
        x_nn_current = x_real_current[physics_to_nn_indices]
        
        # Correctly set the initial state constraint for Acados
        acados_solver.set(0, 'lbx', x_nn_current)
        acados_solver.set(0, 'ubx', x_nn_current)
        
        X_ref_future = get_future_xref(ref_output_trajectory, k, N_horizon, odenn_state_dim, output_indices_in_nn_state, Ts)
        
        for j in range(N_horizon):
            y_ref_j = np.concatenate([X_ref_future[j,:], np.zeros(input_dim)])
            acados_solver.set(j, 'yref', y_ref_j)
        y_ref_e = X_ref_future[N_horizon,:]
        acados_solver.set(N_horizon, 'yref', y_ref_e)
        
        for j in range(N_horizon):
            acados_solver.set(j, 'u', u_guess[j,:])

        solve_start_time = time.time()
        status = acados_solver.solve()
        solver_times.append(time.time() - solve_start_time)
        
        if status != 0:
            print(f"Warning: Acados solver returned status {status} at step {k}")

        u_apply = acados_solver.get(0, 'u')
        
        sol = solve_ivp(
            lambda t, x: cmg_physics_model(t, x, u_apply, f_sim),
            [0, Ts], x_real_current, method="RK45", rtol=1e-6, atol=1e-8)
        
        x_real_next = sol.y[:, -1]
        
        history_x_phys.append(x_real_next.copy())
        history_u.append(u_apply.copy())
        x_real_current = x_real_next
        
        u_guess = np.vstack([acados_solver.get(i, 'u') for i in range(1, N_horizon)])
        u_guess = np.vstack([u_guess, u_guess[-1,:]])

        if k % 20 == 0 or k == total_steps - 1:
            current_history_x = np.array(history_x_phys); current_history_u = np.array(history_u)
            current_time_axis = np.arange(current_history_x.shape[0]) * Ts
            q3_deg = current_history_x[:, 2] * 180/np.pi; q4_deg = current_history_x[:, 3] * 180/np.pi
            line_q3_act.set_data(current_time_axis, q3_deg); line_q4_act.set_data(current_time_axis, q4_deg)
            line_u1.set_data(current_time_axis[:-1], current_history_u[:, 0]); line_u2.set_data(current_time_axis[:-1], current_history_u[:, 1])
            line_dq3.set_data(current_time_axis, current_history_x[:, 6]); line_dq4.set_data(current_time_axis, current_history_x[:, 7])
            for ax in axes: ax.relim(); ax.autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events()

    end_time = time.time()
    print(f"\n--- Simulation finished in {end_time - start_time:.2f}s ---")
    print(f"Average Acados solver time: {np.mean(solver_times)*1000:.2f} ms")

    plt.ioff()
    axes[0].set_title('Acados NMPC Tracking Performance (Final Result)')
    plt.show()