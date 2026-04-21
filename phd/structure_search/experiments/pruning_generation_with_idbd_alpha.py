# %% [markdown]
#  # Pruning + Generation with Non-Stationarity
# 
# 
# 
#  Extension of the single-unit pruning notebook that adds:
# 
#  - **Unit generation**: when total connections drop below `CONNECTION_BUDGET`,
# 
#    a new fully-connected hidden unit is created
# 
#  - **Non-stationarity**: every `PRUNES_PER_CHANGE` prune events, one randomly
# 
#    selected subtask gets a new target matrix
# 
# 
# 
#  This lets us observe:
# 
#  1. Whether pruning drives specialization (snowball effect)
# 
#  2. Whether newly generated units get pruned back to specialize
# 
#  3. Whether task changes cause re-specialization
# 
# 
# 
#  ## Setup
# 
#  - 2 tasks, configurable inputs/outputs/hidden
# 
#  - Leaky ReLU, SGD, per-connection utility (4 modes)
# 
#  - Hover over connections for weight + utility values
# 
#  - Per-output and total loss shown per frame
# 
# 

# %%
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from functools import partial
import optax
from datetime import datetime
from enum import IntEnum
import mlflow

from phd.jax_core.optimizers.idbd_jacobian import optax_idbd


class StepUtilityMode(IntEnum):
    """Which utility function is used during training to score each connection."""
    CONTRIBUTION       = 0  # |w| * |activation|
    PROPAGATED         = 1  # fractional propagated (contribution + scale_c)
    SIGNED_SIGNED      = 2  # signed input + signed output
    SIGNED_CONTRIB     = 3  # signed input + contribution output (scaled)
    SHIFTED_SIGNED     = 4  # shifted-signed (mode 4)


class CullUtilityMode(IntEnum):
    """Which utility signal is used when deciding which connection to prune."""
    ORIGINAL             = 0  # step utility (snap_outputs[4/5])
    HEBBIAN              = 1  # auxiliary Hebbian utility (snap_outputs[11/12])
    PROP_TIMES_HEBBIAN   = 2  # step utility × Hebbian utility
    PROP_PLUS_HEBBIAN    = 3  # step utility + Hebbian utility

# from edan_phd_research.jax_core.optimizers.idbd_jacobian import optax_idbd
# optax_idbd


# ---------------------------------------------------------------------------
# MLflow config
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000/"
MLFLOW_EXPERIMENT_NAME = "pruning_generation_idbd_alpha"
# RUN_NAME = "idbd_70steps_seed2133"  # set to None to use auto-generated name
RUN_NAME = "v1_fractional_propagated"  # set to None to use auto-generated name

# ---------------------------------------------------------------------------
# Experiment hyperparameters
# ---------------------------------------------------------------------------
N_INPUTS_PER_TASK = 2
N_OUTPUTS_PER_TASK = 2
# N_HIDDEN_INIT = 1        # starting hidden units
N_HIDDEN_INIT = 10        # starting hidden units
N_HIDDEN_MAX = 10          # max hidden units (buffer size for arrays)
N_TASKS = 2
INPUT_DIM = N_INPUTS_PER_TASK * N_TASKS
OUTPUT_DIM = N_OUTPUTS_PER_TASK * N_TASKS
# ENABLE_GEN = False
# ENABLE_GEN = True # ! doesn't do anything
# USE_HEBBIAN_FOR_CULL = True # ! doesn't do anything
OPTIMIZER_NAME = "idbd"  # options: "idbd", "sgd"
TRAIN_STEPS_PER_PRUNE = 5_000
# TRAIN_STEPS_PER_PRUNE = 50_000
# TRAIN_STEPS_PER_PRUNE = 500
# TRAIN_STEPS_PER_PRUNE = 500
N_SNAPSHOTS_PER_PRUNE = 1
LR = 3e-3
WEIGHT_DECAY = 0.0  # 0 = off, e.g. 1e-4
# TOTAL_PRUNE_STEPS = 200    # how many prune events to run
TOTAL_PRUNE_STEPS = 70    # how many prune events to run
# CONNECTION_BUDGET = N_HIDDEN_INIT * (INPUT_DIM + OUTPUT_DIM) * 2.5  # generate when below this

# CONNECTION_BUDGET = 40
CONNECTION_BUDGET = 80 #is the max I think

# CONNECTION_BUDGET = 14
N_PRUNE_PER_STEP = 1      # connections to remove each prune event (default = 1, same as old behaviour)
PRUNES_PER_CHANGE = 0     # 0 = stationary, >0 = change one subtask every N prunes
RANK1_TARGETS = False
# STEP_UTILITY_MODE = StepUtilityMode.CONTRIBUTION
STEP_UTILITY_MODE = StepUtilityMode.PROPAGATED
CULL_UTILITY_MODE = CullUtilityMode.ORIGINAL
# LEAKY_RELU_NEGATIVE_SLOPE = 0.01 # ! default
LEAKY_RELU_NEGATIVE_SLOPE = 0.1 # ! bit easier to see
# LEAKY_RELU_NEGATIVE_SLOPE = 1.0

SEED = 2133470219 # int(np.random.randint(0, 2**31)) # 1963565763 (correlated hidden units) # 1793333517 #
print(f'Seed: {SEED}')


# %%
def make_target_matrices(key):
    if RANK1_TARGETS:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        a1 = jax.random.rademacher(k1, (N_INPUTS_PER_TASK,)).astype(jnp.float32)
        b1 = jax.random.rademacher(k2, (N_OUTPUTS_PER_TASK,)).astype(jnp.float32)
        a2 = jax.random.rademacher(k3, (N_INPUTS_PER_TASK,)).astype(jnp.float32)
        b2 = jax.random.rademacher(k4, (N_OUTPUTS_PER_TASK,)).astype(jnp.float32)
        A1, A2 = b1[:, None] * a1[None, :], b2[:, None] * a2[None, :]
    else:
        k1, k2 = jax.random.split(key)
        A1 = jax.random.rademacher(k1, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
        A2 = jax.random.rademacher(k2, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
    return A1, A2

def change_one_subtask(A1, A2, key):
    """Randomly pick one subtask and regenerate its target matrix."""
    k1, k2 = jax.random.split(key)
    which_task = int(jax.random.randint(k1, (), 0, N_TASKS))
    if RANK1_TARGETS:
        ka, kb = jax.random.split(k2)
        a = jax.random.rademacher(ka, (N_INPUTS_PER_TASK,)).astype(jnp.float32)
        b = jax.random.rademacher(kb, (N_OUTPUTS_PER_TASK,)).astype(jnp.float32)
        A_new = b[:, None] * a[None, :]
    else:
        A_new = jax.random.rademacher(k2, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
    if which_task == 0:
        return A_new, A2, which_task
    else:
        return A1, A_new, which_task

def sample_data(key, A1, A2):
    k1, k2 = jax.random.split(key)
    x1 = jax.random.normal(k1, (N_INPUTS_PER_TASK,))
    x2 = jax.random.normal(k2, (N_INPUTS_PER_TASK,))
    return jnp.concatenate([x1, x2]), jnp.concatenate([A1 @ x1, A2 @ x2])

def init_params(key, n_hidden):
    k1, k2 = jax.random.split(key)
    W_in = np.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=np.float32)
    W_out = np.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=np.float32)
    W_in[:n_hidden] = np.array(jax.random.normal(k1, (n_hidden, INPUT_DIM)) * jnp.sqrt(2.0 / INPUT_DIM))
    W_out[:, :n_hidden] = np.array(jax.random.normal(k2, (OUTPUT_DIM, n_hidden)) * jnp.sqrt(2.0 / n_hidden))
    M_in = np.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=np.int32)
    M_out = np.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=np.int32)
    M_in[:n_hidden] = 1
    M_out[:, :n_hidden] = 1
    return jnp.array(W_in), jnp.array(W_out), jnp.array(M_in), jnp.array(M_out)

def forward(W_in, W_out, M_in, M_out, x):
    # h = jax.nn.leaky_relu((W_in * M_in) @ x)
    h = jax.nn.leaky_relu((W_in * M_in) @ x, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE)

    y_hat = (W_out * M_out) @ h
    return y_hat, h

def loss_fn(W_in, W_out, M_in, M_out, x, y):
    y_hat, _ = forward(W_in, W_out, M_in, M_out, x)
    return jnp.mean((y_hat - y) ** 2)

def per_output_loss(W_in, W_out, M_in, M_out, x, y):
    y_hat, _ = forward(W_in, W_out, M_in, M_out, x)
    return (y_hat - y) ** 2



# %%


def compute_step_utility_contribution(W_in, W_out, M_in, M_out, x):
    _, h = forward(W_in, W_out, M_in, M_out, x)
    return jnp.abs(x)[None, :] * jnp.abs(W_in), jnp.abs(h)[None, :] * jnp.abs(W_out)

def compute_signed_output_utility(W_out, M_out, h, y, y_hat):
    residual = y - y_hat
    contrib = W_out * M_out * h[None, :]
    return ((residual[:, None] + contrib) ** 2 - (residual ** 2)[:, None]) * M_out

# ! Andrew new!
def compute_step_utility_normalized(W_in, W_out, M_in, M_out, x): # ! v1
    # Raw contribution utility
    U_in_raw, U_out_raw = compute_step_utility_contribution(
        W_in, W_out, M_in, M_out, x
    )

    # --- Normalize input side (per hidden unit) ---
    in_sum = (U_in_raw * M_in).sum(axis=1, keepdims=True)
    U_in_norm = jnp.where(in_sum > 1e-10,
                            U_in_raw / in_sum,
                            0.0)

    # --- Normalize output side (per hidden unit) ---
    out_sum = (U_out_raw * M_out).sum(axis=0, keepdims=True)
    U_out_norm = jnp.where(out_sum > 1e-10,
                            U_out_raw / out_sum,
                            0.0)

    return U_in_norm * M_in, U_out_norm * M_out   

def compute_step_utility_proportional_backprop(W_in, W_out, M_in, M_out, x):
    # Forward pass
    _, h = forward(W_in, W_out, M_in, M_out, x)

    # Raw contribution utilities
    C_in = jnp.abs(x)[None, :] * jnp.abs(W_in) * M_in          # (H, I)
    C_out = jnp.abs(h)[None, :] * jnp.abs(W_out) * M_out       # (O, H)

    # ------------------------------------------------------------
    # Output nodes have utility 1.0
    # ------------------------------------------------------------
    U_out_node = jnp.ones((OUTPUT_DIM,))                       # (O,)

    # ------------------------------------------------------------
    # Hidden -> Output edge utility:
    # split each output node's utility across its incoming edges
    # according to raw contribution share
    # ------------------------------------------------------------
    out_den = jnp.sum(C_out, axis=1, keepdims=True)            # (O, 1)
    out_pct = jnp.where(out_den > 1e-10, C_out / out_den, 0.0) # (O, H)
    U_out = out_pct * U_out_node[:, None] * M_out              # (O, H)

    # ------------------------------------------------------------
    # Hidden node utility:
    # total utility that flows through hidden unit downstream
    # ------------------------------------------------------------
    U_hidden_node = jnp.sum(U_out, axis=0)                     # (H,)

    # ------------------------------------------------------------
    # Input -> Hidden edge utility:
    # split each hidden node's utility across its incoming edges
    # according to raw contribution share
    # ------------------------------------------------------------
    in_den = jnp.sum(C_in, axis=1, keepdims=True)              # (H, 1)
    in_pct = jnp.where(in_den > 1e-10, C_in / in_den, 0.0)     # (H, I)
    U_in = in_pct * U_hidden_node[:, None] * M_in              # (H, I)

    return U_in, U_out

def compute_step_utility_fractional_propagated(W_in, W_out, M_in, M_out, x):
    # Forward pass
    _, h = forward(W_in, W_out, M_in, M_out, x)

    # Raw local contribution magnitudes
    C_in = jnp.abs(x)[None, :] * jnp.abs(W_in) * M_in      # (H, I)
    C_out = jnp.abs(h)[None, :] * jnp.abs(W_out) * M_out   # (O, H)

    # --------------------------------------------------
    # Output node utilities: each target/output has 1.0
    # --------------------------------------------------
    U_output_node = jnp.ones((OUTPUT_DIM,))                # (O,)

    # --------------------------------------------------
    # Hidden -> Output edges:
    # each edge gets a fraction of the output node utility
    # --------------------------------------------------
    out_total = jnp.sum(C_out, axis=1, keepdims=True)      # (O, 1)
    out_frac = jnp.where(out_total > 1e-10, C_out / out_total, 0.0)
    U_out = out_frac * U_output_node[:, None] * M_out      # (O, H)

    # --------------------------------------------------
    # Hidden node utility = sum of outgoing edge utilities
    # --------------------------------------------------
    U_hidden_node = jnp.sum(U_out, axis=0)                 # (H,)

    # --------------------------------------------------
    # Input -> Hidden edges:
    # each edge gets a fraction of the hidden node utility
    # --------------------------------------------------
    in_total = jnp.sum(C_in, axis=1, keepdims=True)        # (H, 1)
    in_frac = jnp.where(in_total > 1e-10, C_in / in_total, 0.0)
    U_in = in_frac * U_hidden_node[:, None] * M_in         # (H, I)

    return U_in, U_out, U_hidden_node


def compute_signed_input_utility(W_in, W_out, M_in, M_out, x, y, y_hat):
    pre_act = (W_in * M_in) @ x
    h = jax.nn.leaky_relu(pre_act, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE)
    contrib_in = W_in * M_in * x[None, :]
    h_without = jax.nn.leaky_relu(pre_act[:, None] - contrib_in, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE)
    delta_h = h[:, None] - h_without
    delta_out = (W_out * M_out)[:, :, None] * delta_h[None, :, :]
    residual = y - y_hat
    err_diff = (residual[:, None, None] + delta_out) ** 2 - (residual ** 2)[:, None, None]
    return err_diff.sum(axis=0) * M_in


def compute_step_utility_shifted_signed(W_in, W_out, M_in, M_out, x, y):
    """Mode 4: shifted signed utility, scaled to error reduced (output) and
    scaled to parent unit utility (input)."""
    y_hat, h = forward(W_in, W_out, M_in, M_out, x)
    error = y - y_hat  # (OUTPUT_DIM,)

    # --- Output layer: shifted signed, scaled to error reduced ---
    c_out = W_out * M_out * h[None, :]  # (OUTPUT_DIM, N_HIDDEN_MAX)
    U_raw_out = (jnp.abs(error[:, None] + c_out) - jnp.abs(error)[:, None])  # (O, H)
    U_shifted_out = U_raw_out - jnp.min(U_raw_out, axis=1, keepdims=True)  # shift per hidden unit
    error_reduced = jnp.abs(y) - jnp.abs(error)  # (OUTPUT_DIM,)
    U_shifted_sum = jnp.sum(U_shifted_out, axis=1)  # (OUTPUT_DIM,)
    scale_out = jnp.where(jnp.abs(U_shifted_sum) > 1e-10, error_reduced / U_shifted_sum, 1.0)
    U_out = U_shifted_out * scale_out[:, None]  # (OUTPUT_DIM, N_HIDDEN_MAX)
    U_out = U_out * M_out

    # Per-unit utility = sum of output utilities for that unit
    unit_utility = (U_out * M_out).sum(axis=0)  # (N_HIDDEN_MAX,)

    # --- Input layer: shifted signed, scaled to parent unit utility ---
    pre_act = (W_in * M_in) @ x  # (N_HIDDEN_MAX,)
    c_in = W_in * M_in * x[None, :]  # (N_HIDDEN_MAX, INPUT_DIM)
    h_without = jax.nn.leaky_relu(pre_act[:, None] - c_in, negative_slope=LEAKY_RELU_NEGATIVE_SLOPE)  # (H, I)
    delta_h = h[:, None] - h_without  # (H, I) — change in activation when removing input

    # Signed utility per input: how much does removing it change the unit's output contribution
    # Use delta_h as proxy for raw utility (bigger delta = more important input)
    U_raw_in = jnp.abs(delta_h)  # (N_HIDDEN_MAX, INPUT_DIM)
    U_shifted_in = U_raw_in - jnp.min(U_raw_in, axis=1, keepdims=True)  # shift per unit
    U_shifted_in_sum = jnp.sum(U_shifted_in * M_in, axis=1)  # (N_HIDDEN_MAX,)
    scale_in = jnp.where(jnp.abs(U_shifted_in_sum) > 1e-10,
                          unit_utility / U_shifted_in_sum, 1.0)  # (N_HIDDEN_MAX,)
    U_in = U_shifted_in * scale_in[:, None]  # (N_HIDDEN_MAX, INPUT_DIM)
    U_in = U_in * M_in

    return U_in, U_out

def compute_step_utility(W_in, W_out, M_in, M_out, x, y, utility_mode):
    y_hat, h = forward(W_in, W_out, M_in, M_out, x)
    # U_in_c, U_out_c = compute_step_utility_contribution(W_in, W_out, M_in, M_out, x)
    U_in_c, U_out_c, nothing = compute_step_utility_fractional_propagated(W_in, W_out, M_in, M_out, x)
    U_out_s = compute_signed_output_utility(W_out, M_out, h, y, y_hat)
    U_in_s = compute_signed_input_utility(W_in, W_out, M_in, M_out, x, y, y_hat)
    U_out = jnp.where(utility_mode <= 1, U_out_c, U_out_s)
    out_total = (U_out * M_out).sum(axis=0)
    in_total_c = (U_in_c * M_in).sum(axis=1)
    scale_c = out_total / jnp.maximum(in_total_c, 1e-10)
    in_total_s = (U_in_s * M_in).sum(axis=1)
    scale_s = out_total / jnp.maximum(jnp.abs(in_total_s), 1e-10)
    U_in_ss, U_out_ss = compute_step_utility_shifted_signed(W_in, W_out, M_in, M_out, x, y)
    U_in = jnp.where(utility_mode == 0, U_in_c,
           jnp.where(utility_mode == 1, U_in_c * scale_c[:, None],
           jnp.where(utility_mode == 2, U_in_s * scale_s[:, None],
           jnp.where(utility_mode == 3, U_in_c * scale_c[:, None],
                                         U_in_ss))))
    U_out = jnp.where(utility_mode == 4, U_out_ss, U_out)
    return U_in * M_in, U_out * M_out

def _predict_only(W_in, W_out, M_in, M_out, x):
    y_hat, _ = forward(W_in, W_out, M_in, M_out, x)
    return y_hat

def extract_alpha_from_opt_state(opt_state, params):
    """Return per-parameter step-sizes if the optimizer exposes IDBD beta state."""
    if hasattr(opt_state, "beta"):
        return {
            # "W_in": jnp.exp(opt_state.beta["W_in"]),
            # "W_out": jnp.exp(opt_state.beta["W_out"]),

            "W_in": jax.nn.sigmoid(opt_state.beta["W_in"]),
            "W_out": jax.nn.sigmoid(opt_state.beta["W_out"]),
        }
    return {
        "W_in": jnp.zeros_like(params["W_in"]),
        "W_out": jnp.zeros_like(params["W_out"]),
    }

def _train_step(W_in, W_out, M_in, M_out, x, y, optimizer, opt_state):
    params = {
        "W_in": W_in,
        "W_out": W_out,
    }

    def loss_from_params(params):
        return loss_fn(params["W_in"], params["W_out"], M_in, M_out, x, y)

    def pred_from_params(params):
        return _predict_only(params["W_in"], params["W_out"], M_in, M_out, x)

    loss_grads = jax.grad(loss_from_params)(params)
    prediction_grads = jax.jacrev(pred_from_params)(params)

    optimizer_payload = (loss_grads, prediction_grads) if hasattr(opt_state, "beta") else loss_grads
    updates, opt_state = optimizer.update(
        optimizer_payload,
        opt_state,
        params,
    )

    # Freeze pruned connections
    updates = {
        "W_in": updates["W_in"] * M_in,
        "W_out": updates["W_out"] * M_out,
    }

    new_params = optax.apply_updates(params, updates)

    # Keep masked weights exactly zero
    W_in = new_params["W_in"] * M_in
    W_out = new_params["W_out"] * M_out

    return W_in, W_out, opt_state

# ! REGULAR TRIAN STEP
# def _train_step(W_in, W_out, M_in, M_out, x, y):
#     g_in, g_out = jax.grad(loss_fn, argnums=(0, 1))(W_in, W_out, M_in, M_out, x, y)
#     W_in = W_in - LR * (g_in + WEIGHT_DECAY * W_in) * M_in
#     W_out = W_out - LR * (g_out + WEIGHT_DECAY * W_out) * M_out
#     return W_in, W_out

def find_lowest_utility(U_in, U_out, M_in, M_out):
    u_in_flat = jnp.where(M_in.reshape(-1) == 1, U_in.reshape(-1), jnp.inf)
    u_out_flat = jnp.where(M_out.reshape(-1) == 1, U_out.reshape(-1), jnp.inf)
    all_u = jnp.concatenate([u_in_flat, u_out_flat])
    idx = jnp.argmin(all_u)
    n_in = M_in.size
    return idx >= n_in, jnp.where(idx >= n_in, idx - n_in, idx)

def prune_connection(M_in, M_out, is_output, local_idx):
    M_in_f, M_out_f = M_in.reshape(-1), M_out.reshape(-1)
    in_idx = jnp.where(is_output, 0, local_idx)
    out_idx = jnp.where(is_output, local_idx, 0)
    in_val = jnp.where(is_output, M_in_f[in_idx], 0)
    out_val = jnp.where(is_output, 0, M_out_f[out_idx])
    return M_in_f.at[in_idx].set(in_val).reshape(M_in.shape), M_out_f.at[out_idx].set(out_val).reshape(M_out.shape)



# %%

def generate_unit_jax(W_in, W_out, M_in, M_out, key):
    """Add a fully-connected unit in the first inactive slot (pure JAX)."""
    active = M_in.sum(axis=1) + M_out.sum(axis=0)  # (N_HIDDEN_MAX,)
    # First inactive slot: set inactive to large index, take argmin
    slot = jnp.argmin(jnp.where(active == 0, jnp.arange(N_HIDDEN_MAX), N_HIDDEN_MAX))
    has_slot = jnp.any(active == 0)
    k1, k2 = jax.random.split(key)
    limit = jnp.sqrt(3.0 / INPUT_DIM)
    new_w_in = jax.random.uniform(k1, (INPUT_DIM,), minval=-limit, maxval=limit)
    new_w_out = jnp.zeros((OUTPUT_DIM,))
    W_in = jnp.where(has_slot, W_in.at[slot].set(new_w_in), W_in)
    W_out = jnp.where(has_slot, W_out.at[:, slot].set(new_w_out), W_out)
    M_in = jnp.where(has_slot, M_in.at[slot].set(1), M_in)
    M_out = jnp.where(has_slot, M_out.at[:, slot].set(1), M_out)
    return W_in, W_out, M_in, M_out, slot, has_slot

def cleanup_dead_units(M_in, M_out):
    """If a unit has no outgoing connections, zero its incoming mask."""
    has_outgoing = M_out.sum(axis=0) > 0  # (N_HIDDEN_MAX,)
    M_in = M_in * has_outgoing[:, None]
    return M_in

def change_one_subtask_jax(A1, A2, key):
    """Pure JAX version of change_one_subtask."""
    k1, k2 = jax.random.split(key)
    which_task = jax.random.randint(k1, (), 0, N_TASKS)
    if RANK1_TARGETS:
        ka, kb = jax.random.split(k2)
        a = jax.random.rademacher(ka, (N_INPUTS_PER_TASK,)).astype(jnp.float32)
        b = jax.random.rademacher(kb, (N_OUTPUTS_PER_TASK,)).astype(jnp.float32)
        A_new = b[:, None] * a[None, :]
    else:
        A_new = jax.random.rademacher(k2, (N_OUTPUTS_PER_TASK, N_INPUTS_PER_TASK)).astype(jnp.float32)
    A1_new = jnp.where(which_task == 0, A_new, A1)
    A2_new = jnp.where(which_task == 1, A_new, A2)
    return A1_new, A2_new, which_task

@partial(jax.jit, static_argnames=("sub_block", "optimizer"))
def train_sub_block(W_in, W_out, M_in, M_out, A1, A2, utility_mode, opt_state, rng, sub_block, optimizer):
    keys = jax.random.split(rng, sub_block)
    indices = jnp.arange(sub_block)
    def body(carry, ki):
        key, idx = ki
        W_in, W_out, opt_state, U_in_sum, U_out_sum, loss_sum, alpha_in_sum, alpha_out_sum, A_in_sum, A_out_sum = carry
        x, y = sample_data(key, A1, A2)
        W_in, W_out, opt_state = _train_step(W_in, W_out, M_in, M_out, x, y, optimizer, opt_state)
        U_in_s, U_out_s = compute_step_utility(W_in, W_out, M_in, M_out, x, y, utility_mode)
        # Auxiliary Hebbian utility: post-synaptic (hidden) activation × connection utility
        _, h = forward(W_in, W_out, M_in, M_out, x)
        A_in_s  = jnp.abs(h)[:, None] * U_in_s * M_in   # (H, I)
        A_out_s = jnp.abs(h)[None, :] * U_out_s * M_out  # (O, H)
        alpha = extract_alpha_from_opt_state(opt_state, {"W_in": W_in, "W_out": W_out})
        pol = per_output_loss(W_in, W_out, M_in, M_out, x, y)
        return (
            W_in, W_out, opt_state,
            U_in_sum + U_in_s * M_in,
            U_out_sum + U_out_s * M_out,
            loss_sum + pol,
            alpha_in_sum + alpha["W_in"] * M_in,
            alpha_out_sum + alpha["W_out"] * M_out,
            A_in_sum + A_in_s,
            A_out_sum + A_out_s,
        ), None
    init = (
        W_in, W_out, opt_state,
        jnp.zeros_like(W_in),
        jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX)),
        jnp.zeros(OUTPUT_DIM),
        jnp.zeros_like(W_in),
        jnp.zeros_like(W_out),
        jnp.zeros_like(W_in),                  # A_in_sum
        jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX)),  # A_out_sum
    )
    (W_in, W_out, opt_state, U_in_sum, U_out_sum, loss_sum,
     alpha_in_sum, alpha_out_sum, A_in_sum, A_out_sum), _ = jax.lax.scan(body, init, (keys, indices))
    n = sub_block
    return (W_in, W_out, opt_state,
            (U_in_sum/n)*M_in, (U_out_sum/n)*M_out, loss_sum/n,
            (alpha_in_sum/n)*M_in, (alpha_out_sum/n)*M_out,
            (A_in_sum/n)*M_in, (A_out_sum/n)*M_out)

@partial(jax.jit, static_argnames=("sub_block", "n_snapshots", "total_prune_steps",
                                    "prunes_per_change", "connection_budget",
                                    "n_prune_per_step", "cull_utility_mode", "optimizer"))
def run_experiment_jit(rng, A1_init, A2_init, W_in_init, W_out_init, M_in_init, M_out_init,
                       utility_mode, opt_state, sub_block, n_snapshots, total_prune_steps,
                       prunes_per_change, connection_budget, n_prune_per_step,
                       cull_utility_mode, optimizer):

    def snapshot_block(carry, _snap_idx):
        W_in, W_out, M_in, M_out, A1, A2, opt_state, rng = carry
        rng, k = jax.random.split(rng)
        (W_in, W_out, opt_state, U_in, U_out, per_out_loss,
         alpha_in, alpha_out, A_in, A_out) = train_sub_block(
            W_in, W_out, M_in, M_out, A1, A2, utility_mode, opt_state, k, sub_block, optimizer)
        is_out, loc_idx = find_lowest_utility(U_in, U_out, M_in, M_out)
        snap = (W_in, W_out, M_in, M_out, U_in, U_out, per_out_loss, alpha_in, alpha_out, is_out, loc_idx, A_in, A_out)
        return (W_in, W_out, M_in, M_out, A1, A2, opt_state, rng), snap

    def prune_cycle(carry, prune_idx):
        W_in, W_out, M_in, M_out, A1, A2, opt_state, rng = carry

        # Non-stationarity
        should_change = (prunes_per_change > 0) & (prune_idx > 0) & (prune_idx % prunes_per_change == 0)
        rng, change_key = jax.random.split(rng)
        A1_new, A2_new, which_task = change_one_subtask_jax(A1, A2, change_key)
        A1 = jnp.where(should_change, A1_new, A1)
        A2 = jnp.where(should_change, A2_new, A2)

        # Collect snapshots
        snap_carry = (W_in, W_out, M_in, M_out, A1, A2, opt_state, rng)
        snap_carry, snap_outputs = jax.lax.scan(snapshot_block, snap_carry, jnp.arange(n_snapshots))
        W_in, W_out, M_in, M_out, A1, A2, opt_state, rng = snap_carry
        # snap_outputs: tuple of arrays each (n_snapshots, ...)

        # Get final utilities for pruning decision
        _U_in  = snap_outputs[4][-1]   # step utility (last snapshot)
        _U_out = snap_outputs[5][-1]
        _A_in  = snap_outputs[11][-1]  # Hebbian utility (last snapshot)
        _A_out = snap_outputs[12][-1]
        if cull_utility_mode == CullUtilityMode.HEBBIAN:
            U_in_final, U_out_final = _A_in, _A_out
        elif cull_utility_mode == CullUtilityMode.PROP_TIMES_HEBBIAN:
            U_in_final, U_out_final = _U_in * _A_in, _U_out * _A_out
        elif cull_utility_mode == CullUtilityMode.PROP_PLUS_HEBBIAN:
            U_in_final, U_out_final = _U_in + _A_in, _U_out + _A_out
        else:  # CullUtilityMode.ORIGINAL
            U_in_final, U_out_final = _U_in, _U_out

        # Prune n_prune_per_step connections, each time picking the lowest utility
        def prune_one(masks, _):
            M_in_p, M_out_p = masks
            is_out, loc_idx = find_lowest_utility(U_in_final, U_out_final, M_in_p, M_out_p)
            M_in_p, M_out_p = prune_connection(M_in_p, M_out_p, is_out, loc_idx)
            M_in_p = cleanup_dead_units(M_in_p, M_out_p)
            return (M_in_p, M_out_p), None

        (M_in, M_out), _ = jax.lax.scan(prune_one, (M_in, M_out), None, length=n_prune_per_step)

        n_conns = M_in.sum() + M_out.sum()

        # Generate if below budget
        # should_gen = n_conns < connection_budget
        # should_gen = True
        should_gen = False
        rng, gen_key = jax.random.split(rng)
        W_in_g, W_out_g, M_in_g, M_out_g, gen_slot, has_slot = generate_unit_jax(
            W_in, W_out, M_in, M_out, gen_key)
        did_gen = should_gen & has_slot
        W_in = jnp.where(did_gen, W_in_g, W_in)
        W_out = jnp.where(did_gen, W_out_g, W_out)
        M_in = jnp.where(did_gen, M_in_g, M_in)
        M_out = jnp.where(did_gen, M_out_g, M_out)

        # prune_info = (is_out, loc_idx, did_gen, gen_slot, should_change, which_task)
        prune_info = (0, 0, did_gen, gen_slot, should_change, which_task)
        return (W_in, W_out, M_in, M_out, A1, A2, opt_state, rng), (snap_outputs, prune_info)

    init_carry = (W_in_init, W_out_init, M_in_init, M_out_init, A1_init, A2_init, opt_state, rng)
    _, (all_snaps, all_prune_info) = jax.lax.scan(
        prune_cycle, init_carry, jnp.arange(total_prune_steps))
    return all_snaps, all_prune_info

def run_experiment():
    jax_rng = jax.random.key(SEED)
    jax_rng, k1, k2 = jax.random.split(jax_rng, 3)
    A1, A2 = make_target_matrices(k1)
    W_in, W_out, M_in, M_out = init_params(k2, N_HIDDEN_INIT)
    sub_block = TRAIN_STEPS_PER_PRUNE // N_SNAPSHOTS_PER_PRUNE
    utility_mode = jnp.array(int(STEP_UTILITY_MODE))
    if OPTIMIZER_NAME == "idbd":
        optimizer = optax_idbd(
            # meta_lr=0.005,
            meta_lr=1e-3,
            # meta_lr=0.1,
            # meta_lr=0.0000,
            init_lr=1e-6,
            weight_decay=WEIGHT_DECAY,
            autostep=False,
            version="prediction_grads",
            rms_style_beta_update = True,
            # rms_style_beta_update_beta_rho = 0.0,
            rms_style_beta_update_beta_rho = 0.9,
        )
        # optimizer = optax_idbd(
        #     meta_lr=1e-3, 
        #     init_lr=1e-6, 
        #     base_optimizer='adamw',
        #     base_optimizer_params={'beta1': 0.9, 'beta2': 0.999}, 
        #     weight_decay=WEIGHT_DECAY,
        #     autostep=False,
        #     version="prediction_grads",
        #     rms_style_beta_update=True,
        #     rms_style_beta_update_beta_rho=0.9,
        #     )
    elif OPTIMIZER_NAME == "sgd":
        optimizer = optax.sgd(LR)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")

    params = {"W_in": W_in, "W_out": W_out}
    opt_state = optimizer.init(params)

    print(f"Compiling and running ({TOTAL_PRUNE_STEPS} prune steps, "
          f"{N_SNAPSHOTS_PER_PRUNE} snapshots each, {N_PRUNE_PER_STEP} prunes/step)...")
    all_snaps, all_prune_info = run_experiment_jit(
        jax_rng, A1, A2, W_in, W_out, M_in, M_out,
        utility_mode, opt_state, sub_block, N_SNAPSHOTS_PER_PRUNE, TOTAL_PRUNE_STEPS,
        PRUNES_PER_CHANGE, CONNECTION_BUDGET, N_PRUNE_PER_STEP,
        int(CULL_UTILITY_MODE), optimizer)

    # all_snaps: (W_in, W_out, M_in, M_out, U_in, U_out, per_out_loss, alpha_in, alpha_out, is_out, loc_idx, A_in, A_out)
    # each shape: (TOTAL_PRUNE_STEPS, N_SNAPSHOTS_PER_PRUNE, ...)
    (s_W_in, s_W_out, s_M_in, s_M_out, s_U_in, s_U_out,
     s_loss, s_alpha_in, s_alpha_out, s_is_out, s_loc_idx,
     s_A_in, s_A_out) = [np.array(x) for x in all_snaps]
    (p_is_out, p_loc_idx, p_did_gen, p_gen_slot,
     p_changed, p_which_task) = [np.array(x) for x in all_prune_info]

    snapshots = []
    for pi in range(TOTAL_PRUNE_STEPS):
        for si in range(N_SNAPSHOTS_PER_PRUNE):
            is_last = (si == N_SNAPSHOTS_PER_PRUNE - 1)
            event = None
            if is_last:
                if p_did_gen[pi]:
                    event = f"gen h{p_gen_slot[pi]}"
            if si == 0 and p_changed[pi]:
                event = (event + ", " if event else "") + f"task {p_which_task[pi]} changed"

            snapshots.append({
                "W_in": s_W_in[pi, si], "W_out": s_W_out[pi, si],
                "M_in": s_M_in[pi, si], "M_out": s_M_out[pi, si],
                "U_in": s_U_in[pi, si], "U_out": s_U_out[pi, si],
                "alpha_in": s_alpha_in[pi, si], "alpha_out": s_alpha_out[pi, si],
                "A_in": s_A_in[pi, si], "A_out": s_A_out[pi, si],
                "per_output_loss": s_loss[pi, si],
                "total_loss": float(s_loss[pi, si].mean()),
                "prune_step": pi, "sub_step": si,
                "is_prune_frame": is_last,
                "pruned_is_output": bool(s_is_out[pi, si]) if is_last else None,
                "pruned_local_idx": int(s_loc_idx[pi, si]) if is_last else None,
                "event": event,
            })

        # Print progress
        n_conns = int(s_M_in[pi, -1].sum() + s_M_out[pi, -1].sum())
        ct = "out" if p_is_out[pi] else "in"
        extra = ""
        if p_did_gen[pi]: extra += f" -> gen h{p_gen_slot[pi]}"
        if p_changed[pi]: extra += f" [task {p_which_task[pi]} changed]"
        print(f"  Prune {pi}: {ct}[{p_loc_idx[pi]}] ({n_conns} conns){extra}")

    return snapshots

print(f"Budget: {CONNECTION_BUDGET} conns, Non-stationary: {PRUNES_PER_CHANGE > 0}, Prune/step: {N_PRUNE_PER_STEP}")
snapshots = run_experiment()
print(f"Total frames: {len(snapshots)}")


# %% [markdown]
#  ## Network Diagram
# 
# 
# 
#  Hover over connections for weight + utility. Red = about to be pruned. Per-output
# 
#  loss shown next to outputs. Events (generation, task changes) noted in the title.
# 
# 

# %%
def plot_diagram(snapshots, color_mode='utility'):
    """color_mode: 'utility' | 'hebbian'  — controls edge colour scale."""
    n_active_units = max(int(s['M_in'].any(axis=1).sum()) for s in snapshots)
    input_y = np.linspace(0, 1, INPUT_DIM)
    hidden_y = np.array([0.5]) if n_active_units <= 1 else np.linspace(0.1, 0.9, N_HIDDEN_MAX)
    output_y = np.linspace(0, 1, OUTPUT_DIM)

    def _color_arrays(snap):
        if color_mode == 'hebbian':
            return snap['A_in'], snap['A_out']
        return snap['U_in'], snap['U_out']

    all_utils = []
    for s in snapshots:
        c_in, c_out = _color_arrays(s)
        all_utils.extend((c_in * s['M_in'])[s['M_in'] > 0].tolist())
        all_utils.extend((c_out * s['M_out'])[s['M_out'] > 0].tolist())
    vmin, vmax = 0, np.percentile(all_utils, 95) if all_utils else 1

    frames = []
    for si, snap in enumerate(snapshots):
        M_in_s, M_out_s = snap['M_in'], snap['M_out']
        U_in_s, U_out_s = snap['U_in'], snap['U_out']
        C_in_s, C_out_s = _color_arrays(snap)
        alpha_in_s = snap.get('alpha_in', np.zeros_like(snap['W_in']))
        alpha_out_s = snap.get('alpha_out', np.zeros_like(snap['W_out']))
        A_in_s = snap.get('A_in', np.zeros_like(snap['W_in']))
        A_out_s = snap.get('A_out', np.zeros_like(snap['W_out']))
        W_in_s, W_out_s = snap['W_in'], snap['W_out']
        pruned_is_out = snap['pruned_is_output']
        pruned_idx = snap['pruned_local_idx']

        shapes, hover_x, hover_y, hover_text = [], [], [], []

        for k in range(N_HIDDEN_MAX):
            for i in range(INPUT_DIM):
                if not M_in_s[k, i]: continue
                c = float(C_in_s[k, i])
                u, w = float(U_in_s[k, i]), float(W_in_s[k, i])
                a_hebb = float(A_in_s[k, i])
                alpha = float(alpha_in_s[k, i])
                norm_u = min(max((c - vmin) / max(vmax - vmin, 1e-10), 0), 1)
                color = f'rgba({int(255*(1-norm_u))}, {int(100*norm_u)}, {int(255*norm_u)}, 0.7)'
                width = 1.5 + 3 * norm_u
                is_pruned = (pruned_is_out is not None and not pruned_is_out
                             and pruned_idx == k * INPUT_DIM + i)
                if is_pruned: color, width = 'rgba(255, 0, 0, 0.9)', 4
                shapes.append(dict(type='line', x0=0, y0=float(input_y[i]),
                                   x1=1, y1=float(hidden_y[k]),
                                   line=dict(color=color, width=width)))
                mx = 0.5; my = 0.5 * (float(input_y[i]) + float(hidden_y[k]))
                hover_x.append(mx); hover_y.append(my)
                hover_text.append(f'in[{k},{i}] w={w:.4f} u={u:.4f} hebb={a_hebb:.4f} alpha={alpha:.4e}')

        for j in range(OUTPUT_DIM):
            for k in range(N_HIDDEN_MAX):
                if not M_out_s[j, k]: continue
                c = float(C_out_s[j, k])
                u, w = float(U_out_s[j, k]), float(W_out_s[j, k])
                a_hebb = float(A_out_s[j, k])
                alpha = float(alpha_out_s[j, k])
                norm_u = min(max((c - vmin) / max(vmax - vmin, 1e-10), 0), 1)
                color = f'rgba({int(255*(1-norm_u))}, {int(100*norm_u)}, {int(255*norm_u)}, 0.7)'
                width = 1.5 + 3 * norm_u
                is_pruned = (pruned_is_out is not None and pruned_is_out
                             and pruned_idx == j * N_HIDDEN_MAX + k)
                if is_pruned: color, width = 'rgba(255, 0, 0, 0.9)', 4
                shapes.append(dict(type='line', x0=1, y0=float(hidden_y[k]),
                                   x1=2, y1=float(output_y[j]),
                                   line=dict(color=color, width=width)))
                mx = 1.5; my = 0.5 * (float(hidden_y[k]) + float(output_y[j]))
                hover_x.append(mx); hover_y.append(my)
                hover_text.append(f'out[{j},{k}] w={w:.4f} u={u:.4f} hebb={a_hebb:.4f} alpha={alpha:.4e}')

        loss_annots = []
        pol = snap['per_output_loss']
        for j in range(OUTPUT_DIM):
            task = j // N_OUTPUTS_PER_TASK
            c = '#4a90d9' if task == 0 else '#d94a4a'
            loss_annots.append(dict(x=2.4, y=float(output_y[j]), text=f'L={pol[j]:.3f}',
                                    showarrow=False, font=dict(size=9, color=c)))
        loss_annots.append(dict(x=1.0, y=-0.15, text=f'Total loss: {snap["total_loss"]:.4f}',
                                showarrow=False, font=dict(size=11, color='black')))
        if snap.get('event'):
            loss_annots.append(dict(x=1.0, y=1.12, text=snap['event'],
                                    showarrow=False, font=dict(size=11, color='green')))

        label = f'P{snap["prune_step"]}'
        label += ' [PRUNE]' if snap['is_prune_frame'] else f'.{snap["sub_step"]+1}'
        frames.append((shapes, hover_x, hover_y, hover_text, loss_annots, label))

    fig = go.Figure()
    colors_in = ['#4a90d9'] * N_INPUTS_PER_TASK + ['#d94a4a'] * N_INPUTS_PER_TASK
    colors_out = ['#4a90d9'] * N_OUTPUTS_PER_TASK + ['#d94a4a'] * N_OUTPUTS_PER_TASK
    active_hidden = sorted(set(
        k for s in snapshots for k in range(N_HIDDEN_MAX)
        if s['M_in'][k].any() or s['M_out'][:, k].any()))
    hidden_colors = ['#666'] * N_HIDDEN_MAX

    fig.add_trace(go.Scatter(x=[0]*INPUT_DIM, y=input_y.tolist(), mode='markers+text',
        marker=dict(size=12, color=colors_in),
        text=[f'x{i}' for i in range(INPUT_DIM)],
        textposition='middle left', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[1]*N_HIDDEN_MAX, y=hidden_y.tolist(), mode='markers+text',
        marker=dict(size=14, color=hidden_colors),
        text=[f'h{k}' for k in range(N_HIDDEN_MAX)],
        textposition='top center', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[2]*OUTPUT_DIM, y=output_y.tolist(), mode='markers+text',
        marker=dict(size=12, color=colors_out),
        text=[f'y{j}' for j in range(OUTPUT_DIM)],
        textposition='middle right', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=frames[0][1], y=frames[0][2], mode='markers',
        marker=dict(size=15, opacity=0), text=frames[0][3],
        hoverinfo='text', showlegend=False))

    mid_in = (input_y[N_INPUTS_PER_TASK-1] + input_y[N_INPUTS_PER_TASK]) / 2
    fig.add_hline(y=float(mid_in), line_dash='dot', line_color='gray', opacity=0.5)

    base_annots = [
        dict(x=0, y=1.08, text='Task 1', showarrow=False, font=dict(color='#4a90d9', size=12)),
        dict(x=0, y=-0.08, text='Task 2', showarrow=False, font=dict(color='#d94a4a', size=12)),
        dict(x=2, y=1.08, text='Task 1', showarrow=False, font=dict(color='#4a90d9', size=12)),
        dict(x=2, y=-0.08, text='Task 2', showarrow=False, font=dict(color='#d94a4a', size=12)),
    ]
    fig.update_layout(shapes=frames[0][0], annotations=base_annots + frames[0][4])

    node_x = [[0]*INPUT_DIM, [1]*N_HIDDEN_MAX, [2]*OUTPUT_DIM]
    node_y = [input_y.tolist(), hidden_y.tolist(), output_y.tolist()]
    node_text = [[f'x{i}' for i in range(INPUT_DIM)],
                 [f'h{k}' for k in range(N_HIDDEN_MAX)],
                 [f'y{j}' for j in range(OUTPUT_DIM)]]

    steps = [dict(method='update',
                  args=[{'x': node_x + [hx], 'y': node_y + [hy], 'text': node_text + [ht]},
                        {'shapes': sh, 'annotations': base_annots + la}],
                  label=lbl)
             for sh, hx, hy, ht, la, lbl in frames]

    color_label = 'Hebbian Utility' if color_mode == 'hebbian' else 'Utility'
    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix='Step: '), steps=steps, pad=dict(t=60))],
        title=f'Pruning + Generation  [{color_label}]',
        xaxis=dict(range=[-0.3, 2.7], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.25, 1.2], showgrid=False, zeroline=False, showticklabels=False),
        height=600, width=950)
    return fig

fig_utility = plot_diagram(snapshots, color_mode='utility')
fig_hebbian = plot_diagram(snapshots, color_mode='hebbian')

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path_all_utility = f"./plots/plot_all_utility_{ts}.html"
path_all_hebbian = f"./plots/plot_all_hebbian_{ts}.html"
fig_utility.write_html(path_all_utility)
fig_hebbian.write_html(path_all_hebbian)
# fig_utility.show()


# %%
def plot_diagram_per_output(snapshots, color_mode='hebbian'):
    """Like plot_diagram but split into one panel per output, showing only
    the connections that feed into that output.

    color_mode: 'utility' | 'hebbian'
      - 'utility'  : edges coloured by per-output fractional propagated utility
      - 'hebbian'  : edges coloured by per-output auxiliary Hebbian utility
    """
    n_active_units = max(int(s['M_in'].any(axis=1).sum()) for s in snapshots)
    input_y  = np.linspace(0, 1, INPUT_DIM)
    hidden_y = np.array([0.5]) if n_active_units <= 1 else np.linspace(0.1, 0.9, N_HIDDEN_MAX)
    out_y_pos = 0.5

    def _per_output_u_in(u_in, u_out):
        """(H,I),(O,H) -> (O,H,I): apportion u_in to each output via u_out fractions."""
        u_hidden = u_out.sum(axis=0)  # (H,)
        return u_in[None, :, :] * u_out[:, :, None] / np.maximum(u_hidden[None, :, None], 1e-10)

    all_utils = []
    for s in snapshots:
        u_out = s['U_out'] * s['M_out']
        mask_in = s['M_in'] > 0
        src_in  = s['A_in']  if color_mode == 'hebbian' else s['U_in']
        src_out = s['A_out'] if color_mode == 'hebbian' else u_out
        u_in_po = _per_output_u_in(src_in * s['M_in'], u_out)
        all_utils.extend(u_in_po[:, mask_in].flatten().tolist())
        all_utils.extend((src_out * s['M_out'])[s['M_out'] > 0].tolist())
    vmin, vmax = 0, np.percentile(all_utils, 95) if all_utils else 1

    PANEL_W = 3.5

    frames = []
    for snap in snapshots:
        M_in_s, M_out_s = snap['M_in'], snap['M_out']
        U_in_s, U_out_s = snap['U_in'], snap['U_out']
        A_in_s  = snap.get('A_in',  np.zeros_like(snap['W_in']))
        A_out_s = snap.get('A_out', np.zeros_like(snap['W_out']))
        u_out_masked = U_out_s * M_out_s

        # Per-output color arrays for input→hidden edges: (O, H, I)
        src_in = A_in_s if color_mode == 'hebbian' else U_in_s
        C_in_po_s = _per_output_u_in(src_in * M_in_s, u_out_masked)

        # Always compute both for hover text
        U_in_po_s = _per_output_u_in(U_in_s * M_in_s, u_out_masked)
        A_in_po_s = _per_output_u_in(A_in_s * M_in_s, u_out_masked)

        alpha_in_s  = snap.get('alpha_in',  np.zeros_like(snap['W_in']))
        alpha_out_s = snap.get('alpha_out', np.zeros_like(snap['W_out']))
        W_in_s, W_out_s = snap['W_in'], snap['W_out']
        pruned_is_out = snap['pruned_is_output']
        pruned_idx    = snap['pruned_local_idx']

        shapes, hover_x, hover_y, hover_text = [], [], [], []
        panel_node_x, panel_node_y = [], []

        for j in range(OUTPUT_DIM):
            x0 = j * PANEL_W
            connected_hidden = {k for k in range(N_HIDDEN_MAX) if M_out_s[j, k]}
            connected_inputs  = {i for k in connected_hidden
                                   for i in range(INPUT_DIM) if M_in_s[k, i]}

            panel_node_x.append([x0   if i in connected_inputs  else None for i in range(INPUT_DIM)])
            panel_node_x.append([x0+1 if k in connected_hidden else None for k in range(N_HIDDEN_MAX)])
            panel_node_x.append([x0+2])
            panel_node_y.append(input_y.tolist())
            panel_node_y.append(hidden_y.tolist())
            panel_node_y.append([out_y_pos])

            for k in sorted(connected_hidden):
                # ── input → hidden edges ────────────────────────────────
                for i in range(INPUT_DIM):
                    if not M_in_s[k, i]: continue
                    c      = float(C_in_po_s[j, k, i])
                    u_fp   = float(U_in_po_s[j, k, i])
                    a_hebb = float(A_in_po_s[j, k, i])
                    w      = float(W_in_s[k, i])
                    alpha  = float(alpha_in_s[k, i])
                    norm_u = min(max((c - vmin) / max(vmax - vmin, 1e-10), 0), 1)
                    color  = f'rgba({int(255*(1-norm_u))},{int(100*norm_u)},{int(255*norm_u)},0.7)'
                    width  = 1.5 + 3 * norm_u
                    is_pruned = (pruned_is_out is not None and not pruned_is_out
                                 and pruned_idx == k * INPUT_DIM + i)
                    if is_pruned: color, width = 'rgba(255,0,0,0.9)', 4
                    shapes.append(dict(type='line',
                                       x0=x0, y0=float(input_y[i]),
                                       x1=x0+1, y1=float(hidden_y[k]),
                                       line=dict(color=color, width=width)))
                    hover_x.append(x0+0.5)
                    hover_y.append(0.5*(float(input_y[i])+float(hidden_y[k])))
                    hover_text.append(
                        f'y{j}: in[{k},{i}] w={w:.4f} u_fp={u_fp:.4f} hebb={a_hebb:.4f} alpha={alpha:.4e}')

                # ── hidden → output edge ────────────────────────────────
                u_out_val  = float(U_out_s[j, k])
                a_hebb_out = float(A_out_s[j, k])
                c_out      = a_hebb_out if color_mode == 'hebbian' else u_out_val
                w      = float(W_out_s[j, k])
                alpha  = float(alpha_out_s[j, k])
                norm_u = min(max((c_out - vmin) / max(vmax - vmin, 1e-10), 0), 1)
                color  = f'rgba({int(255*(1-norm_u))},{int(100*norm_u)},{int(255*norm_u)},0.7)'
                width  = 1.5 + 3 * norm_u
                is_pruned = (pruned_is_out is not None and pruned_is_out
                             and pruned_idx == j * N_HIDDEN_MAX + k)
                if is_pruned: color, width = 'rgba(255,0,0,0.9)', 4
                shapes.append(dict(type='line',
                                   x0=x0+1, y0=float(hidden_y[k]),
                                   x1=x0+2, y1=out_y_pos,
                                   line=dict(color=color, width=width)))
                hover_x.append(x0+1.5)
                hover_y.append(0.5*(float(hidden_y[k])+out_y_pos))
                hover_text.append(
                    f'y{j}: out[{j},{k}] w={w:.4f} u={u_out_val:.4f} hebb={a_hebb_out:.4f} alpha={alpha:.4e}')

        loss_annots = []
        pol = snap['per_output_loss']
        for j in range(OUTPUT_DIM):
            x0   = j * PANEL_W
            task = j // N_OUTPUTS_PER_TASK
            c    = '#4a90d9' if task == 0 else '#d94a4a'
            loss_annots.append(dict(x=x0+2.4, y=out_y_pos, text=f'L={pol[j]:.3f}',
                                    showarrow=False, font=dict(size=9, color=c)))
        cx = (OUTPUT_DIM * PANEL_W - PANEL_W / 2)
        loss_annots.append(dict(x=cx, y=-0.15,
                                text=f'Total loss: {snap["total_loss"]:.4f}',
                                showarrow=False, font=dict(size=11, color='black')))
        if snap.get('event'):
            loss_annots.append(dict(x=cx, y=1.12, text=snap['event'],
                                    showarrow=False, font=dict(size=11, color='green')))

        label = f'P{snap["prune_step"]}'
        label += ' [PRUNE]' if snap['is_prune_frame'] else f'.{snap["sub_step"]+1}'
        frames.append((shapes, hover_x, hover_y, hover_text,
                       loss_annots, label, panel_node_x, panel_node_y))

    # ── Build figure ────────────────────────────────────────────────────────
    fig = go.Figure()
    colors_in = ['#4a90d9'] * N_INPUTS_PER_TASK + ['#d94a4a'] * N_INPUTS_PER_TASK

    init_pnx, init_pny = frames[0][6], frames[0][7]
    for j in range(OUTPUT_DIM):
        task      = j // N_OUTPUTS_PER_TASK
        out_color = '#4a90d9' if task == 0 else '#d94a4a'
        fig.add_trace(go.Scatter(
            x=init_pnx[j*3], y=init_pny[j*3], mode='markers+text',
            marker=dict(size=12, color=colors_in),
            text=[f'x{i}' for i in range(INPUT_DIM)],
            textposition='middle left', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=init_pnx[j*3+1], y=init_pny[j*3+1], mode='markers+text',
            marker=dict(size=14, color=['#666']*N_HIDDEN_MAX),
            text=[f'h{k}' for k in range(N_HIDDEN_MAX)],
            textposition='top center', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=init_pnx[j*3+2], y=init_pny[j*3+2], mode='markers+text',
            marker=dict(size=12, color=[out_color]),
            text=[f'y{j}'],
            textposition='middle right', showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter(x=frames[0][1], y=frames[0][2], mode='markers',
        marker=dict(size=15, opacity=0), text=frames[0][3],
        hoverinfo='text', showlegend=False))

    base_annots = []
    base_shapes = []
    for j in range(OUTPUT_DIM):
        x0   = j * PANEL_W
        task = j // N_OUTPUTS_PER_TASK
        c    = '#4a90d9' if task == 0 else '#d94a4a'
        base_annots.append(dict(x=x0+1, y=1.08, text=f'y{j} — Task {task+1}',
                                showarrow=False, font=dict(color=c, size=12)))
        if j > 0:
            base_shapes.append(dict(type='line',
                                    x0=x0-0.4, y0=-0.2, x1=x0-0.4, y1=1.15,
                                    line=dict(color='lightgray', width=1, dash='dot')))

    fig.update_layout(shapes=base_shapes + frames[0][0],
                      annotations=base_annots + frames[0][4])

    node_text_static = []
    for j in range(OUTPUT_DIM):
        node_text_static += [
            [f'x{i}' for i in range(INPUT_DIM)],
            [f'h{k}' for k in range(N_HIDDEN_MAX)],
            [f'y{j}'],
        ]

    steps = []
    for sh, hx, hy, ht, la, lbl, pnx, pny in frames:
        steps.append(dict(
            method='update',
            args=[{'x': pnx + [hx], 'y': pny + [hy],
                   'text': node_text_static + [ht]},
                  {'shapes': base_shapes + sh,
                   'annotations': base_annots + la}],
            label=lbl))

    color_label = 'Hebbian Utility' if color_mode == 'hebbian' else 'Fractional Propagated Utility'
    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix='Step: '),
                      steps=steps, pad=dict(t=60))],
        title=f'Pruning + Generation — Per-Output View  [{color_label}]',
        xaxis=dict(range=[-0.5, OUTPUT_DIM*PANEL_W], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.25, 1.2], showgrid=False,
                   zeroline=False, showticklabels=False),
        height=500, width=max(950, 300*OUTPUT_DIM))
    return fig

fig2_hebbian = plot_diagram_per_output(snapshots, color_mode='hebbian')
fig2_utility = plot_diagram_per_output(snapshots, color_mode='utility')
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path_per_output_hebbian = f"./plots/plot_per_output_hebbian_{ts}.html"
path_per_output_utility = f"./plots/plot_per_output_utility_{ts}.html"
fig2_hebbian.write_html(path_per_output_hebbian)
fig2_utility.write_html(path_per_output_utility)
# fig2_hebbian.show()

# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params({
        "seed": SEED,
        "n_inputs_per_task": N_INPUTS_PER_TASK,
        "n_outputs_per_task": N_OUTPUTS_PER_TASK,
        "n_hidden_init": N_HIDDEN_INIT,
        "n_hidden_max": N_HIDDEN_MAX,
        "n_tasks": N_TASKS,
        "optimizer": OPTIMIZER_NAME,
        "train_steps_per_prune": TRAIN_STEPS_PER_PRUNE,
        "n_snapshots_per_prune": N_SNAPSHOTS_PER_PRUNE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "total_prune_steps": TOTAL_PRUNE_STEPS,
        "connection_budget": CONNECTION_BUDGET,
        "n_prune_per_step": N_PRUNE_PER_STEP,
        "prunes_per_change": PRUNES_PER_CHANGE,
        "rank1_targets": RANK1_TARGETS,
        "step_utility_mode": STEP_UTILITY_MODE.name,
        "cull_utility_mode": CULL_UTILITY_MODE.name,
        "leaky_relu_negative_slope": LEAKY_RELU_NEGATIVE_SLOPE,
    })
    mlflow.log_artifact(path_all_utility, artifact_path="plots")
    mlflow.log_artifact(path_all_hebbian, artifact_path="plots")
    mlflow.log_artifact(path_per_output_utility, artifact_path="plots")
    mlflow.log_artifact(path_per_output_hebbian, artifact_path="plots")
    print(f"MLflow run logged: {mlflow.active_run().info.run_id}")


# %%
## Exact value tester

# 

import jax
import jax.numpy as jnp

SEED = 2133470219
jax_rng = jax.random.key(SEED)
jax_rng, k1, k2 = jax.random.split(jax_rng, 3)

# make_target_matrices uses k1
k1a, k1b = jax.random.split(k1)
A1 = jax.random.rademacher(k1a, (2, 2)).astype(jnp.float32)
A2 = jax.random.rademacher(k1b, (2, 2)).astype(jnp.float32)
print("A1 =", A1)
print("A2 =", A2)


# %%


# %%


# %% [markdown]
# A1 = [[ 1. -1.]
#  [ 1.  1.]]
# A2 = [[ 1. -1.]
#  [-1. -1.]]
# 
# exact values are:
# y0 = x0 - x1
# y1 = x0 + x1
# y2 = x2 - x3
# y3 = -x2 - x3
# 
# 


