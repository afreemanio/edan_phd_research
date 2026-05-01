"""Debug task separation at MNIST scale with batch pruning + EMA utility.

Extends the toy debug script to use real MNIST data, cross-entropy loss,
partial input connectivity, and the script's batch pruning + EMA mechanism.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from phd.jax_core.optimizers.idbd_jacobian import IDBDState, optax_idbd
from functools import partial
import time


@contextlib.contextmanager
def _null_context():
    yield


@dataclass
class PresetConfig:
    """Named structural preset for one comparative condition."""
    name: str
    run_kwargs: Dict[str, Any]
    prune_per_block: Optional[int] = 0    # None → use global PRUNE_PER_BLOCK
    generate_mode: Optional[str] = 'NO_GENERATE'  # None → use global GENERATE_MODE
    description: str = ''


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
N_TASKS = 2
# N_TASKS = 1
NUM_CLASSES = 10
INPUT_PER_TASK = 784
INPUT_DIM = INPUT_PER_TASK * N_TASKS   # 1568
OUTPUT_DIM = NUM_CLASSES * N_TASKS     # 20


# Per-layer unit caps; one entry per hidden layer, must match N_HIDDEN_LAYERS
N_HIDDEN_MAX_PER_LAYER = (100, 100)
LEAKY_RELU_SLOPE = 0.01
# LEAKY_RELU_SLOPE = 1.0
MAX_CONNS = 1569        # input connections generated per unit (input -> first hidden layer)
# MAX_CONNS = 128        # input connections per unit
# * DOUBLE MNIST HAS 1568 image inputs BY DEFAULT (784 per task × 2 tasks),
# * or 1569 with the bias.

# Block size is how much data is streamed before a cull happens, and also how long the utility estimate trace is
BLOCK_SIZE = 500      # = prune_frequency # ! default
# BLOCK_SIZE = 20_000      # = prune_frequency # ! default
# BLOCK_SIZE = 290_000      # = prune_frequency # ! default
# BLOCK_SIZE = 190_000      # = prune_frequency # ! default
# BLOCK_SIZE = 19_000      # = prune_frequency # ! default
# BLOCK_SIZE = 200      # = prune_frequency # ! default
# BLOCK_SIZE = 1000      # = prune_frequency # ! default
# BLOCK_SIZE = 500      # = prune_frequency
# BLOCK_SIZE = 1000      # = prune_frequency
# BLOCK_SIZE = 5000      # = prune_frequency
# BLOCK_SIZE = 10_000      # = prune_frequency
# BLOCK_SIZE = 50000      # = prune_frequency
TOTAL_STEPS = 500_000
# TOTAL_STEPS = 300_000
# TOTAL_STEPS = 100_000
# CONNECTION_BUDGET = 3000
# CONNECTION_BUDGET = 5000
# CONNECTION_BUDGET = 50000
# CONNECTION_BUDGET = 10000
# CONNECTION_BUDGET = 20000
# CONNECTION_BUDGET = 159000
# CONNECTION_BUDGET = 1590
# CONNECTION_BUDGET = 1440 # 1590 - 149 of connections that get spawned in each unit
# CONNECTION_BUDGET = 79350 # 79500 - 149 of connections that get spawned in each unit for 10 blocksparse
# CONNECTION_BUDGET = 6000 # 7950 - 149 of connections that get spawned in each unit for 10 blocksparse
# CONNECTION_BUDGET = 6000 # 7950 - 149 of connections that get spawned in each unit for 10 blocksparse
CONNECTION_BUDGET = 3182 # 7950 - 149 of connections that get spawned in each unit for 10 blocksparse
# Per-layer connection budget (None = use global CONNECTION_BUDGET only)
# Tuple length must equal N_HIDDEN_LAYERS + 1; index i = edge i budget
CONNECTION_BUDGET_PER_LAYER = None  # e.g. (2000, 1000, 182)
PRUNE_RATE = 0.01 # default
EMA_DECAY = 0.999
MAX_GEN_PER_STEP = 20000

# PRUNE_PER_BLOCK = 0 # Default  
PRUNE_PER_BLOCK = 129 # Default  
# PRUNE_PER_BLOCK = 149 # Default  
# PRUNE_PER_BLOCK = 1066 # Default  
# PRUNE_PER_BLOCK = 1266 # Default  
# PRUNE_PER_BLOCK = 130 # Default  
# PRUNE_PER_BLOCK = 200 # Default  
# PRUNE_PER_BLOCK = 784 # Default  
# PRUNE_PER_BLOCK = 1460 # Default  
# PRUNE_PER_BLOCK = 784 # Default  
# PRUNE_PER_BLOCK = 1466 # Default  
# PRUNE_PER_BLOCK = 106 # Default  
# PRUNE_PER_BLOCK = 300 
# PRUNE_PER_BLOCK = 15000
# PRUNE_PER_BLOCK = 500
# PRUNE_PER_BLOCK = 500
# PRUNE_PER_BLOCK = 20_000
# PRUNE_PER_BLOCK = 316 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 158 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 632 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 600 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 1200 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 2400 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 10_000 # full connected fractional utility reduction to 7k
# PRUNE_PER_BLOCK = 300 


# SGD_LR = 1e-3
SGD_LR = 1e-3
ADAM_LR = 1e-4
# ADAMW_LR = 1e-4 # best on mnist
ADAMW_LR = 1e-3 # best on mnist
# ADAMW_LR = 1e-4



IDBD_INIT_LR = 1e-6
IDBD_META_LR = 1e-3
WEIGHT_DECAY = 0.0
# WEIGHT_DECAY = 0.0001
# WEIGHT_DECAY = 0.001


### ! Below settings for I guess a default culling
GENERATE_MODE = "FULL"
# GENERATE_MODE = "NO_GENERATE"
# NO_GENERATE
# GENERATE_MODE = "RANDOM_SPARSE"
FC_ISOLATED_DEFAULT = False
# FC_ISOLATED_DEFAULT = True # for fully connected, sets whether cross-task connections are prevented

# SKIP_LAST_CULL = False # If true, doesn't do the last prune/generate for the last block before validation
SKIP_LAST_CULL = True # If true, doesn't do the last prune/generate for the last block before validation

AUX_HEBBIAN_MODE_DEFAULT = 'none'
# AUX_HEBBIAN_MODE_DEFAULT = 'post_activation'
CULL_UTILITY_MODE_DEFAULT = 'original'
# CULL_UTILITY_MODE_DEFAULT = 'hebbian'
# CULL_UTILITY_MODE_DEFAULT = 'product'
# options: 'original', 'hebbian', 'product', 'sum'
UTILITY_MODE_DEFAULT = 'contribution'
OPTIMIZER_DEFAULT = 'sgd'


# # ! EDAN_DEFAULT_SETTINGS:
# N_TASKS = 2
# NUM_CLASSES = 10
# INPUT_PER_TASK = 784
# INPUT_DIM = INPUT_PER_TASK * N_TASKS   # 1568
# OUTPUT_DIM = NUM_CLASSES * N_TASKS     # 20

# N_HIDDEN_MAX = 100
# MAX_CONNS = 64        # input connections per unit (half of script's 128)

# BLOCK_SIZE = 100      # = prune_frequency
# TOTAL_STEPS = 100_000
# CONNECTION_BUDGET = 3000
# PRUNE_RATE = 0.0001
# EMA_DECAY = 0.999
# MAX_GEN_PER_STEP = 20

# SGD_LR = 1e-3
# ADAM_LR = 1e-3
# SEED = 42

# SEED = 43
# SEED = 421
SEED = 6969

# USE_BIAS_FEATURE = False  # when True, appends one always-1 bias input
USE_BIAS_FEATURE = True  # when True, appends one always-1 bias input


# PER_PIXEL_NORMALIZE = False
PER_PIXEL_NORMALIZE = True


# ══════════════════════════════════════════════════════════════════════
# MLFLOW CONFIG
# ══════════════════════════════════════════════════════════════════════

# PER_UNIT_LOG_MAX_UNITS = 50  # log per-unit metrics only when active units <= this
PER_UNIT_LOG_MAX_UNITS = 0  # log per-unit metrics only when active units <= this

MLFLOW_TRACKING_URI = "http://localhost:5000"
# MLFLOW_EXPERIMENT_NAME = "double_mnist_prune_and_gen"
# MLFLOW_EXPERIMENT_NAME = "double_mnist_single_unit_bias_comparison_weight_test"
# MLFLOW_EXPERIMENT_NAME = "double_mnist_single_unit_different_time_magnitude_check"
# MLFLOW_EXPERIMENT_NAME = "benchmarking_current_advances"
MLFLOW_EXPERIMENT_NAME = "new_multilayer_test_01"
# MLFLOW_EXPERIMENT_NAME = "double_mnist_single_unit_prune_784_test"
# GLOBAL_RUN_NAME = "run_group_1"  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "SEED43_NOCROSSTASK_double_mnist_fully_connected_600n_NO_BIAS"  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "gentest_SEED43_20k_connections_15000_to_prune_every_20k_128_default_inputs_weightdecay0.0_"  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "prune_200_once"  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "bias:"  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "NORMALIZE 200k bias cull 130 every16k"  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "cull adam baseline 200k no bias 130 every16k"  # appended with experiment name for each run
GLOBAL_RUN_NAME = ""  # appended with experiment name for each run
# GLOBAL_RUN_NAME = "200k bias cull"  # appended with experiment name for each run

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FREQ = 10_000   # log every this many steps (used when LOG_MODE='interval')
LOG_MODE = 'interval'  # 'interval': every LOG_FREQ steps; 'block': every block boundary


# Derived
BIAS_DIM = 1 if USE_BIAS_FEATURE else 0  # single global bias appended after all task inputs
INPUT_DIM = INPUT_PER_TASK * N_TASKS + BIAS_DIM
OUTPUT_DIM = NUM_CLASSES * N_TASKS


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════


def load_mnist():
    from data import load_dataset
    images, labels, _, _, _, _ = load_dataset('mnist', split='train', per_pixel_normalize=PER_PIXEL_NORMALIZE)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

class ParallelMNIST:
    """Parallel MNIST stream with optional label permutation."""
    def __init__(self, images, labels, seed=0, permute_period=0):
        self.images = images
        self.labels = labels
        self.n = len(images)
        self.rng = np.random.default_rng(seed)
        self.permute_period = permute_period
        self.perms = [np.arange(NUM_CLASSES) for _ in range(N_TASKS)]
        self.step = 0

    def sample_block(self, n_steps):
        xs = np.zeros((n_steps, INPUT_DIM), dtype=np.float32)
        ys = np.zeros((n_steps, N_TASKS), dtype=np.int32)
        for s in range(n_steps):
            self.step += 1
            if self.permute_period > 0 and self.step % self.permute_period == 0:
                k = self.rng.integers(0, N_TASKS)
                self.perms[k] = self.rng.permutation(NUM_CLASSES)
            for t in range(N_TASKS):
                idx = self.rng.integers(0, self.n)
                xs[s, t * INPUT_PER_TASK:(t + 1) * INPUT_PER_TASK] = self.images[idx]
                ys[s, t] = self.perms[t][self.labels[idx]]
            if USE_BIAS_FEATURE:
                xs[s, -1] = 1.0
        return xs, ys

# ══════════════════════════════════════════════════════════════════════
# NETWORK / MULTILAYER PARAMETER HELPERS
# ══════════════════════════════════════════════════════════════════════

# Set this to 1 to recover the old architecture, or 2 for the first
# multilayer experiments. The code below is written for any positive value,
# but the generation policy is intentionally conservative and path-based.
N_HIDDEN_LAYERS = 2
assert len(N_HIDDEN_MAX_PER_LAYER) == N_HIDDEN_LAYERS, (
    f"N_HIDDEN_MAX_PER_LAYER length {len(N_HIDDEN_MAX_PER_LAYER)} != N_HIDDEN_LAYERS {N_HIDDEN_LAYERS}")
HIDDEN_SIZES = N_HIDDEN_MAX_PER_LAYER
N_HIDDEN_MAX = max(N_HIDDEN_MAX_PER_LAYER)  # backward-compat alias

# ── Per-layer initialization ────────────────────────────────────────────────
# Starting units at each hidden layer (random-sparse path and FC non-isolated).
# FC isolated always fills to N_HIDDEN_MAX_PER_LAYER (original behaviour).
N_HIDDEN_INIT_PER_LAYER = (2, 2)
# Init-time connection counts per edge (None = fully connected for that edge).
# Index 0 = input→h0, index l = h_{l-1}→h_l.
INIT_CONNS_PER_LAYER = (MAX_CONNS, None)

# ── Per-layer generation ────────────────────────────────────────────────────
# Maximum units to add per hidden layer per generate_fill call.
# Since each path adds 1 unit per layer, the effective path cap is
# min(GEN_UNITS_PER_LAYER); set to N_HIDDEN_MAX_PER_LAYER for no added cap.
GEN_UNITS_PER_LAYER = N_HIDDEN_MAX_PER_LAYER
# Connection counts for spawned units per edge (None = fully connected).
# Index 0 = input→h0 connections, index l = h_{l-1}→h_l connections.
GEN_CONNS_PER_LAYER = (MAX_CONNS, 1)


def layer_dims():
    return (INPUT_DIM,) + HIDDEN_SIZES + (OUTPUT_DIM,)


def edge_names():
    return tuple(f"W{i}" for i in range(N_HIDDEN_LAYERS + 1))


def mask_names():
    return tuple(f"M{i}" for i in range(N_HIDDEN_LAYERS + 1))


def empty_params_and_masks():
    dims = layer_dims()
    params = {}
    masks = {}
    utilities = {}
    hebbian = {}
    for i, name in enumerate(edge_names()):
        shape = (dims[i + 1], dims[i])
        params[name] = jnp.zeros(shape)
        masks[f"M{i}"] = jnp.zeros(shape, dtype=jnp.int32)
        utilities[name] = jnp.zeros(shape)
        hebbian[name] = jnp.zeros(shape)
    return params, masks, utilities, hebbian


def apply_masks(params, masks):
    return {k: params[k] * masks[k.replace('W', 'M')] for k in params}


def forward(params, masks, x, return_activations=False):
    """Forward pass for input -> hidden layers -> output.

    params/masks have one matrix per edge:
      W0/M0: hidden0 x input
      W1/M1: hidden1 x hidden0
      ...
      WN/MN: output x hidden{N-1}
    """
    a = x
    activations = [x]
    for i in range(N_HIDDEN_LAYERS):
        W = params[f"W{i}"] * masks[f"M{i}"]
        a = jax.nn.leaky_relu(W @ a, negative_slope=LEAKY_RELU_SLOPE)
        activations.append(a)
    W_out = params[f"W{N_HIDDEN_LAYERS}"] * masks[f"M{N_HIDDEN_LAYERS}"]
    raw = W_out @ activations[-1]
    if return_activations:
        return raw, activations
    return raw, activations[-1]


def forward_from_hidden(params, masks, activations, start_hidden_layer):
    """Recompute logits after replacing activations[start_hidden_layer + 1]."""
    a = activations[start_hidden_layer + 1]
    for edge in range(start_hidden_layer + 1, N_HIDDEN_LAYERS):
        W = params[f"W{edge}"] * masks[f"M{edge}"]
        a = jax.nn.leaky_relu(W @ a, negative_slope=LEAKY_RELU_SLOPE)
    W_out = params[f"W{N_HIDDEN_LAYERS}"] * masks[f"M{N_HIDDEN_LAYERS}"]
    return W_out @ a


def loss_ce(params, masks, x, label):
    raw, _ = forward(params, masks, x)
    logits = raw.reshape(N_TASKS, NUM_CLASSES)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(label, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

# ══════════════════════════════════════════════════════════════════════
# UTILITY / OPTIMIZER STATE RESET
# ══════════════════════════════════════════════════════════════════════

def reset_adam_family_state(opt_state, grown_masks):
    if hasattr(opt_state, "mu") and hasattr(opt_state, "nu"):
        return opt_state._replace(
            mu=jax.tree.map(lambda m, s: jnp.where(m, 0, s), grown_masks, opt_state.mu),
            nu=jax.tree.map(lambda m, s: jnp.where(m, 0, s), grown_masks, opt_state.nu),
        )
    return opt_state


def reset_idbd_state(opt_state: IDBDState, grown_masks):
    return IDBDState(
        init_beta=opt_state.init_beta,
        beta=jax.tree.map(lambda m, b: jnp.where(m, jnp.full_like(b, opt_state.init_beta), b), grown_masks, opt_state.beta),
        h=jax.tree.map(lambda m, h: jnp.where(m, 0, h), grown_masks, opt_state.h),
        v=None if opt_state.v is None else jax.tree.map(lambda m, v: jnp.where(m, 0, v), grown_masks, opt_state.v),
        beta_sq_avg=None if opt_state.beta_sq_avg is None else jax.tree.map(lambda m, v: jnp.where(m, 0, v), grown_masks, opt_state.beta_sq_avg),
        step=opt_state.step,
        exp_avg=None if opt_state.exp_avg is None else jax.tree.map(lambda m, v: jnp.where(m, 0, v), grown_masks, opt_state.exp_avg),
        exp_avg_sq=None if opt_state.exp_avg_sq is None else jax.tree.map(lambda m, v: jnp.where(m, 0, v), grown_masks, opt_state.exp_avg_sq),
    )


def maybe_reset_opt_state(opt_state, grown_masks):
    if isinstance(opt_state, IDBDState):
        return reset_idbd_state(opt_state, grown_masks)
    return reset_adam_family_state(opt_state, grown_masks)


@jax.jit
def _reset_grown_connections(opt_state, masks, old_masks):
    grown = {w: (masks[w.replace('W', 'M')] == 1) & (old_masks[w.replace('W', 'M')] == 0)
             for w in edge_names()}
    return maybe_reset_opt_state(opt_state, grown)


def compute_utility_contribution(params, masks, x):
    _, acts = forward(params, masks, x, return_activations=True)
    utils = {}
    for i, wname in enumerate(edge_names()):
        mname = f"M{i}"
        utils[wname] = jnp.abs(acts[i])[None, :] * jnp.abs(params[wname]) * masks[mname]
    return utils


def compute_utility_fractional_propagated(params, masks, x):
    raw, acts = forward(params, masks, x, return_activations=True)
    node_u = jnp.ones((OUTPUT_DIM,))
    utils_rev = {}
    target_signal = raw
    for i in reversed(range(N_HIDDEN_LAYERS + 1)):
        wname, mname = f"W{i}", f"M{i}"
        contrib = jnp.abs(acts[i])[None, :] * jnp.abs(params[wname]) * masks[mname]
        total_into_target = jnp.sum(contrib, axis=1, keepdims=True)
        frac = jnp.where(total_into_target > 1e-10, contrib / total_into_target, 0.0)
        edge_u = frac * node_u[:, None] * masks[mname]
        utils_rev[wname] = edge_u
        node_u = jnp.sum(edge_u, axis=0)
    return {w: utils_rev[w] for w in edge_names()}


def compute_utility_propagated(params, masks, x):
    # For multilayer networks, fractional backward propagation is the most stable
    # extension of the original one-hidden-layer propagated utility.
    return compute_utility_fractional_propagated(params, masks, x)


def _ce_from_logits(logits, y):
    logits = logits.reshape(N_TASKS, NUM_CLASSES)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def compute_utility_edan_cross_entropy(params, masks, x, y):
    raw, acts = forward(params, masks, x, return_activations=True)
    base_ce = _ce_from_logits(raw, y)
    utils = {w: jnp.zeros_like(params[w]) for w in edge_names()}

    for hidden_layer in range(N_HIDDEN_LAYERS):
        h = acts[hidden_layer + 1]
        n_units = h.shape[0]

        def loo_unit(u):
            h_wo = h.at[u].set(0.0)
            new_acts = list(acts)
            new_acts[hidden_layer + 1] = h_wo
            logits_wo = forward_from_hidden(params, masks, new_acts, hidden_layer)
            return _ce_from_logits(logits_wo, y) - base_ce

        u_hidden = jax.vmap(loo_unit)(jnp.arange(n_units))

        in_edge = f"W{hidden_layer}"
        in_mask = f"M{hidden_layer}"
        utils[in_edge] = utils[in_edge] + u_hidden[:, None] * jnp.abs(params[in_edge]) * masks[in_mask]

        out_edge_idx = hidden_layer + 1
        out_edge = f"W{out_edge_idx}"
        out_mask = f"M{out_edge_idx}"
        utils[out_edge] = utils[out_edge] + u_hidden[None, :] * jnp.abs(params[out_edge]) * masks[out_mask]

    return utils


def compute_auxiliary_hebbian_utility(params, masks, x, base_utils, aux_hebbian_mode='post_activation'):
    if aux_hebbian_mode in (None, 'none'):
        return {w: jnp.zeros_like(base_utils[w]) for w in edge_names()}
    if aux_hebbian_mode != 'post_activation':
        raise ValueError(f"Unknown aux_hebbian_mode: {aux_hebbian_mode}")

    raw, acts = forward(params, masks, x, return_activations=True)
    target_acts = acts[1:] + [raw]
    return {
        w: jnp.abs(target_acts[i])[:, None] * base_utils[w] * masks[f"M{i}"]
        for i, w in enumerate(edge_names())
    }


def select_cull_utility(U, H, cull_utility_mode='original'):
    if cull_utility_mode == 'original':
        return U
    if cull_utility_mode == 'hebbian':
        return H
    if cull_utility_mode == 'product':
        return {w: U[w] * H[w] for w in edge_names()}
    if cull_utility_mode == 'sum':
        return {w: U[w] + H[w] for w in edge_names()}
    raise ValueError(f"Unknown cull_utility_mode: {cull_utility_mode}")


def compute_utility(params, masks, x, utility_mode, y=None):
    if utility_mode == 'contribution':
        return compute_utility_contribution(params, masks, x)
    if utility_mode == 'propagated':
        return compute_utility_propagated(params, masks, x)
    if utility_mode == 'fractional_propagated':
        return compute_utility_fractional_propagated(params, masks, x)
    if utility_mode == 'edan_cross_entropy':
        return compute_utility_edan_cross_entropy(params, masks, x, y)
    raise ValueError(f"Unknown utility_mode: {utility_mode}")


def _predict_only(params, masks, x):
    raw, _ = forward(params, masks, x)
    return raw


def _train_step_optax(params, masks, x, y, optimizer, opt_state):
    def loss_from_params(p):
        return loss_ce(p, masks, x, y)

    def pred_from_params(p):
        return _predict_only(p, masks, x)

    loss_grads = jax.grad(loss_from_params)(params)
    if hasattr(opt_state, "beta"):
        prediction_grads = jax.jacrev(pred_from_params)(params)
        optimizer_payload = (loss_grads, prediction_grads)
    else:
        optimizer_payload = loss_grads

    updates, opt_state = optimizer.update(optimizer_payload, opt_state, params)
    updates = {w: updates[w] * masks[w.replace('W', 'M')] for w in edge_names()}
    params = optax.apply_updates(params, updates)
    params = apply_masks(params, masks)
    return params, opt_state

# ══════════════════════════════════════════════════════════════════════
# BATCH PRUNING / CLEANUP
# ══════════════════════════════════════════════════════════════════════

@jax.jit
def batch_prune(U, masks, n_to_prune, rng):
    flat_us = []
    flat_active = []
    for i, w in enumerate(edge_names()):
        m = masks[f"M{i}"]
        u = U[w]
        flat_us.append(jnp.where(m.reshape(-1) == 1, u.reshape(-1), jnp.inf))
        flat_active.append(m.reshape(-1) == 1)
    all_u = jnp.concatenate(flat_us)
    active = jnp.concatenate(flat_active)
    noise = jax.random.uniform(rng, all_u.shape, minval=-1e-10, maxval=1e-10)
    perturbed = jnp.where(jnp.isinf(all_u), all_u, all_u + noise)
    safe_n = jnp.minimum(n_to_prune, jnp.maximum(active.sum() - 1, 0))
    thresh = jnp.sort(perturbed)[safe_n]
    thresh = jnp.minimum(thresh, 1e30)
    prune = (perturbed < thresh) & active

    new_masks = {}
    offset = 0
    for i in range(N_HIDDEN_LAYERS + 1):
        mname = f"M{i}"
        size = masks[mname].size
        p = prune[offset:offset + size].reshape(masks[mname].shape)
        new_masks[mname] = masks[mname] * (1 - p.astype(jnp.int32))
        offset += size
    return new_masks


def cleanup_dead(masks):
    masks = dict(masks)
    # A hidden unit is useful only if it has at least one incoming and one outgoing edge.
    # Iterate a few times so removing one layer's units can cascade to neighbours.
    for _ in range(N_HIDDEN_LAYERS + 1):
        changed = False
        for h_layer in range(N_HIDDEN_LAYERS):
            in_m = masks[f"M{h_layer}"]
            out_m = masks[f"M{h_layer + 1}"]
            alive = (in_m.sum(axis=1) > 0) & (out_m.sum(axis=0) > 0)
            masks[f"M{h_layer}"] = in_m * alive[:, None]
            masks[f"M{h_layer + 1}"] = out_m * alive[None, :]
    return masks

# ══════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=('all_outputs',))
def generate_unit_path(params, masks, key, all_outputs=True):
    """Generate one path: random inputs -> one new unit per hidden layer -> outputs.

    Slot search uses per-layer caps from N_HIDDEN_MAX_PER_LAYER.
    Connection counts per edge are taken from GEN_CONNS_PER_LAYER:
      index 0 = input->h0 connections, index l = h_{l-1}->h_l connections.
    None means fully connected for that edge.
    Middle-edge weights start at 0 (existing convention).
    """
    keys = jax.random.split(key, 2 * N_HIDDEN_LAYERS + 4)
    new_params = dict(params)
    new_masks = dict(masks)

    slots = []
    has_all = jnp.array(True)
    for h_layer in range(N_HIDDEN_LAYERS):
        cap = N_HIDDEN_MAX_PER_LAYER[h_layer]
        in_m = masks[f"M{h_layer}"]
        out_m = masks[f"M{h_layer + 1}"]
        active = (in_m.sum(axis=1) + out_m.sum(axis=0)) > 0
        slot = jnp.argmin(jnp.where(~active, jnp.arange(cap), cap))
        has = jnp.any(~active)
        slots.append(slot)
        has_all = has_all & has

    # Edge 0: input -> h0
    _gc0 = GEN_CONNS_PER_LAYER[0]
    n_conns_0 = INPUT_DIM if _gc0 is None else min(_gc0, INPUT_DIM)
    noise = jax.random.uniform(keys[0], (INPUT_DIM,))
    perm = jnp.argsort(noise)
    im = jnp.zeros(INPUT_DIM, dtype=jnp.int32).at[perm[:n_conns_0]].set(1)
    bound = jnp.sqrt(3.0 / n_conns_0)
    w0 = jax.random.uniform(keys[1], (INPUT_DIM,), minval=-bound, maxval=bound) * im
    h0 = slots[0]
    new_params["W0"] = jnp.where(has_all, new_params["W0"].at[h0].set(w0), new_params["W0"])
    new_masks["M0"] = jnp.where(has_all, new_masks["M0"].at[h0].set(im), new_masks["M0"])

    # Middle edges: h_{l-1} -> h_l
    mid_conn_costs = []
    for h_layer in range(1, N_HIDDEN_LAYERS):
        h = slots[h_layer]
        wname, mname = f"W{h_layer}", f"M{h_layer}"
        prev_cap = N_HIDDEN_MAX_PER_LAYER[h_layer - 1]
        _gcl = GEN_CONNS_PER_LAYER[h_layer]
        n_conns_mid = prev_cap if _gcl is None else min(_gcl, prev_cap)
        mid_conn_costs.append(n_conns_mid)
        if _gcl is None:
            im_mid = jnp.ones(prev_cap, dtype=jnp.int32)
        else:
            noise_mid = jax.random.uniform(keys[2 + h_layer], (prev_cap,))
            perm_mid = jnp.argsort(noise_mid)
            im_mid = jnp.zeros(prev_cap, dtype=jnp.int32).at[perm_mid[:n_conns_mid]].set(1)
        # Weights start at 0 (existing convention for generated middle edges)
        w_mid = jnp.zeros(prev_cap, dtype=jnp.float32)
        new_params[wname] = jnp.where(has_all, new_params[wname].at[h, :].set(w_mid), new_params[wname])
        new_masks[mname] = jnp.where(has_all, new_masks[mname].at[h, :].set(im_mid), new_masks[mname])

    # Output edge
    out_edge = N_HIDDEN_LAYERS
    last_h = slots[-1]
    om_all = jnp.ones(OUTPUT_DIM, dtype=jnp.int32)
    oi = jax.random.randint(keys[-1], (), 0, OUTPUT_DIM)
    om_single = jnp.zeros(OUTPUT_DIM, dtype=jnp.int32).at[oi].set(1)
    om = jnp.where(all_outputs, om_all, om_single)
    new_params[f"W{out_edge}"] = jnp.where(has_all, new_params[f"W{out_edge}"].at[:, last_h].set(0.0), new_params[f"W{out_edge}"])
    new_masks[f"M{out_edge}"] = jnp.where(has_all, new_masks[f"M{out_edge}"].at[:, last_h].set(om), new_masks[f"M{out_edge}"])

    mid_total = sum(mid_conn_costs)
    cost = jnp.where(all_outputs,
                     n_conns_0 + mid_total + OUTPUT_DIM,
                     n_conns_0 + mid_total + 1)
    return new_params, new_masks, has_all, cost


def generate_fill(params, masks, budget, key, all_outputs=True):
    budget_f = float(budget)
    # GEN_UNITS_PER_LAYER caps how many paths (= units per layer) per call.
    # Each path adds 1 unit to every hidden layer, so the effective cap is
    # min(GEN_UNITS_PER_LAYER). Defaults to N_HIDDEN_MAX_PER_LAYER so the
    # connection budget remains the binding constraint (original behaviour).
    max_paths = min(GEN_UNITS_PER_LAYER)
    paths_added = 0
    for _ in range(MAX_GEN_PER_STEP):
        if budget_f <= 0 or paths_added >= max_paths:
            break
        key, gk = jax.random.split(key)
        params, masks, gen, cost = generate_unit_path(params, masks, gk, all_outputs)
        if not bool(gen):
            break
        budget_f -= int(cost)
        paths_added += 1
    return params, masks


@jax.jit
def grow_one_connection(params, masks, key):
    keys = jax.random.split(key, 5)
    edge = jax.random.randint(keys[0], (), 0, N_HIDDEN_LAYERS + 1)

    def grow_edge(i, state):
        params, masks = state
        wname, mname = f"W{i}", f"M{i}"
        missing = masks[mname] == 0
        n_missing = missing.sum()

        def add_it(_):
            flat_candidates = jnp.where(missing.reshape(-1), size=masks[mname].size, fill_value=masks[mname].size)[0]
            pos = jax.random.randint(keys[1], (), 0, n_missing)
            flat = flat_candidates[pos]
            row = flat // masks[mname].shape[1]
            col = flat % masks[mname].shape[1]
            fan_in = jnp.maximum(masks[mname][row].sum(), 1)
            bound = jnp.sqrt(3.0 / fan_in)
            w = jax.random.uniform(keys[2], (), minval=-bound, maxval=bound)
            params2 = dict(params)
            masks2 = dict(masks)
            params2[wname] = params2[wname].at[row, col].set(w)
            masks2[mname] = masks2[mname].at[row, col].set(1)
            return params2, masks2, jnp.int32(1)

        return jax.lax.cond(n_missing > 0, add_it,
                            lambda _: (params, masks, jnp.int32(0)), operand=None)

    used = jnp.int32(0)
    for i in range(N_HIDDEN_LAYERS + 1):
        params_i, masks_i, used_i = grow_edge(i, (params, masks))
        take = edge == i
        params = jax.tree.map(lambda a, b: jnp.where(take, a, b), params_i, params)
        masks = jax.tree.map(lambda a, b: jnp.where(take, a, b), masks_i, masks)
        used = jnp.where(take, used_i, used)
    return params, masks, used


@jax.jit
def generate_fill_sparse(params, masks, budget, key):
    budget = jnp.asarray(budget, dtype=jnp.int32)
    last_used = jnp.int32(1)

    def cond(state):
        _, _, _, budget, last_used = state
        return (budget > 0) & (last_used > 0)

    def body(state):
        params, masks, key, budget, _ = state
        key, gk = jax.random.split(key)
        params, masks, used = grow_one_connection(params, masks, gk)
        return params, masks, key, budget - used, used

    params, masks, _, _, _ = jax.lax.while_loop(cond, body, (params, masks, key, budget, last_used))
    return params, masks

# ══════════════════════════════════════════════════════════════════════
# JIT TRAINING BLOCKS
# ══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=('optimizer', 'utility_mode', 'aux_hebbian_mode', 'n_steps'))
def train_block_optax(params, masks, images, labels, U, H, acc,
                      opt_state, optimizer, utility_mode, aux_hebbian_mode, n_steps):
    n_active = sum(m.sum() for m in masks.values()).astype(jnp.float32)
    acc_inc = PRUNE_RATE * n_active

    def body(carry, data):
        params, U, H, acc, opt_state = carry
        x, y = data
        params, opt_state = _train_step_optax(params, masks, x, y, optimizer, opt_state)
        u = compute_utility(params, masks, x, utility_mode, y)
        h = compute_auxiliary_hebbian_utility(params, masks, x, u, aux_hebbian_mode)
        U = {w: EMA_DECAY * U[w] + (1 - EMA_DECAY) * u[w] for w in edge_names()}
        H = {w: EMA_DECAY * H[w] + (1 - EMA_DECAY) * h[w] for w in edge_names()}
        return (params, U, H, acc + acc_inc, opt_state), None

    (params, U, H, acc, opt_state), _ = jax.lax.scan(
        body, (params, U, H, acc, opt_state), (images, labels)
    )
    return params, U, H, acc, opt_state

# ══════════════════════════════════════════════════════════════════════
# METRIC
# ══════════════════════════════════════════════════════════════════════

def _input_task_reach():
    reach = np.zeros((N_TASKS, INPUT_DIM), dtype=bool)
    for t in range(N_TASKS):
        reach[t, t * INPUT_PER_TASK:(t + 1) * INPUT_PER_TASK] = True
        if USE_BIAS_FEATURE:
            reach[t, -1] = True
    return reach


def _output_task_reach():
    reach = np.zeros((N_TASKS, OUTPUT_DIM), dtype=bool)
    for t in range(N_TASKS):
        reach[t, t * NUM_CLASSES:(t + 1) * NUM_CLASSES] = True
    return reach


def measure_purity(masks):
    masks_np = {k: np.array(v) for k, v in masks.items()}

    # Forward task reachability: which task inputs can reach each hidden unit?
    fwd_reach = [_input_task_reach()]
    fwd_counts = []
    for i in range(N_HIDDEN_LAYERS):
        m = masks_np[f"M{i}"].astype(bool)
        counts = np.stack([(m & fwd_reach[-1][t][None, :]).sum(axis=1) for t in range(N_TASKS)])
        reach = counts > 0
        fwd_counts.append(counts)
        fwd_reach.append(reach)

    # Backward task reachability: which task outputs can be reached from each hidden unit?
    bwd_reach = [None] * (N_HIDDEN_LAYERS + 1)
    bwd_reach[N_HIDDEN_LAYERS] = _output_task_reach()
    bwd_counts = [None] * N_HIDDEN_LAYERS
    for i in reversed(range(N_HIDDEN_LAYERS)):
        m = masks_np[f"M{i + 1}"].astype(bool)  # target x source
        counts = np.stack([(m & bwd_reach[i + 1][t][:, None]).sum(axis=0) for t in range(N_TASKS)])
        reach = counts > 0
        bwd_counts[i] = counts
        bwd_reach[i] = reach

    all_rows = []
    layer_metrics = []
    for layer in range(N_HIDDEN_LAYERS):
        rows = []
        in_counts = fwd_counts[layer]
        out_counts = bwd_counts[layer]
        for h in range(N_HIDDEN_MAX_PER_LAYER[layer]):
            ti = int(in_counts[:, h].sum())
            to = int(out_counts[:, h].sum())
            if ti == 0 and to == 0:
                continue
            ic = [int(in_counts[t, h]) for t in range(N_TASKS)]
            oc = [int(out_counts[t, h]) for t in range(N_TASKS)]
            ip = max(ic) / max(ti, 1)
            op = max(oc) / max(to, 1)
            ie = sum(-c / ti * np.log2(c / ti + 1e-15) for c in ic if c > 0) if ti > 0 else 0.0
            rows.append(dict(layer=layer, u=h, ic=ic, oc=oc, ip=ip, op=op, ie=ie, ti=ti, to=to))
        all_rows.extend(rows)
        if rows:
            layer_metrics.append((np.mean([r['ip'] for r in rows]), np.mean([r['op'] for r in rows]), np.mean([r['ie'] for r in rows])))
        else:
            layer_metrics.append((0.0, 0.0, 1.0))

    if not all_rows:
        print("  No active units!")
        return 0.0, 0.0, 1.0

    mip = float(np.mean([r['ip'] for r in all_rows]))
    mop = float(np.mean([r['op'] for r in all_rows]))
    mie = float(np.mean([r['ie'] for r in all_rows]))

    for layer in range(N_HIDDEN_LAYERS):
        rows = [r for r in all_rows if r['layer'] == layer]
        print(f"  --- Hidden layer {layer} purity ---")
        rows.sort(key=lambda r: r['ie'], reverse=True)
        n_show = min(5, len(rows))
        shown = rows[:n_show]
        if len(rows) > 2 * n_show:
            shown.append(None)
        shown += rows[-n_show:]
        for r in shown:
            if r is None:
                print(f'  ... ({len(rows) - 2 * n_show} more units) ...')
                continue
            ic_s = '/'.join(str(c) for c in r['ic'])
            oc_s = '/'.join(str(c) for c in r['oc'])
            print(f"  L{r['layer']} h{r['u']:3d}: in=[{ic_s}]({r['ti']:3d}) pur={r['ip']:.2f} ent={r['ie']:.3f}"
                  f" | out=[{oc_s}]({r['to']:2d}) pur={r['op']:.2f}")
        lip, lop, lie = layer_metrics[layer]
        print(f"  Layer {layer}: purity in={lip:.3f} out={lop:.3f} | entropy in={lie:.3f}")

    nc = int(sum(m.sum() for m in masks_np.values()))
    print(f'  Active hidden units: {len(all_rows)} | Connections: {nc}')
    print(f'  >>> Purity: in={mip:.3f} out={mop:.3f} | Entropy: in={mie:.3f}')
    return mip, mop, mie


@jax.jit
def _eval_per_task_loss_full(params, masks, imgs, lbls):
    def one(x, y):
        raw, _ = forward(params, masks, x)
        logits = raw.reshape(N_TASKS, NUM_CLASSES)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.array([
            -jnp.sum(jax.nn.one_hot(y[t], NUM_CLASSES) * log_probs[t])
            for t in range(N_TASKS)
        ])
    return jax.vmap(one)(imgs, lbls).mean(axis=0)


@partial(jax.jit, static_argnames=('active_task',))
def _eval_task_loss_inputs_only(params, masks, imgs, lbls, active_task):
    task_mask = jnp.zeros(INPUT_DIM).at[
        active_task * INPUT_PER_TASK:(active_task + 1) * INPUT_PER_TASK
    ].set(1.0)
    if USE_BIAS_FEATURE:
        task_mask = task_mask.at[-1].set(1.0)

    def one(x, y):
        raw, _ = forward(params, masks, x * task_mask)
        logits = raw.reshape(N_TASKS, NUM_CLASSES)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(jax.nn.one_hot(y[active_task], NUM_CLASSES) * log_probs[active_task])

    return jax.vmap(one)(imgs, lbls).mean()


def run_single_task_validation(params, masks, test_imgs, test_lbls, per_task_full_losses):
    print('\n  --- Single-Task Input Validation ---')
    imgs_j = jnp.array(test_imgs)
    lbls_j = jnp.array(test_lbls)

    task_results = []
    for t in range(N_TASKS):
        loss_isolated = float(_eval_task_loss_inputs_only(params, masks, imgs_j, lbls_j, t))
        loss_full = float(per_task_full_losses[t])
        delta = loss_isolated - loss_full
        task_results.append((loss_full, loss_isolated, delta))
        print(f'  Task {t}:  full_inputs={loss_full:.4f}  own_inputs_only={loss_isolated:.4f}  '
              f'Δ={delta:+.4f}{"  (relies on other inputs)" if delta > 0.01 else ""}')
    print()
    return task_results

# ══════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

def unit_specs(n_units=1, *, n_in_t0=0, n_out_t0=0, n_in_t1=0, n_out_t1=0):
    spec = {
        'inputs':  {t: n for t, n in {0: n_in_t0, 1: n_in_t1}.items() if n > 0},
        'outputs': {t: n for t, n in {0: n_out_t0, 1: n_out_t1}.items() if n > 0},
    }
    return [spec] * n_units


def init_custom_units(seed, unit_specs_list):
    """Custom specs define layer-0 input edges and final output edges.

    For N_HIDDEN_LAYERS > 1, matching-index bridge edges h_l -> h_{l+1}
    are inserted so every custom unit becomes a path through the hidden stack.
    """
    if len(unit_specs_list) > N_HIDDEN_MAX:
        raise ValueError(f"Too many unit specs ({len(unit_specs_list)}) for N_HIDDEN_MAX={N_HIDDEN_MAX}")

    params, masks, _, _ = empty_params_and_masks()
    params_np = {k: np.array(v) for k, v in params.items()}
    masks_np = {k: np.array(v) for k, v in masks.items()}
    rng = np.random.default_rng(seed)

    for h, spec in enumerate(unit_specs_list):
        in_conns = spec.get('inputs', {})
        out_conns = spec.get('outputs', {})
        n_in_total = sum(in_conns.values()) + (1 if USE_BIAS_FEATURE else 0)
        bound = float(np.sqrt(3.0 / max(n_in_total, 1)))

        for task_id, n_conns in in_conns.items():
            task_start = task_id * INPUT_PER_TASK
            chosen = rng.choice(INPUT_PER_TASK, size=min(n_conns, INPUT_PER_TASK), replace=False)
            masks_np['M0'][h, task_start + chosen] = 1
            params_np['W0'][h, task_start + chosen] = rng.uniform(-bound, bound, size=len(chosen))

        if USE_BIAS_FEATURE:
            masks_np['M0'][h, -1] = 1
            params_np['W0'][h, -1] = float(rng.uniform(-bound, bound))

        for layer in range(1, N_HIDDEN_LAYERS):
            masks_np[f'M{layer}'][h, h] = 1
            params_np[f'W{layer}'][h, h] = 0.0

        for task_id, n_conns in out_conns.items():
            task_out_start = task_id * NUM_CLASSES
            chosen = rng.choice(NUM_CLASSES, size=min(n_conns, NUM_CLASSES), replace=False)
            masks_np[f'M{N_HIDDEN_LAYERS}'][task_out_start + chosen, h] = 1

    return ({k: jnp.array(v) for k, v in params_np.items()},
            {k: jnp.array(v) for k, v in masks_np.items()})


def init_fully_connected(seed, n_hidden_init_per_layer, fc_isolated,
                          init_conns_per_layer=None):
    """Initialize network with per-layer unit counts.

    fc_isolated=True: each task gets its own isolated sub-network, filling to
        N_HIDDEN_MAX_PER_LAYER (original behaviour, n_hidden_init_per_layer ignored).
    fc_isolated=False: n_hidden_init_per_layer[l] units at each layer.
        Edge connectivity follows init_conns_per_layer[l] (falls back to
        INIT_CONNS_PER_LAYER when not provided):
          None  -> fully connected to all upstream units/inputs.
          int   -> that many randomly chosen upstream connections per unit.
    """
    icl = INIT_CONNS_PER_LAYER if init_conns_per_layer is None else tuple(init_conns_per_layer)
    params, masks, _, _ = empty_params_and_masks()
    params_np = {k: np.array(v) for k, v in params.items()}
    masks_np = {k: np.array(v) for k, v in masks.items()}
    rng = np.random.default_rng(seed)

    if fc_isolated:
        # Fill to per-layer cap, split evenly between tasks (original behaviour)
        for layer_idx, cap in enumerate(N_HIDDEN_MAX_PER_LAYER):
            units_per_task = cap // N_TASKS
            for t in range(N_TASKS):
                h0, h1 = t * units_per_task, (t + 1) * units_per_task
                if layer_idx == 0:
                    n_ic = icl[0]
                    if n_ic is None:
                        masks_np['M0'][h0:h1, t * INPUT_PER_TASK:(t + 1) * INPUT_PER_TASK] = 1
                        if USE_BIAS_FEATURE:
                            masks_np['M0'][h0:h1, -1] = 1
                    else:
                        for h in range(h0, h1):
                            chosen = rng.choice(INPUT_PER_TASK, size=min(n_ic, INPUT_PER_TASK), replace=False)
                            masks_np['M0'][h, t * INPUT_PER_TASK + chosen] = 1
                            if USE_BIAS_FEATURE:
                                masks_np['M0'][h, -1] = 1
        for layer in range(1, N_HIDDEN_LAYERS):
            cur_upt = N_HIDDEN_MAX_PER_LAYER[layer] // N_TASKS
            prev_upt = N_HIDDEN_MAX_PER_LAYER[layer - 1] // N_TASKS
            n_ic_mid = icl[layer]
            for t in range(N_TASKS):
                h0, h1 = t * cur_upt, (t + 1) * cur_upt
                ph0, ph1 = t * prev_upt, (t + 1) * prev_upt
                if n_ic_mid is None:
                    masks_np[f'M{layer}'][h0:h1, ph0:ph1] = 1
                else:
                    prev_count = ph1 - ph0
                    for h in range(h0, h1):
                        chosen = rng.choice(prev_count, size=min(n_ic_mid, prev_count), replace=False)
                        masks_np[f'M{layer}'][h, ph0 + chosen] = 1
        for t in range(N_TASKS):
            last_upt = N_HIDDEN_MAX_PER_LAYER[-1] // N_TASKS
            h0, h1 = t * last_upt, (t + 1) * last_upt
            masks_np[f'M{N_HIDDEN_LAYERS}'][t * NUM_CLASSES:(t + 1) * NUM_CLASSES, h0:h1] = 1
    else:
        # Edge 0: input -> h0
        h0_count = n_hidden_init_per_layer[0]
        n_ic = icl[0]
        if n_ic is None:
            masks_np['M0'][:h0_count, :] = 1
        else:
            for h in range(h0_count):
                chosen = rng.choice(INPUT_DIM, size=min(n_ic, INPUT_DIM), replace=False)
                masks_np['M0'][h, chosen] = 1

        # Middle edges: h_{l-1} -> h_l
        for layer in range(1, N_HIDDEN_LAYERS):
            h_count = n_hidden_init_per_layer[layer]
            prev_h_count = n_hidden_init_per_layer[layer - 1]
            n_ic_mid = icl[layer]
            if n_ic_mid is None:
                masks_np[f'M{layer}'][:h_count, :prev_h_count] = 1
            else:
                for h in range(h_count):
                    chosen = rng.choice(prev_h_count, size=min(n_ic_mid, prev_h_count), replace=False)
                    masks_np[f'M{layer}'][h, chosen] = 1

        # Output edge: all outputs to last hidden layer's units
        masks_np[f'M{N_HIDDEN_LAYERS}'][:, :n_hidden_init_per_layer[-1]] = 1

    for i, w in enumerate(edge_names()):
        fan_in = np.maximum(masks_np[f'M{i}'].sum(axis=1, keepdims=True), 1)
        bound = np.sqrt(3.0 / fan_in)
        params_np[w] = rng.uniform(-bound, bound, size=params_np[w].shape) * masks_np[f'M{i}']
    return ({k: jnp.array(v, dtype=jnp.float32) for k, v in params_np.items()},
            {k: jnp.array(v, dtype=jnp.int32) for k, v in masks_np.items()})

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════

def _should_log(step: int, block_idx: int) -> bool:
    """Return True when this step should emit a training log line."""
    if LOG_MODE == 'block':
        return True
    return LOG_FREQ > 0 and step % LOG_FREQ == 0


@jax.jit
def _eval_batch_loss(params, masks, imgs, lbls):
    return jax.vmap(lambda x, y: loss_ce(params, masks, x, y))(imgs, lbls).mean()


@jax.jit
def _eval_batch_accuracy(params, masks, imgs, lbls):
    def one(x, y):
        raw, _ = forward(params, masks, x)
        preds = jnp.argmax(raw.reshape(N_TASKS, NUM_CLASSES), axis=-1)
        return jnp.mean(preds == y)
    return jax.vmap(one)(imgs, lbls).mean()


@jax.jit
def _eval_per_task_accuracy_full(params, masks, imgs, lbls):
    def one(x, y):
        raw, _ = forward(params, masks, x)
        preds = jnp.argmax(raw.reshape(N_TASKS, NUM_CLASSES), axis=-1)
        return (preds == y).astype(jnp.float32)
    return jax.vmap(one)(imgs, lbls).mean(axis=0)


def run_experiment(name, mnist_images, mnist_labels, *,
                   utility=None, generation='all_output',
                   optimizer=None, n_hidden_init_per_layer=None, permute_period=None,
                   seed=None, fully_connected_init=True, fc_isolated=None,
                   skip_last_cull=None,
                   aux_hebbian_mode=None,
                   cull_utility_mode=None,
                   custom_init=None,
                   n_hidden_layers=None,
                   prune_per_block=None,
                   generate_mode=None,
                   init_conns_per_layer=None,
                   learning_rate=None):
    if utility is None:
        utility = UTILITY_MODE_DEFAULT
    if optimizer is None:
        optimizer = OPTIMIZER_DEFAULT
    if permute_period is None:
        permute_period = 0
    if seed is None:
        seed = SEED
    if fc_isolated is None:
        fc_isolated = FC_ISOLATED_DEFAULT
    if skip_last_cull is None:
        skip_last_cull = SKIP_LAST_CULL
    if aux_hebbian_mode is None:
        aux_hebbian_mode = AUX_HEBBIAN_MODE_DEFAULT
    if cull_utility_mode is None:
        cull_utility_mode = CULL_UTILITY_MODE_DEFAULT
    if n_hidden_layers is None:
        n_hidden_layers = N_HIDDEN_LAYERS
    if prune_per_block is None:
        prune_per_block = PRUNE_PER_BLOCK
    if generate_mode is None:
        generate_mode = GENERATE_MODE

    if n_hidden_layers != N_HIDDEN_LAYERS:
        raise ValueError("Set the module-level N_HIDDEN_LAYERS before JIT compilation; dynamic per-run layer counts are not supported.")

    utility_mode = utility
    all_out = (generation != 'single_output')
    n_hil = n_hidden_init_per_layer if n_hidden_init_per_layer is not None else N_HIDDEN_INIT_PER_LAYER

    print(f'\n{"=" * 65}')
    print(f'{name}')
    init_desc = (f'custom({len(custom_init)} paths)' if custom_init is not None
                 else ('fc_isolated' if fully_connected_init and fc_isolated
                       else 'fc' if fully_connected_init
                       else f'random_paths({n_hil})'))
    print(f'  hidden_layers={N_HIDDEN_LAYERS} utility={utility} aux_hebbian={aux_hebbian_mode} '
          f'cull={cull_utility_mode} gen={generation} opt={optimizer} init={init_desc} '
          f'permute={permute_period} fc_isolated={fc_isolated}')

    stream = ParallelMNIST(mnist_images, mnist_labels, seed=seed, permute_period=permute_period)
    rng = jax.random.key(seed)

    params, masks, U, H = empty_params_and_masks()
    acc = jnp.array(0.0)

    if custom_init is not None:
        params, masks = init_custom_units(seed, custom_init)
    elif fully_connected_init:
        params, masks = init_fully_connected(seed, n_hil, fc_isolated,
                                              init_conns_per_layer=init_conns_per_layer)
    else:
        # Generate max(n_hil) paths so every hidden layer reaches its target count.
        for _ in range(max(n_hil)):
            rng, gk = jax.random.split(rng)
            params, masks, _, _ = generate_unit_path(params, masks, gk, all_out)

    _lr_sgd = learning_rate if learning_rate is not None else SGD_LR
    _lr_adam = learning_rate if learning_rate is not None else ADAM_LR
    _lr_adamw = learning_rate if learning_rate is not None else ADAMW_LR

    if optimizer == 'sgd':
        opt = optax.sgd(_lr_sgd)
    elif optimizer == 'adam':
        opt = optax.adam(_lr_adam)
    elif optimizer == 'adamw':
        opt = optax.adamw(_lr_adamw)
    elif optimizer == 'adamw_weight_decay':
        opt = optax.adamw(_lr_adamw, weight_decay=WEIGHT_DECAY)
    elif optimizer == 'idbd_rms_base_adam':
        opt = optax_idbd(
            meta_lr=1e-4,
            init_lr=ADAMW_LR,
            base_optimizer='adamw',
            base_optimizer_params={'beta1': 0.9, 'beta2': 0.999},
            weight_decay=WEIGHT_DECAY,
            autostep=False,
            version="prediction_grads",
            rms_style_beta_update=True,
            rms_style_beta_update_beta_rho=0.0,
        )
    elif optimizer == 'idbd_rms':
        opt = optax_idbd(
            meta_lr=IDBD_META_LR,
            init_lr=IDBD_INIT_LR,
            weight_decay=WEIGHT_DECAY,
            autostep=False,
            version="prediction_grads",
            rms_style_beta_update=True,
            rms_style_beta_update_beta_rho=0.9,
        )
    elif optimizer == 'idbd':
        opt = optax_idbd(
            meta_lr=IDBD_META_LR,
            init_lr=IDBD_INIT_LR,
            weight_decay=WEIGHT_DECAY,
            autostep=False,
            version="prediction_grads",
            rms_style_beta_update=False,
            rms_style_beta_update_beta_rho=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    opt_state = opt.init(params)
    t0 = time.time()
    n_blocks = TOTAL_STEPS // BLOCK_SIZE
    step_log: List[dict] = []

    for bi in range(n_blocks):
        imgs, lbls = stream.sample_block(BLOCK_SIZE)
        imgs_j, lbls_j = jnp.array(imgs), jnp.array(lbls)

        params, U, H, acc, opt_state = train_block_optax(
            params, masks, imgs_j, lbls_j, U, H, acc, opt_state,
            opt, utility_mode, aux_hebbian_mode, BLOCK_SIZE
        )

        is_last_block = (bi == n_blocks - 1)
        old_masks = masks
        if not (skip_last_cull and is_last_block):
            active_count = sum(m.sum() for m in masks.values()).astype(jnp.int32)
            ntp = jnp.minimum(jnp.int32(prune_per_block), active_count)
            rng, pk = jax.random.split(rng)
            cull_U = select_cull_utility(U, H, cull_utility_mode)
            masks = batch_prune(cull_U, masks, ntp, pk)
            masks = cleanup_dead(masks)
            params = apply_masks(params, masks)
            U = {w: U[w] * masks[w.replace('W', 'M')] for w in edge_names()}
            H = {w: H[w] * masks[w.replace('W', 'M')] for w in edge_names()}
            acc = acc - ntp.astype(jnp.float32)

            if CONNECTION_BUDGET_PER_LAYER is not None:
                # Tightest per-layer constraint limits the path generator.
                per_layer_avail = [
                    int(CONNECTION_BUDGET_PER_LAYER[i]) - int(masks[f'M{i}'].sum())
                    for i in range(min(len(CONNECTION_BUDGET_PER_LAYER), N_HIDDEN_LAYERS + 1))
                ]
                gen_budget = jnp.int32(max(0, min(per_layer_avail)))
            else:
                gen_budget = jnp.int32(CONNECTION_BUDGET) - sum(m.sum() for m in masks.values()).astype(jnp.int32)
            rng, gk = jax.random.split(rng)
            if generate_mode == "RANDOM_SPARSE":
                params, masks = generate_fill_sparse(params, masks, gen_budget, gk)
            elif generate_mode == "NO_GENERATE":
                pass
            else:
                params, masks = generate_fill(params, masks, gen_budget, gk, all_out)
            params = apply_masks(params, masks)
        else:
            ntp = jnp.int32(0)

        opt_state = _reset_grown_connections(opt_state, masks, old_masks)

        step = (bi + 1) * BLOCK_SIZE
        if _should_log(step, bi):
            masks_np = {k: np.array(v) for k, v in masks.items()}
            units_per_layer = []
            for h_layer in range(N_HIDDEN_LAYERS):
                alive = (masks_np[f'M{h_layer}'].sum(axis=1) + masks_np[f'M{h_layer + 1}'].sum(axis=0)) > 0
                units_per_layer.append(int(alive.sum()))
            active_units = sum(units_per_layer)
            conns_per_layer = [int(masks_np[f'M{i}'].sum()) for i in range(N_HIDDEN_LAYERS + 1)]
            nc2 = sum(conns_per_layer)
            batch_loss = float(_eval_batch_loss(params, masks, imgs_j, lbls_j))
            batch_acc = float(_eval_batch_accuracy(params, masks, imgs_j, lbls_j))
            units_str = '/'.join(str(u) for u in units_per_layer)
            conns_str = '/'.join(str(c) for c in conns_per_layer)
            print(f'  [{step:>6d}/{TOTAL_STEPS}] conns={nc2:5d}({conns_str}) units={active_units}({units_str}) pruned={int(ntp):3d} loss={batch_loss:.4f} acc={batch_acc:.4f}')
            step_metrics: dict = {
                'step': step,
                'train_loss': batch_loss,
                'train_accuracy': batch_acc,
                'n_connections': nc2,
                'n_hidden_units_total': active_units,
                'n_pruned': int(ntp),
            }
            for i, u in enumerate(units_per_layer):
                step_metrics[f'n_units_layer_{i}'] = u
            for i, c in enumerate(conns_per_layer):
                step_metrics[f'n_connections_layer_{i}'] = c
            step_log.append(step_metrics)
            if mlflow.active_run():
                mlflow.log_metrics({k: v for k, v in step_metrics.items() if k != 'step'}, step=step)

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s')

    test_imgs, test_lbls = stream.sample_block(500)
    imgs_j, lbls_j = jnp.array(test_imgs), jnp.array(test_lbls)

    per_task_losses = np.array(_eval_per_task_loss_full(params, masks, imgs_j, lbls_j))
    per_task_accs = np.array(_eval_per_task_accuracy_full(params, masks, imgs_j, lbls_j))
    final_loss = float(per_task_losses.mean())
    final_acc = float(per_task_accs.mean())
    per_task_loss_str = '  '.join(f'T{t}={per_task_losses[t]:.4f}' for t in range(N_TASKS))
    per_task_acc_str = '  '.join(f'T{t}={per_task_accs[t]:.4f}' for t in range(N_TASKS))
    print(f'  Final loss: {final_loss:.4f}  (per task: {per_task_loss_str})')
    print(f'  Final acc:  {final_acc:.4f}  (per task: {per_task_acc_str})')

    mip, mop, mie = measure_purity(masks)
    task_val = run_single_task_validation(params, masks, test_imgs, test_lbls, per_task_losses)

    if mlflow.active_run():
        masks_np_final = {k: np.array(v) for k, v in masks.items()}
        final_units_per_layer = []
        for h_layer in range(N_HIDDEN_LAYERS):
            alive = (masks_np_final[f'M{h_layer}'].sum(axis=1) + masks_np_final[f'M{h_layer + 1}'].sum(axis=0)) > 0
            final_units_per_layer.append(int(alive.sum()))
        final_conns_per_layer = [int(masks_np_final[f'M{i}'].sum()) for i in range(N_HIDDEN_LAYERS + 1)]

        final_metrics = {
            'in_purity': mip,
            'out_purity': mop,
            'in_entropy': mie,
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'final_n_connections': sum(final_conns_per_layer),
            'final_n_hidden_units_total': sum(final_units_per_layer),
        }
        for i, u in enumerate(final_units_per_layer):
            final_metrics[f'final_n_units_layer_{i}'] = u
        for i, c in enumerate(final_conns_per_layer):
            final_metrics[f'final_n_connections_layer_{i}'] = c
        for t in range(N_TASKS):
            final_metrics[f'task_{t}_final_loss'] = float(per_task_losses[t])
            final_metrics[f'task_{t}_final_accuracy'] = float(per_task_accs[t])
        for t, (loss_full, loss_isolated, delta) in enumerate(task_val):
            final_metrics[f'task_{t}_loss_full'] = loss_full
            final_metrics[f'task_{t}_loss_isolated'] = loss_isolated
            final_metrics[f'task_{t}_delta'] = delta
        mlflow.log_metrics(final_metrics)

    return {
        'ip': mip, 'op': mop, 'ie': mie,
        'fl': final_loss, 'fa': final_acc,
        'task_val': task_val,
        'step_log': step_log,
    }

# ══════════════════════════════════════════════════════════════════════
# NAMED PRESETS
# ══════════════════════════════════════════════════════════════════════

def _build_presets(custom_init_from: str = '') -> Dict[str, PresetConfig]:
    """Build the named preset catalogue.

    Must be called after ``_init_globals`` so that N_HIDDEN_LAYERS, INPUT_PER_TASK,
    NUM_CLASSES, and USE_BIAS_FEATURE reflect the current config.

    Presets
    -------
    DENSE
        5 fully-connected units (not task-isolated).  Each unit sees all inputs
        from both tasks and all output classes.  ~8k total connections.
    TASKWISE-DENSE  (formerly BLOCKSPARSE)
        5 units per task, fully connected within each task, isolated across tasks.
        2 × 5 = 10 units total.  ~7.9k connections.
    SPARSE
        10 units (not task-isolated).  Each unit sees ~785 random inputs drawn
        from the full cross-task input space and is fully connected to all outputs.
    CUSTOM
        Uses the pruner + generator with all training-loop globals from the config.
        ``custom_init_from`` (one of 'DENSE', 'TASKWISE-DENSE', 'SPARSE', or '')
        controls which baseline's initialisation is borrowed; '' = config defaults.
    """
    bias_d = 1 if USE_BIAS_FEATURE else 0
    units_per_task = 5
    sparse_n = units_per_task * N_TASKS  # 10

    # TASKWISE-DENSE: task-isolated units built via custom_init
    taskwise_init = (
        unit_specs(units_per_task, n_in_t0=INPUT_PER_TASK, n_out_t0=NUM_CLASSES)
        + unit_specs(units_per_task, n_in_t1=INPUT_PER_TASK, n_out_t1=NUM_CLASSES)
    )

    presets: Dict[str, PresetConfig] = {
        'DENSE': PresetConfig(
            name='DENSE',
            description=(
                f'{units_per_task} fully-connected units per hidden layer, '
                'cross-task (not task-isolated)'
            ),
            run_kwargs=dict(
                fully_connected_init=True,
                fc_isolated=False,
                n_hidden_init_per_layer=tuple([units_per_task] * N_HIDDEN_LAYERS),
                init_conns_per_layer=tuple(None for _ in range(N_HIDDEN_LAYERS)),
            ),
            prune_per_block=0,
            generate_mode='NO_GENERATE',
        ),
        'TASKWISE-DENSE': PresetConfig(
            name='TASKWISE-DENSE',
            description=(
                f'{units_per_task} fully-connected units per task, '
                'task-isolated (no cross-task connections)'
            ),
            run_kwargs=dict(
                custom_init=taskwise_init,
            ),
            prune_per_block=0,
            generate_mode='NO_GENERATE',
        ),
        'SPARSE': PresetConfig(
            name='SPARSE',
            description=(
                f'{sparse_n} units per hidden layer, '
                f'~{INPUT_PER_TASK + bias_d} random inputs each from full input space, '
                'fully connected to all outputs'
            ),
            run_kwargs=dict(
                fully_connected_init=True,
                fc_isolated=False,
                n_hidden_init_per_layer=tuple([sparse_n] * N_HIDDEN_LAYERS),
                # Each unit draws INPUT_PER_TASK + bias connections randomly from
                # the full cross-task input (not restricted to one task's slice).
                init_conns_per_layer=(INPUT_PER_TASK + bias_d, None),
            ),
            prune_per_block=0,
            generate_mode='NO_GENERATE',
        ),
    }

    # CUSTOM: pruner + generator, optionally borrowing a baseline's initialisation
    init_from = custom_init_from.strip() if custom_init_from else ''
    _init_keys = ('fully_connected_init', 'fc_isolated', 'n_hidden_init_per_layer',
                  'custom_init', 'init_conns_per_layer')
    if init_from and init_from in presets:
        custom_kw = {k: v for k, v in presets[init_from].run_kwargs.items()
                     if k in _init_keys}
        desc = f'Pruner+generator, initialised from {init_from}'
    else:
        custom_kw = {}
        desc = 'Pruner+generator, global config defaults'
    presets['CUSTOM'] = PresetConfig(
        name='CUSTOM',
        description=desc,
        run_kwargs=custom_kw,
        prune_per_block=None,  # use global PRUNE_PER_BLOCK
        generate_mode=None,    # use global GENERATE_MODE
    )

    return presets


def _log_aggregated_metrics(results: List[dict]) -> None:
    """Log mean ± std of step-level and final metrics across seeds to the active MLflow run.

    Step-level series: logged as ``<metric>_mean`` / ``<metric>_std`` at each step.
    Final metrics: logged once as ``final_<metric>_mean`` / ``final_<metric>_std``.
    """
    if not results:
        return

    # ── Aggregate step-level time series ─────────────────────────────────────
    all_step_logs = [r.get('step_log', []) for r in results]
    non_empty = [sl for sl in all_step_logs if sl]
    if non_empty:
        n_points = min(len(sl) for sl in non_empty)
        metric_keys = [k for k in non_empty[0][0] if k != 'step']
        for point_i in range(n_points):
            step = non_empty[0][point_i]['step']
            agg: dict = {}
            for k in metric_keys:
                vals = [sl[point_i][k] for sl in non_empty if point_i < len(sl)]
                agg[f'{k}_mean'] = float(np.mean(vals))
                if len(vals) > 1:
                    agg[f'{k}_std'] = float(np.std(vals))
            mlflow.log_metrics(agg, step=step)

    # ── Aggregate final scalars ───────────────────────────────────────────────
    final_keys = ['fl', 'fa', 'ip', 'op', 'ie']
    final_agg: dict = {}
    for k in final_keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            final_agg[f'final_{k}_mean'] = float(np.mean(vals))
            if len(vals) > 1:
                final_agg[f'final_{k}_std'] = float(np.std(vals))
    if final_agg:
        mlflow.log_metrics(final_agg)


def run_preset_seeds(
    preset: PresetConfig,
    seeds: List[int],
    mnist_images,
    mnist_labels,
    use_mlflow: bool,
    *,
    learning_rate: Optional[float] = None,
    sweep_mode: bool = False,
) -> List[dict]:
    """Run ``preset`` once per seed.

    MLflow structure (when ``use_mlflow=True`` and ``sweep_mode=False``):
    - Parent run per preset: config params + aggregated step/final metrics.
    - Child run per seed (nested=True): per-step and final metrics.

    When ``sweep_mode=True`` the caller (e.g. mlflow_sweeper) owns the active
    run.  Each seed is logged as a nested child of that run; parent-level params
    and aggregated time-series are skipped to avoid conflicts across presets.

    ``learning_rate`` overrides the global optimizer LR for this call.
    """
    all_results: List[dict] = []

    if sweep_mode:
        parent_ctx: Any = _null_context()
    else:
        parent_run_name = (
            f"{GLOBAL_RUN_NAME} | {preset.name}" if GLOBAL_RUN_NAME else preset.name
        )
        parent_ctx = mlflow.start_run(run_name=parent_run_name) if use_mlflow else _null_context()

    with parent_ctx:
        if not sweep_mode and use_mlflow:
            mlflow.log_params({
                'preset': preset.name,
                'n_seeds': len(seeds),
                'seeds': str(seeds),
                'n_tasks': N_TASKS,
                'n_hidden_layers': N_HIDDEN_LAYERS,
                'n_hidden_max_per_layer': str(N_HIDDEN_MAX_PER_LAYER),
                'block_size': BLOCK_SIZE,
                'total_steps': TOTAL_STEPS,
                'log_freq': LOG_FREQ,
                'log_mode': LOG_MODE,
                'prune_per_block': preset.prune_per_block if preset.prune_per_block is not None else PRUNE_PER_BLOCK,
                'generate_mode': preset.generate_mode if preset.generate_mode is not None else GENERATE_MODE,
                'connection_budget': CONNECTION_BUDGET,
                'optimizer': OPTIMIZER_DEFAULT,
                'learning_rate': learning_rate if learning_rate is not None else SGD_LR,
                'use_bias_feature': USE_BIAS_FEATURE,
            })

        for seed in seeds:
            child_run_name = f"{preset.name} | seed={seed}"
            child_ctx = (
                mlflow.start_run(run_name=child_run_name, nested=True)
                if use_mlflow else _null_context()
            )
            kw = {
                **preset.run_kwargs,
                'seed': seed,
                'prune_per_block': preset.prune_per_block,
                'generate_mode': preset.generate_mode,
                'learning_rate': learning_rate,
            }
            with child_ctx:
                if use_mlflow:
                    mlflow.log_params({
                        'preset': preset.name,
                        'seed': seed,
                        'learning_rate': learning_rate if learning_rate is not None else SGD_LR,
                    })
                result = run_experiment(
                    f"{preset.name} seed={seed}",
                    mnist_images, mnist_labels,
                    **kw,
                )
            all_results.append({**result, 'seed': seed})

        # Aggregate across seeds and log to the active run.
        # In sweep mode the active run is the sweeper's trial run; step-level
        # time-series are skipped there to keep the trial run lightweight.
        if use_mlflow and all_results and not sweep_mode:
            _log_aggregated_metrics(all_results)

    return all_results


# ══════════════════════════════════════════════════════════════════════
# SWEEP INTEGRATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def _get_optuna_uri() -> str:
    """Resolve the Optuna storage URI from the environment or a local default."""
    import os
    uri = os.environ.get('OPTUNA_TRACKING_URI', 'sqlite:///lr_sweep_optuna.db')
    prefix = 'sqlite:///'
    if uri.startswith(prefix) and not os.path.isabs(uri[len(prefix):]):
        uri = f'sqlite:///{os.path.abspath(uri[len(prefix):])}'
    return uri


def _load_best_lr_from_study(study_name: str, optuna_storage: str) -> Optional[float]:
    """Return the best LR from a completed Optuna study, or None if unavailable.

    The mlflow_sweeper stores studies as "{experiment}/{sweep_name}".  If
    ``study_name`` has no slash the experiment prefix is added automatically.
    """
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    if '/' not in study_name:
        study_name = f'{MLFLOW_EXPERIMENT_NAME}/{study_name}'
    try:
        study = _optuna.load_study(study_name=study_name, storage=optuna_storage)
    except Exception as e:
        print(f"[sweep] Could not load study '{study_name}': {e}")
        available = _optuna.get_all_study_names(optuna_storage)
        if available:
            print(f"[sweep] Available studies: {available}")
        return None
    completed = [
        t for t in study.trials
        if t.state.name == 'COMPLETE' and t.value is not None and 'lr' in t.params
    ]
    if not completed:
        print(f"[sweep] No completed trials with 'lr' param in study '{study_name}'.")
        return None
    best = min(completed, key=lambda t: t.value)
    import math as _math
    print(f"[sweep] Best LR from '{study_name}': {best.params['lr']:.6g} "
          f"(2^{_math.log2(best.params['lr']):+.1f}, loss={best.value:.4f})")
    return float(best.params['lr'])


# ══════════════════════════════════════════════════════════════════════
# CONFIG → GLOBALS
# ══════════════════════════════════════════════════════════════════════

def _init_globals(cfg: DictConfig) -> None:
    """Set all module-level constants from a Hydra DictConfig.

    Must be called before any JIT-compiled function is first invoked.
    """
    global N_TASKS, NUM_CLASSES, INPUT_PER_TASK, BIAS_DIM, INPUT_DIM, OUTPUT_DIM
    global N_HIDDEN_LAYERS, N_HIDDEN_MAX_PER_LAYER, HIDDEN_SIZES, N_HIDDEN_MAX
    global N_HIDDEN_INIT_PER_LAYER, INIT_CONNS_PER_LAYER
    global GEN_UNITS_PER_LAYER, GEN_CONNS_PER_LAYER, MAX_CONNS
    global LEAKY_RELU_SLOPE, USE_BIAS_FEATURE, PER_PIXEL_NORMALIZE
    global EMA_DECAY, PRUNE_RATE
    global BLOCK_SIZE, TOTAL_STEPS, CONNECTION_BUDGET, CONNECTION_BUDGET_PER_LAYER
    global PRUNE_PER_BLOCK, MAX_GEN_PER_STEP
    global SGD_LR, ADAM_LR, ADAMW_LR, IDBD_INIT_LR, IDBD_META_LR, WEIGHT_DECAY
    global GENERATE_MODE, FC_ISOLATED_DEFAULT, SKIP_LAST_CULL
    global UTILITY_MODE_DEFAULT, OPTIMIZER_DEFAULT
    global AUX_HEBBIAN_MODE_DEFAULT, CULL_UTILITY_MODE_DEFAULT
    global SEED, PER_UNIT_LOG_MAX_UNITS
    global MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, GLOBAL_RUN_NAME
    global LOG_FREQ, LOG_MODE

    N_TASKS          = cfg.dataset.n_tasks
    NUM_CLASSES      = cfg.dataset.num_classes
    INPUT_PER_TASK   = cfg.dataset.input_per_task
    USE_BIAS_FEATURE = cfg.model.use_bias_feature
    PER_PIXEL_NORMALIZE = cfg.dataset.per_pixel_normalize
    BIAS_DIM         = 1 if USE_BIAS_FEATURE else 0
    INPUT_DIM        = INPUT_PER_TASK * N_TASKS + BIAS_DIM
    OUTPUT_DIM       = NUM_CLASSES * N_TASKS

    N_HIDDEN_LAYERS        = cfg.model.n_hidden_layers
    N_HIDDEN_MAX_PER_LAYER = tuple(cfg.model.n_hidden_max_per_layer)
    assert len(N_HIDDEN_MAX_PER_LAYER) == N_HIDDEN_LAYERS, (
        f"n_hidden_max_per_layer length {len(N_HIDDEN_MAX_PER_LAYER)} != n_hidden_layers {N_HIDDEN_LAYERS}")
    HIDDEN_SIZES           = N_HIDDEN_MAX_PER_LAYER
    N_HIDDEN_MAX           = max(N_HIDDEN_MAX_PER_LAYER)
    N_HIDDEN_INIT_PER_LAYER = tuple(cfg.model.n_hidden_init_per_layer)
    INIT_CONNS_PER_LAYER   = tuple(
        None if v is None else int(v) for v in cfg.model.init_conns_per_layer
    )
    LEAKY_RELU_SLOPE       = cfg.model.leaky_relu_slope

    MAX_CONNS              = cfg.structure_tracker.max_conns
    CONNECTION_BUDGET      = cfg.structure_tracker.connection_budget
    CONNECTION_BUDGET_PER_LAYER = (
        None if cfg.structure_tracker.connection_budget_per_layer is None
        else tuple(cfg.structure_tracker.connection_budget_per_layer)
    )
    PRUNE_PER_BLOCK        = cfg.structure_tracker.prune_per_block
    PRUNE_RATE             = cfg.structure_tracker.prune_rate
    EMA_DECAY              = cfg.structure_tracker.decay_rate
    MAX_GEN_PER_STEP       = cfg.structure_tracker.max_new_units_per_step
    BLOCK_SIZE             = cfg.structure_tracker.prune_frequency
    GENERATE_MODE          = cfg.structure_tracker.generate_strategy.upper()
    SKIP_LAST_CULL         = cfg.structure_tracker.skip_last_cull
    FC_ISOLATED_DEFAULT    = cfg.structure_tracker.fc_isolated
    GEN_UNITS_PER_LAYER    = tuple(cfg.structure_tracker.gen_units_per_layer)
    GEN_CONNS_PER_LAYER    = tuple(
        None if v is None else int(v) for v in cfg.structure_tracker.gen_conns_per_layer
    )

    TOTAL_STEPS            = cfg.train.total_steps

    OPTIMIZER_DEFAULT      = cfg.optimizer.name
    SGD_LR                 = cfg.optimizer.sgd_lr
    ADAM_LR                = cfg.optimizer.adam_lr
    ADAMW_LR               = cfg.optimizer.adamw_lr
    IDBD_INIT_LR           = cfg.optimizer.idbd_init_lr
    IDBD_META_LR           = cfg.optimizer.meta_learning_rate
    WEIGHT_DECAY           = cfg.optimizer.weight_decay

    UTILITY_MODE_DEFAULT   = cfg.utility.mode
    AUX_HEBBIAN_MODE_DEFAULT  = cfg.utility.aux_hebbian_mode
    CULL_UTILITY_MODE_DEFAULT = cfg.utility.cull_utility_mode

    SEED                   = cfg.seed
    PER_UNIT_LOG_MAX_UNITS = cfg.per_unit_log_max_units
    MLFLOW_TRACKING_URI    = cfg.mlflow_tracking_uri
    MLFLOW_EXPERIMENT_NAME = cfg.mlflow_experiment_name
    GLOBAL_RUN_NAME        = cfg.mlflow_run_name

    LOG_FREQ = int(cfg.train.log_freq)
    LOG_MODE = str(cfg.train.get('log_mode', 'interval'))


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

@hydra.main(config_path='conf', config_name='config_custom_tuned_lr', version_base='1.1')
def main(cfg: DictConfig) -> None:
    _init_globals(cfg)

    # ── Preset × seed sweep ──────────────────────────────────────────────
    preset_names: List[str] = list(OmegaConf.to_container(cfg.get('run_presets', []), resolve=True))
    seeds: List[int] = list(OmegaConf.to_container(cfg.get('seeds', [SEED]), resolve=True)) or [SEED]
    custom_init_from: str = str(cfg.get('custom_init_from', ''))

    print("Loading MNIST...")
    mnist_images, mnist_labels = load_mnist()
    print(f"Loaded {len(mnist_images)} images, shape={mnist_images.shape}")

    if cfg.mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Auto-load best LR from sweep study (if configured) ──────────────
    global SGD_LR, ADAM_LR, ADAMW_LR
    _sweep_study_name = str(OmegaConf.select(cfg, 'sweep.study_name', default='') or '')
    if _sweep_study_name:
        _best_lr = _load_best_lr_from_study(_sweep_study_name, _get_optuna_uri())
        if _best_lr is not None:
            SGD_LR = ADAM_LR = ADAMW_LR = _best_lr

    if preset_names:
        presets = _build_presets(custom_init_from=custom_init_from)
        unknown = [p for p in preset_names if p not in presets]
        if unknown:
            raise ValueError(f"Unknown preset(s): {unknown}. Available: {list(presets)}")

        all_preset_results: Dict[str, List[dict]] = {}
        for pname in preset_names:
            preset = presets[pname]
            print(f"\n{'='*70}\nPreset: {preset.name}  |  seeds={seeds}")
            print(f"  {preset.description}")
            results = run_preset_seeds(preset, seeds, mnist_images, mnist_labels, cfg.mlflow)
            all_preset_results[pname] = results

        # ── Cross-preset summary ─────────────────────────────────────────
        print(f'\n{"="*85}')
        print('PRESET SWEEP SUMMARY  (mean across seeds)')
        print(f'{"="*85}')
        print(f'{"Preset":<20} {"Seeds":>5}  {"InPur":>6} {"OutPur":>7} {"InEnt":>6} {"Loss":>8} {"Acc":>7}')
        print('-' * 60)
        for pname, results in all_preset_results.items():
            n = len(results)
            mip = sum(r['ip'] for r in results) / n
            mop = sum(r['op'] for r in results) / n
            mie = sum(r['ie'] for r in results) / n
            mfl = sum(r['fl'] for r in results) / n
            mfa = sum(r['fa'] for r in results) / n
            flag = ' <--' if mip < 0.8 else ''
            print(f'{pname:<20} {n:>5}  {mip:>6.3f} {mop:>7.3f} {mie:>6.3f} {mfl:>8.4f} {mfa:>7.4f}{flag}')
        return

    # ── Legacy single-run path (no run_presets in config) ────────────────
    experiments = [
        # ('1. Baseline (contrib, all_out, SGD, stationary)', {}),
        ('Baseline', {}),

    #  ('baseline - hebbian',
    #  dict(
    #     # utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='product',
    #       ),
    # ),

   

    # # # ('5. IDBD optimizer', dict(optimizer='idbd')),
    # # ('2. Fractional propagated', dict(utility='fractional_propagated')),
    # ('IDBD',
    #  dict(optimizer='idbd')),
    # # ('4. Adam optimizer',
    # #  dict(optimizer='adam')),

    # ('Baseline', {}),

    # ('Adamw optimizer + contribution utility',
    #  dict(optimizer='adamw')),

    # ('Adamw + edan cross-entropy utility',
    #  dict(optimizer='adamw', utility='edan_cross_entropy')),

    # ('Adamw_weight_decay + contribution utility',
    #  dict(optimizer='adamw_weight_decay')),

    # # ('Adamw_weight_decay + edan cross-entropy utility',
    # #  dict(optimizer='adamw_weight_decay', utility='edan_cross_entropy')),
    # # ('Adamw_weight_decay + fractional propagated',
    # #  dict(optimizer='adamw_weight_decay', utility='fractional_propagated')),

    # ('Adamw_weight_decay + fractional propagated, hebbian only',
    #  dict(optimizer='adamw_weight_decay', 
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='hebbian', 
    # )),
    # ('Adamw_weight_decay + contribution, hebbian only',
    #  dict(optimizer='adamw_weight_decay', 
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='hebbian', 
    # )),



    # ('Adamw_weight_decay + fractional propagated, hebbian product',
    #  dict(optimizer='adamw_weight_decay', 
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='product', 
    # )),

    # # ('IDBD + fractional propagated',
    # #  dict(optimizer='idbd', utility='fractional_propagated')),
    # # ('Adamw_weight_decay optimizer, fractional propagated',
    # #  dict(optimizer='adamw_weight_decay', utility='fractional_propagated')),


    # ('idbd_rms optimizer',
    #  dict(optimizer='idbd_rms')),
    # ('idbd_rms optimizer, fractional propagated',
    #  dict(optimizer='idbd_rms', utility='fractional_propagated')),

    # ('idbd_rms_base_adam optimizer',
    #  dict(optimizer='idbd_rms_base_adam')),


    # ('idbd_rms_base_adam optimizer + contribution',
    #  dict(optimizer='idbd_rms_base_adam')),

#     # # ('idbd_rms_base_adam optimizer + edan cross-entropy',
#     # #  dict(optimizer='idbd_rms_base_adam', utility='edan_cross_entropy')),
#     ('idbd_rms_base_adam optimizer, fractional propagated',
#      dict(optimizer='idbd_rms_base_adam', utility='fractional_propagated')),

    # ('idbd_rms_base_adam optimizer, fractional propagated, hebbian product',
    #  dict(
    # optimizer='idbd_rms_base_adam',
    # utility='fractional_propagated',
    # aux_hebbian_mode='post_activation',
    # cull_utility_mode='product',
    # )),

#     ('idbd_rms_base_adam optimizer, fractional propagated, hebbian only',
#      dict(
#     optimizer='idbd_rms_base_adam',
#     utility='fractional_propagated',
#     aux_hebbian_mode='post_activation',
#     cull_utility_mode='hebbian',
# )),

#     ('idbd_rms_base_adam optimizer, fractional propagated, hebbian product',
#      dict(
#     optimizer='idbd_rms_base_adam',
#     utility='fractional_propagated',
#     aux_hebbian_mode='post_activation',
#     cull_utility_mode='product',
# )),

#     ('idbd_rms_base_adam optimizer, contribution, hebbian only',
#      dict(
#     optimizer='idbd_rms_base_adam',
#     aux_hebbian_mode='post_activation',
#     cull_utility_mode='hebbian',
# )),

#     ('idbd_rms_base_adam optimizer, contribution, hebbian product',
#      dict(
#     optimizer='idbd_rms_base_adam',
#     aux_hebbian_mode='post_activation',
#     cull_utility_mode='product',
# )),

    # ('2. Propagated utility',
    #  dict(utility='propagated')),
    # ('3. Single output gen',
    #  dict(generation='single_output')),


    # ('Adam optimizer, fractional propagated',
    #  dict(optimizer='adam', utility='fractional_propagated')),

    # ('4. IDBD + fractional propagated, no weight decay',
    #  dict(optimizer='idbd', utility='fractional_propagated')),

    

    # ('4. IDBD + fractional propagated, weight decay',
    #  dict(optimizer='idbd', utility='fractional_propagated')),
     

    # ('5. Non-stationary (permute=2000)',
    #  dict(permute_period=2000)),
    # ('6. Empty init',
    #  dict(n_hidden_init=0)),
    # ('7. Propagated + single out + Adam',
    #  dict(utility='propagated', generation='single_output', optimizer='adam')),
    # ('8. All combined (script-like)',
    #  dict(utility='propagated', generation='single_output', optimizer='adam',
    #       n_hidden_init=0, permute_period=2000)),

    # ── Custom-init progression ──────────────────────────────────────────
    # # # 1. Baseline: no hidden units at all
    # ('custom_init: no units (baseline)',
    #  dict(custom_init=[])),

    # # # 2. Single unit, 1 random input from task-0, 1 random output to task-0
    # ('1 unit, 1-in-t0 / 1-out-t0',
    #  dict(custom_init=unit_specs(1, n_in_t0=1, n_out_t0=1))),

    # # # 3. Single unit, n random inputs from task-0, n random outputs to task-0
    # # # ('custom_init: 1 unit, 10-in-t0 / 10-out-t0',
    # # # ('1 unit, 784-input-t0 / 10-out-t0',
    # ('1 unit, 784-input-t0 / 1-out-t0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=1))),

    # 4. Single unit spanning both tasks
    #    n inputs/outputs task-0, m inputs task-1, j outputs task-1
    # ('1 cross-task unit, 784 input on both, output only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=1, n_in_t1=784, n_out_t1=0)),
    # ),

    # ('1 cross-task unit, 784 input on only task 0, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=0, n_out_t1=0)),
    # ),

    # ('1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0)),
    # ),

    # ('fractional propagated: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    # #     aux_hebbian_mode='post_activation',
    # #     cull_utility_mode='product',
    #       ),
    # ),

    # ('fractional hebbian only: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='hebbian',
    #     ),
    # ),
    # ('fractional hebbian product: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='product',
    #     ),
    # ),
    # ('contribution hebbian only: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     # utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='hebbian',
    #     ),
    # ),

    # ('adamw: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     optimizer='adamw_weight_decay',
    #  ),
    # ),

    # ('adamw fractional propagated: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    #     optimizer='adamw_weight_decay',
    # #     aux_hebbian_mode='post_activation',
    # #     cull_utility_mode='product',
    #       ),
    # ),

    # ('adamw fractional hebbian only: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='hebbian',
    #     optimizer='adamw_weight_decay',
    #     ),
    # ),
    # ('adamw fractional hebbian product: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='product',
    #     optimizer='adamw_weight_decay',
    #     ),
    # ),
    # ('adamw contribution hebbian only: 1 cross-task unit, 784 input on both, 10 outputs only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=10, n_in_t1=784, n_out_t1=0),
    #     # utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='hebbian',
    #     optimizer='adamw_weight_decay',
    #     ),
    # ),


    # ('fractional: 1 cross-task unit, 784 input on both, output only on task 0',
    #  dict(custom_init=unit_specs(1, n_in_t0=784, n_out_t0=1, n_in_t1=784, n_out_t1=0),
    #     utility='fractional_propagated',
    #     aux_hebbian_mode='post_activation',
    #     cull_utility_mode='product',
    #     ),
    # ),
     

    # 5. Multiple units all with same cross-task pattern (task-0 primary)
    # ('custom_init: 5 cross-task units (t0 primary)',
    #  dict(custom_init=unit_specs(5, n_in_t0=10, n_out_t0=5, n_in_t1=8, n_out_t1=4))),

    # 6. Symmetric: units for each task, each with cross-task connections
    # ('custom_init: 5 t0-primary + 5 t1-primary units',
    #  dict(custom_init=(
    #      unit_specs(5, n_in_t0=10, n_out_t0=5, n_in_t1=8,  n_out_t1=4) +
    #      unit_specs(5, n_in_t1=10, n_out_t1=5, n_in_t0=8,  n_out_t0=4)
    #  ))),
    ]

    results = {}
    for name, kw in experiments:
        run_name = f"{GLOBAL_RUN_NAME} | {name}" if GLOBAL_RUN_NAME else name
        ctx = mlflow.start_run(run_name=run_name) if cfg.mlflow else _null_context()
        with ctx:
            if cfg.mlflow:
                mlflow.log_params({
                    'experiment_name': name,
                    'utility': kw.get('utility', UTILITY_MODE_DEFAULT),
                    'optimizer': kw.get('optimizer', OPTIMIZER_DEFAULT),
                    'aux_hebbian_mode': kw.get('aux_hebbian_mode', AUX_HEBBIAN_MODE_DEFAULT),
                    'cull_utility_mode': kw.get('cull_utility_mode', CULL_UTILITY_MODE_DEFAULT),
                    'n_tasks': N_TASKS,
                    'n_hidden_max_per_layer': str(N_HIDDEN_MAX_PER_LAYER),
                    'n_hidden_init_per_layer': str(N_HIDDEN_INIT_PER_LAYER),
                    'init_conns_per_layer': str(INIT_CONNS_PER_LAYER),
                    'gen_units_per_layer': str(GEN_UNITS_PER_LAYER),
                    'gen_conns_per_layer': str(GEN_CONNS_PER_LAYER),
                    'max_conns': MAX_CONNS,
                    'block_size': BLOCK_SIZE,
                    'total_steps': TOTAL_STEPS,
                    'connection_budget': CONNECTION_BUDGET,
                    'connection_budget_per_layer': str(CONNECTION_BUDGET_PER_LAYER),
                    'prune_rate': PRUNE_RATE,
                    'ema_decay': EMA_DECAY,
                    'prune_per_block': PRUNE_PER_BLOCK,
                    'seed': SEED,
                    'adamw_lr': ADAMW_LR,
                    'sgd_lr': SGD_LR,
                    'adam_lr': ADAM_LR,
                    'generate_mode': GENERATE_MODE,
                    'skip_last_cull': SKIP_LAST_CULL,
                    'use_bias_feature': USE_BIAS_FEATURE,
                    'input_per_task': INPUT_PER_TASK,
                    'num_classes': NUM_CLASSES,
                })
            res = run_experiment(name, mnist_images, mnist_labels, **kw)
        results[name] = res

    print(f'\n{"=" * 85}')
    print('SUMMARY')
    print(f'{"=" * 85}')
    print(f'{"Experiment":<55} {"InPur":>6} {"OutPur":>7} {"InEnt":>6} {"Loss":>8} {"Acc":>7}')
    print('-' * 95)
    for name, res in results.items():
        flag = ' <--' if res['ip'] < 0.8 else ''
        print(f'{name:<55} {res["ip"]:>6.3f} {res["op"]:>7.3f} {res["ie"]:>6.3f} {res["fl"]:>8.4f} {res["fa"]:>7.4f}{flag}')

    print(f'\n{"=" * 85}')
    print('SINGLE-TASK INPUT VALIDATION SUMMARY')
    print(f'  (Δloss = single-task loss minus full-input loss; pure network → Δ≈0, other_mag≈0)')
    print(f'{"=" * 85}')
    for name, res in results.items():
        print(f'\n  {name}')
        for t, (loss_full, loss_isolated, delta) in enumerate(res['task_val']):
            flag = '  <-- relies on other inputs' if delta > 0.01 else ''
            print(f'    Task {t}:  full={loss_full:.4f}  isolated={loss_isolated:.4f}  '
                  f'Δ={delta:+.4f}{flag}')


if __name__ == '__main__':
    main()




