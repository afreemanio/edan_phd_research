"""Debug task separation at MNIST scale with batch pruning + EMA utility.

Extends the toy debug script to use real MNIST data, cross-entropy loss,
partial input connectivity, and the script's batch pruning + EMA mechanism.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from phd.jax_core.optimizers.idbd_jacobian import optax_idbd
from functools import partial
import time

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
# N_TASKS = 2
N_TASKS = 1
NUM_CLASSES = 10
INPUT_PER_TASK = 784
INPUT_DIM = INPUT_PER_TASK * N_TASKS   # 1568
OUTPUT_DIM = NUM_CLASSES * N_TASKS     # 20

N_HIDDEN_MAX = 100
MAX_CONNS = 64        # input connections per unit (half of script's 128)

# Block size is how much data is streamed before a cull happens, and also how long the utility estimate trace is
# BLOCK_SIZE = 100      # = prune_frequency # ! default
BLOCK_SIZE = 500      # = prune_frequency
# BLOCK_SIZE = 5000      # = prune_frequency
# BLOCK_SIZE = 50000      # = prune_frequency
TOTAL_STEPS = 100_000
# CONNECTION_BUDGET = 3000
CONNECTION_BUDGET = 5000
# PRUNE_RATE = 0.0001 # default
PRUNE_RATE = 0.01 # default
EMA_DECAY = 0.999
MAX_GEN_PER_STEP = 20

# PRUNE_PER_BLOCK = 0 # Default  
PRUNE_PER_BLOCK = 500
# PRUNE_PER_BLOCK = 500

SGD_LR = 1e-3
ADAM_LR = 1e-3
# ADAMW_LR = 1e-3 # best on mnist
ADAMW_LR = 1e-4

SEED = 42

IDBD_INIT_LR = 1e-6
IDBD_META_LR = 1e-3
# WEIGHT_DECAY = 0.0
WEIGHT_DECAY = 0.0001

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_mnist():
    from data import load_dataset
    images, labels, _, _ = load_dataset('mnist', split='train')
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
        return xs, ys

# ══════════════════════════════════════════════════════════════════════
# NETWORK
# ══════════════════════════════════════════════════════════════════════

def forward(Wi, Wo, Mi, Mo, x):
    h = jax.nn.leaky_relu((Wi * Mi) @ x)
    return (Wo * Mo) @ h, h

def loss_ce(Wi, Wo, Mi, Mo, x, label):
    """Cross-entropy loss across parallel tasks."""
    raw, _ = forward(Wi, Wo, Mi, Mo, x)
    logits = raw.reshape(N_TASKS, NUM_CLASSES)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(label, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

# ══════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════

# def compute_utility(Wi, Wo, Mi, Mo, x, use_prop):
#     _, h = forward(Wi, Wo, Mi, Mo, x)
#     U_in_c = jnp.abs(x)[None, :] * jnp.abs(Wi) * Mi
#     U_out = jnp.abs(h)[None, :] * jnp.abs(Wo) * Mo
#     parent = U_out.sum(axis=0)
#     raw = U_in_c.sum(axis=1)
#     sc = jnp.where(jnp.abs(raw) > 1e-10, parent / raw, 1.0)
#     U_in_p = U_in_c * sc[:, None] * Mi
#     U_in = jnp.where(use_prop, U_in_p, U_in_c)
#     return U_in, U_out

def compute_utility_contribution(Wi, Wo, Mi, Mo, x):
    _, h = forward(Wi, Wo, Mi, Mo, x)
    U_in = jnp.abs(x)[None, :] * jnp.abs(Wi) * Mi
    U_out = jnp.abs(h)[None, :] * jnp.abs(Wo) * Mo
    return U_in, U_out

def compute_utility_propagated(Wi, Wo, Mi, Mo, x):
    U_in_c, U_out = compute_utility_contribution(Wi, Wo, Mi, Mo, x)
    parent = U_out.sum(axis=0)
    raw = U_in_c.sum(axis=1)
    sc = jnp.where(jnp.abs(raw) > 1e-10, parent / raw, 1.0)
    U_in = U_in_c * sc[:, None] * Mi
    return U_in, U_out

def compute_utility_fractional_propagated(Wi, Wo, Mi, Mo, x):
    _, h = forward(Wi, Wo, Mi, Mo, x)

    # Raw local contribution magnitudes
    C_in = jnp.abs(x)[None, :] * jnp.abs(Wi) * Mi
    C_out = jnp.abs(h)[None, :] * jnp.abs(Wo) * Mo

    # Output node utilities: each output gets utility 1
    U_output_node = jnp.ones((OUTPUT_DIM,))

    # Hidden -> Output edges get a fraction of each output node utility
    out_total = jnp.sum(C_out, axis=1, keepdims=True)
    out_frac = jnp.where(out_total > 1e-10, C_out / out_total, 0.0)
    U_out = out_frac * U_output_node[:, None] * Mo

    # Hidden node utility = sum of outgoing edge utilities
    U_hidden_node = jnp.sum(U_out, axis=0)

    # Input -> Hidden edges get a fraction of the hidden node utility
    in_total = jnp.sum(C_in, axis=1, keepdims=True)
    in_frac = jnp.where(in_total > 1e-10, C_in / in_total, 0.0)
    U_in = in_frac * U_hidden_node[:, None] * Mi

    return U_in, U_out

def compute_utility(Wi, Wo, Mi, Mo, x, utility_mode):
    if utility_mode == 'contribution':
        return compute_utility_contribution(Wi, Wo, Mi, Mo, x)
    if utility_mode == 'propagated':
        return compute_utility_propagated(Wi, Wo, Mi, Mo, x)
    if utility_mode == 'fractional_propagated':
        return compute_utility_fractional_propagated(Wi, Wo, Mi, Mo, x)
    raise ValueError(f"Unknown utility_mode: {utility_mode}")

def _predict_only(Wi, Wo, Mi, Mo, x):
    raw, _ = forward(Wi, Wo, Mi, Mo, x)
    return raw

def _train_step_optax(Wi, Wo, Mi, Mo, x, y, optimizer, opt_state):
    params = {"Wi": Wi, "Wo": Wo}

    def loss_from_params(params):
        return loss_ce(params["Wi"], params["Wo"], Mi, Mo, x, y)

    def pred_from_params(params):
        return _predict_only(params["Wi"], params["Wo"], Mi, Mo, x)

    loss_grads = jax.grad(loss_from_params)(params)

    # IDBD needs prediction jacobians; normal optax optimizers do not
    if hasattr(opt_state, "beta"):
        prediction_grads = jax.jacrev(pred_from_params)(params)
        optimizer_payload = (loss_grads, prediction_grads)
    else:
        optimizer_payload = loss_grads

    updates, opt_state = optimizer.update(optimizer_payload, opt_state, params)
    updates = {
        "Wi": updates["Wi"] * Mi,
        "Wo": updates["Wo"] * Mo,
    }
    new_params = optax.apply_updates(params, updates)

    Wi = new_params["Wi"] * Mi
    Wo = new_params["Wo"] * Mo
    return Wi, Wo, opt_state


# ══════════════════════════════════════════════════════════════════════
# BATCH PRUNING
# ══════════════════════════════════════════════════════════════════════

@jax.jit
def batch_prune(Ui, Uo, Mi, Mo, n_to_prune, rng):
    """Prune the n_to_prune lowest-utility connections globally."""
    hi = jnp.where(Mi.reshape(-1) == 1, Ui.reshape(-1), jnp.inf)
    ho = jnp.where(Mo.reshape(-1) == 1, Uo.reshape(-1), jnp.inf)
    all_u = jnp.concatenate([hi, ho])
    noise = jax.random.uniform(rng, all_u.shape, minval=-1e-10, maxval=1e-10)
    perturbed = jnp.where(jnp.isinf(all_u), all_u, all_u + noise)
    thresh = jnp.sort(perturbed)[n_to_prune]
    thresh = jnp.minimum(thresh, 1e30)
    active = jnp.concatenate([(Mi.reshape(-1) == 1), (Mo.reshape(-1) == 1)])
    prune = (perturbed < thresh) & active
    n_in = Mi.size
    Mi = Mi * (1 - prune[:n_in].reshape(Mi.shape).astype(jnp.int32))
    Mo = Mo * (1 - prune[n_in:].reshape(Mo.shape).astype(jnp.int32))
    return Mi, Mo

def cleanup_dead(Mi, Mo):
    alive = Mo.sum(axis=0) > 0
    return Mi * alive[:, None], Mo

# ══════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=('all_outputs',))
def generate_unit(Wi, Wo, Mi, Mo, key, all_outputs=True):
    """Generate one unit: MAX_CONNS random inputs, all or 1 output."""
    active = (Mi.sum(axis=1) + Mo.sum(axis=0)) > 0
    slot = jnp.argmin(jnp.where(~active, jnp.arange(N_HIDDEN_MAX), N_HIDDEN_MAX))
    has = jnp.any(~active)

    k1, k2, k3 = jax.random.split(key, 3)
    n_conns = min(MAX_CONNS, INPUT_DIM)
    noise = jax.random.uniform(k1, (INPUT_DIM,))
    perm = jnp.argsort(noise)
    im = jnp.zeros(INPUT_DIM, dtype=jnp.int32).at[perm[:n_conns]].set(1)
    bound = jnp.sqrt(3.0 / n_conns)
    w = jax.random.uniform(k2, (INPUT_DIM,), minval=-bound, maxval=bound) * im

    # Output connections
    om_all = jnp.ones(OUTPUT_DIM, dtype=jnp.int32)
    oi = jax.random.randint(k3, (), 0, OUTPUT_DIM)
    om_single = jnp.zeros(OUTPUT_DIM, dtype=jnp.int32).at[oi].set(1)
    om = jnp.where(all_outputs, om_all, om_single)
    cost = jnp.where(all_outputs, n_conns + OUTPUT_DIM, n_conns + 1)

    Wi = jnp.where(has, Wi.at[slot].set(w), Wi)
    Wo = jnp.where(has, Wo.at[:, slot].set(0.0), Wo)
    Mi = jnp.where(has, Mi.at[slot].set(im), Mi)
    Mo = jnp.where(has, Mo.at[:, slot].set(om), Mo)
    return Wi, Wo, Mi, Mo, has, cost

def generate_fill(Wi, Wo, Mi, Mo, budget, key, all_outputs=True):
    budget_f = float(budget)
    for _ in range(MAX_GEN_PER_STEP):
        if budget_f <= 0:
            break
        key, gk = jax.random.split(key)
        Wi, Wo, Mi, Mo, gen, cost = generate_unit(Wi, Wo, Mi, Mo, gk, all_outputs)
        if not bool(gen):
            break
        budget_f -= int(cost)
    return Wi, Wo, Mi, Mo

# ══════════════════════════════════════════════════════════════════════
# JIT TRAINING BLOCKS
# ══════════════════════════════════════════════════════════════════════

_adam = optax.adam(ADAM_LR)

# @partial(jax.jit, static_argnames=('n_steps',))
# def train_block_sgd(Wi, Wo, Mi, Mo, images, labels, Ui, Uo, acc, use_prop, n_steps):
#     n_active = (Mi.sum() + Mo.sum()).astype(jnp.float32)
#     acc_inc = PRUNE_RATE * n_active

#     def body(carry, data):
#         Wi, Wo, Ui, Uo, acc = carry
#         x, y = data
#         gi, go = jax.grad(loss_ce, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
#         Wi = Wi - SGD_LR * gi * Mi
#         Wo = Wo - SGD_LR * go * Mo
#         ui, uo = compute_utility(Wi, Wo, Mi, Mo, x, use_prop)
#         Ui = EMA_DECAY * Ui + (1 - EMA_DECAY) * ui
#         Uo = EMA_DECAY * Uo + (1 - EMA_DECAY) * uo
#         return (Wi, Wo, Ui, Uo, acc + acc_inc), None

#     (Wi, Wo, Ui, Uo, acc), _ = jax.lax.scan(
#         body, (Wi, Wo, Ui, Uo, acc), (images, labels))
#     return Wi, Wo, Ui, Uo, acc

# @partial(jax.jit, static_argnames=('n_steps',))
# def train_block_adam(Wi, Wo, Mi, Mo, images, labels, Ui, Uo, acc, ost, use_prop, n_steps):
#     n_active = (Mi.sum() + Mo.sum()).astype(jnp.float32)
#     acc_inc = PRUNE_RATE * n_active

#     def body(carry, data):
#         Wi, Wo, ost, Ui, Uo, acc = carry
#         x, y = data
#         gi, go = jax.grad(loss_ce, argnums=(0, 1))(Wi, Wo, Mi, Mo, x, y)
#         upd, ost2 = _adam.update((gi, go), ost, (Wi, Wo))
#         Wi = Wi + upd[0] * Mi
#         Wo = Wo + upd[1] * Mo
#         ui, uo = compute_utility(Wi, Wo, Mi, Mo, x, use_prop)
#         Ui = EMA_DECAY * Ui + (1 - EMA_DECAY) * ui
#         Uo = EMA_DECAY * Uo + (1 - EMA_DECAY) * uo
#         return (Wi, Wo, ost2, Ui, Uo, acc + acc_inc), None

#     (Wi, Wo, ost, Ui, Uo, acc), _ = jax.lax.scan(
#         body, (Wi, Wo, ost, Ui, Uo, acc), (images, labels))
#     return Wi, Wo, Ui, Uo, acc, ost


@partial(jax.jit, static_argnames=('optimizer', 'utility_mode', 'n_steps'))
def train_block_optax(Wi, Wo, Mi, Mo, images, labels, Ui, Uo, acc,
                      opt_state, optimizer, utility_mode, n_steps):
    n_active = (Mi.sum() + Mo.sum()).astype(jnp.float32)
    acc_inc = PRUNE_RATE * n_active

    def body(carry, data):
        Wi, Wo, Ui, Uo, acc, opt_state = carry
        x, y = data

        Wi, Wo, opt_state = _train_step_optax(
            Wi, Wo, Mi, Mo, x, y, optimizer, opt_state
        )

        ui, uo = compute_utility(Wi, Wo, Mi, Mo, x, utility_mode)
        Ui = EMA_DECAY * Ui + (1 - EMA_DECAY) * ui
        Uo = EMA_DECAY * Uo + (1 - EMA_DECAY) * uo

        return (Wi, Wo, Ui, Uo, acc + acc_inc, opt_state), None

    (Wi, Wo, Ui, Uo, acc, opt_state), _ = jax.lax.scan(
        body, (Wi, Wo, Ui, Uo, acc, opt_state), (images, labels)
    )
    return Wi, Wo, Ui, Uo, acc, opt_state

# ══════════════════════════════════════════════════════════════════════
# METRIC
# ══════════════════════════════════════════════════════════════════════

def measure_purity(Mi, Mo):
    Mi, Mo = np.array(Mi), np.array(Mo)
    rows = []
    for k in range(N_HIDDEN_MAX):
        if not (Mi[k].any() or Mo[:, k].any()):
            continue
        ic = [int(Mi[k, t * INPUT_PER_TASK:(t + 1) * INPUT_PER_TASK].sum())
              for t in range(N_TASKS)]
        oc = [int(Mo[t * NUM_CLASSES:(t + 1) * NUM_CLASSES, k].sum())
              for t in range(N_TASKS)]
        ti, to = sum(ic), sum(oc)
        ip = max(ic) / max(ti, 1)
        op = max(oc) / max(to, 1)
        ie = sum(-c / ti * np.log2(c / ti + 1e-15) for c in ic if c > 0) if ti > 0 else 0.0
        rows.append(dict(u=k, ic=ic, oc=oc, ip=ip, op=op, ie=ie, ti=ti, to=to))

    if not rows:
        print("  No active units!")
        return 0.0, 0.0, 1.0

    mip = np.mean([r['ip'] for r in rows])
    mop = np.mean([r['op'] for r in rows])
    mie = np.mean([r['ie'] for r in rows])

    # Show 5 worst and 5 best by input entropy
    rows.sort(key=lambda r: r['ie'], reverse=True)
    n_show = min(5, len(rows))
    shown = rows[:n_show]
    if len(rows) > 2 * n_show:
        shown.append(None)  # separator
    shown += rows[-n_show:]
    for r in shown:
        if r is None:
            print(f'  ... ({len(rows) - 2 * n_show} more units) ...')
            continue
        ic_s = '/'.join(str(c) for c in r['ic'])
        oc_s = '/'.join(str(c) for c in r['oc'])
        print(f"  h{r['u']:3d}: in=[{ic_s}]({r['ti']:3d}) pur={r['ip']:.2f} ent={r['ie']:.3f}"
              f" | out=[{oc_s}]({r['to']:2d}) pur={r['op']:.2f}")

    nc = int(Mi.sum() + Mo.sum())
    print(f'  Active units: {len(rows)} | Connections: {nc}')
    print(f'  >>> Purity: in={mip:.3f} out={mop:.3f} | Entropy: in={mie:.3f}')
    return mip, mop, mie

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_experiment(name, mnist_images, mnist_labels, *,
                   utility='contribution', generation='all_output',
                   optimizer='sgd', n_hidden_init=10, permute_period=0,
                   seed=SEED):
    utility_mode = utility
    all_out = (generation == 'all_output')

    print(f'\n{"=" * 65}')
    print(f'{name}')
    print(f'  utility={utility} gen={generation} opt={optimizer} '
          f'init={n_hidden_init} permute={permute_period}')

    stream = ParallelMNIST(mnist_images, mnist_labels, seed=seed,
                           permute_period=permute_period)
    rng = jax.random.key(seed)

    # Init arrays
    Wi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
    Wo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
    Mi = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM), dtype=jnp.int32)
    Mo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX), dtype=jnp.int32)
    Ui = jnp.zeros((N_HIDDEN_MAX, INPUT_DIM))
    Uo = jnp.zeros((OUTPUT_DIM, N_HIDDEN_MAX))
    acc = jnp.array(0.0)

    # Generate initial units
    for _ in range(n_hidden_init):
        rng, gk = jax.random.split(rng)
        Wi, Wo, Mi, Mo, _, _ = generate_unit(Wi, Wo, Mi, Mo, gk, all_out)

    if optimizer == 'sgd':
        opt = optax.sgd(SGD_LR)
    elif optimizer == 'adam':
        opt = optax.adam(ADAM_LR)
    elif optimizer == 'adamw':
        opt = optax.adamw(ADAMW_LR)
    elif optimizer == 'adamw_weight_decay':
        opt = optax.adamw(ADAMW_LR, weight_decay=WEIGHT_DECAY)
    elif optimizer == 'idbd_rms_base_adam':
        opt = optax_idbd(
            # meta_lr=0.0, 
            meta_lr=1e-3, 
            init_lr=ADAMW_LR, 
            base_optimizer='adamw',
            base_optimizer_params={'beta1': 0.9, 'beta2': 0.999}, 
            weight_decay=WEIGHT_DECAY,
            autostep=False,
            version="prediction_grads",
            rms_style_beta_update=True,
            rms_style_beta_update_beta_rho=0.9,
            # rms_style_beta_update_beta_rho=0.0,
            )
        # opt = optax_idbd(
        #     meta_lr=IDBD_META_LR, 
        #     init_lr=IDBD_INIT_LR, 
        #     base_optimizer='adamw',
        #     base_optimizer_params={'beta1': 0.9, 'beta2': 0.999}, 
        #     weight_decay=WEIGHT_DECAY,
        #     autostep=False,
        #     version="prediction_grads",
        #     rms_style_beta_update=True,
        #     rms_style_beta_update_beta_rho=0.9,)
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

    opt_state = opt.init({"Wi": Wi, "Wo": Wo})

    t0 = time.time()
    n_blocks = TOTAL_STEPS // BLOCK_SIZE

    for bi in range(n_blocks):
        # Sample data
        imgs, lbls = stream.sample_block(BLOCK_SIZE)
        imgs_j, lbls_j = jnp.array(imgs), jnp.array(lbls)

        # Train + accumulate
        # if use_adam:
        #     Wi, Wo, Ui, Uo, acc, ost = train_block_adam(
        #         Wi, Wo, Mi, Mo, imgs_j, lbls_j, Ui, Uo, acc, ost,
        #         use_prop, BLOCK_SIZE)
        # else:
        #     Wi, Wo, Ui, Uo, acc = train_block_sgd(
        #         Wi, Wo, Mi, Mo, imgs_j, lbls_j, Ui, Uo, acc,
        #         use_prop, BLOCK_SIZE)
        Wi, Wo, Ui, Uo, acc, opt_state = train_block_optax(
            Wi, Wo, Mi, Mo, imgs_j, lbls_j,
            Ui, Uo, acc, opt_state, opt, utility_mode, BLOCK_SIZE
        )

        # Batch prune
        # * Prune based on percent
        # n_active = int(Mi.sum() + Mo.sum())
        # ntp = int(jnp.floor(acc))
        # ntp = max(0, min(ntp, n_active)) 
        # * Prune based on static value
        n_active = int(Mi.sum() + Mo.sum())
        ntp = min(PRUNE_PER_BLOCK, n_active) 

        if ntp > 0:
            rng, pk = jax.random.split(rng)
            Mi, Mo = batch_prune(Ui, Uo, Mi, Mo, jnp.array(ntp, dtype=jnp.int32), pk)
            Ui = Ui * Mi
            Uo = Uo * Mo
        acc = acc - jnp.array(ntp, dtype=jnp.float32)

        # Cleanup dead units
        Mi, Mo = cleanup_dead(Mi, Mo)
        Ui = Ui * Mi

        # Generate to fill budget
        nc = int(Mi.sum() + Mo.sum())
        # gen_budget = ntp # * add n pruned
        gen_budget = CONNECTION_BUDGET - nc # * add to fill budget
        if gen_budget > 0:
            rng, gk = jax.random.split(rng)
            Wi, Wo, Mi, Mo = generate_fill(Wi, Wo, Mi, Mo, gen_budget, gk, all_out)

        step = (bi + 1) * BLOCK_SIZE
        if step % 10_000 == 0:
            na = int(np.array((Mi.sum(axis=1) + Mo.sum(axis=0)) > 0).sum())
            nc2 = int(Mi.sum() + Mo.sum())
            print(f'  [{step:>6d}/{TOTAL_STEPS}] conns={nc2:5d} units={na:3d} pruned={ntp:3d}')

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s')

    # Quick loss check
    test_imgs, test_lbls = stream.sample_block(200)

    @jax.jit
    def eval_loss(Wi, Wo, Mi, Mo, imgs, lbls):
        def one(x, y):
            return loss_ce(Wi, Wo, Mi, Mo, x, y)
        return jax.vmap(one)(imgs, lbls).mean()

    final_loss = float(eval_loss(Wi, Wo, Mi, Mo, jnp.array(test_imgs), jnp.array(test_lbls)))
    print(f'  Final loss: {final_loss:.4f}')

    mip, mop, mie = measure_purity(Mi, Mo)
    return mip, mop, mie, final_loss

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ('1. Baseline (contrib, all_out, SGD, stationary)', {}),
    # ('Baseline', {}),
    # # # ('5. IDBD optimizer', dict(optimizer='idbd')),
    # # ('2. Fractional propagated', dict(utility='fractional_propagated')),
    # ('IDBD',
    #  dict(optimizer='idbd')),
    # # ('4. Adam optimizer',
    # #  dict(optimizer='adam')),

    # # ('Adamw optimizer',
    # #  dict(optimizer='adamw')),
    # ('Adamw_weight_decay optimizer',
    #  dict(optimizer='adamw_weight_decay')),

    # ('IDBD + fractional propagated',
    #  dict(optimizer='idbd', utility='fractional_propagated')),
    # ('Adamw_weight_decay optimizer, fractional propagated',
    #  dict(optimizer='adamw_weight_decay', utility='fractional_propagated')),

    # ('idbd_rms_base_adam optimizer',
    #  dict(optimizer='idbd_rms_base_adam')),

    ('idbd_rms_base_adam optimizer, fractional propagated',
     dict(optimizer='idbd_rms_base_adam', utility='fractional_propagated')),
    # # ('2. Propagated utility',
    # #  dict(utility='propagated')),
    # # ('3. Single output gen',
    # #  dict(generation='single_output')),


    # ('Adam optimizer, fractional propagated',
    #  dict(optimizer='adam', utility='fractional_propagated')),
    # ('Adamw optimizer, fractional propagated',
    #  dict(optimizer='adamw', utility='fractional_propagated')),

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
]

if __name__ == '__main__':
    print("Loading MNIST...")
    mnist_images, mnist_labels = load_mnist()
    print(f"Loaded {len(mnist_images)} images, shape={mnist_images.shape}")

    results = {}
    for name, kw in EXPERIMENTS:
        ip, op, ie, fl = run_experiment(name, mnist_images, mnist_labels, **kw)
        results[name] = (ip, op, ie, fl)

    print(f'\n{"=" * 85}')
    print('SUMMARY')
    print(f'{"=" * 85}')
    print(f'{"Experiment":<55} {"InPur":>6} {"OutPur":>7} {"InEnt":>6} {"Loss":>8}')
    print('-' * 85)
    for name, (ip, op, ie, fl) in results.items():
        flag = ' <--' if ip < 0.8 else ''
        print(f'{name:<55} {ip:>6.3f} {op:>7.3f} {ie:>6.3f} {fl:>8.4f}{flag}')
