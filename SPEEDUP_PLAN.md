The “fast” version is **not** “run multiple `w.train()` calls at the same time.” The fast version is:

> For a group of `K` clients, stack their model parameters, stack one batch from each client dataset, and perform one vectorized local-training step that updates `K` independent model copies at once.

Your current architecture has one `DynamicWorker` per honest client, initialized in `_init_workers`, and the training loop calls `w.train()` one worker at a time inside `run_experiment` . The goal is to keep the FL logic around sampling, rewards, Byzantine workers, aggregation, and tracking mostly intact, while replacing the local training implementation.

---

# 0. Current situation

Right now, one round does roughly this:

```python
prev_weights = [w.pull(None).detach().clone() for w in honest_workers]

for w in honest_workers:
    w.train()

h_weights = [w.pull(None) for w in honest_workers]
h_deltas = [
    current - previous
    for current, previous in zip(h_weights, prev_weights, strict=True)
]
```

That means:

```text
client 0 local SGD
client 1 local SGD
client 2 local SGD
...
client N local SGD
```

Sequential.

The fast version should become:

```python
prev_weights = [w.pull(None).detach().clone() for w in honest_workers]

batched_trainer.train_workers(honest_workers)

h_weights = [w.pull(None) for w in honest_workers]
h_deltas = [
    current - previous
    for current, previous in zip(h_weights, prev_weights, strict=True)
]
```

But internally, `batched_trainer.train_workers(...)` should train groups like this:

```text
clients [0, 1, 2, ..., K-1]    together
clients [K, ..., 2K-1]         together
clients [2K, ..., 3K-1]        together
...
```

Each group does vectorized local SGD.

---

# 1. What you are actually optimizing

Your current bottleneck is likely this pattern:

```text
tiny model forward
tiny model backward
tiny optimizer step
repeat for client 0

tiny model forward
tiny model backward
tiny optimizer step
repeat for client 1

...
```

For small models, the GPU is underfed. Kernel launch overhead and Python overhead dominate.

The fast version tries to turn this into:

```text
one larger forward for K clients
one larger backward for K clients
one batched parameter update for K clients
```

So instead of `K` tiny matmuls, the GPU sees larger batched matmuls.

---

# 2. Key constraint: clients have different datasets

This is fine.

You do **not** centralize the data.

For a group of `K` clients, at each local step you draw:

```text
client 0 -> batch x_0, y_0
client 1 -> batch x_1, y_1
client 2 -> batch x_2, y_2
...
client K -> batch x_K, y_K
```

Then stack them:

```python
x = torch.stack([x_0, x_1, ..., x_K])  # [K, B, ...]
y = torch.stack([y_0, y_1, ..., y_K])  # [K, B]
```

Each client still trains on its own dataset. The only thing shared is the GPU call.

---

# 3. The architectural target

Do **not** try to mutate `DynamicWorker.train()` into a spaghetti monster.

Add a new component:

```python
class BatchedLocalTrainer:
    def train_group(self, workers: list[DynamicWorker]) -> None:
        ...
```

Then in `run_experiment`, replace only the local-training part.

Current:

```python
for w in honest_workers:
    w.train()
```

Target:

```python
batched_trainer.train_workers(honest_workers)
```

Where:

```python
class BatchedLocalTrainer:
    def train_workers(self, workers):
        for group in chunks(workers, self.clients_per_batch):
            self.train_group(group)
```

This keeps your existing FL simulation structure alive.

---

# 4. Step-by-step migration plan

## Phase 1: make `DynamicWorker` expose what batched training needs

Before optimizing, inspect `DynamicWorker.train()`.

You need to understand four things:

```text
1. How does it get batches?
2. How does it compute loss?
3. How does it flatten/pull/set weights?
4. How does it record last_train_loss and last_gradient_norm?
```

Your current simulation expects workers to support:

```python
w.train()
w.pull(None)
w.aggregate(...)
w.compute_metrics_on_loader(...)
w.last_train_loss
w.last_gradient_norm
```

So the batched trainer must update the actual worker models, or at least update whatever `w.pull(None)` reads.

You want to add methods like:

```python
class DynamicWorker:
    def next_train_batch(self):
        ...

    def get_param_dict(self):
        ...

    def set_param_dict(self, params):
        ...

    def get_buffer_dict(self):
        ...
```

or, if you already flatten weights internally:

```python
class DynamicWorker:
    def get_flat_params(self) -> torch.Tensor:
        ...

    def set_flat_params(self, flat: torch.Tensor) -> None:
        ...
```

The vectorized version is much easier if you can convert between:

```text
model.state_dict() <-> parameter dict <-> stacked parameter dict
```

---

## Phase 2: force compatible client batches

For vmap-style training, every client in a group should produce batches with the same shape.

So enforce:

```text
same model architecture
same input shape
same batch size
same loss function
same number of local steps
```

Given your config already has one shared model name and one shared batch size via `_build_worker_config`, this likely matches your setup .

I would use **fixed local steps**, not local epochs.

You already have:

```python
nb_local_steps=cfg.optimization.nb_local_steps
```

in `_build_worker_config` .

Good. That is exactly what you want.

Use:

```text
every client performs S local steps
each local step uses one batch from its own loader
```

If a client’s dataloader runs out, cycle it.

---

## Phase 3: build a functional version of the model

The clean PyTorch way is `torch.func.functional_call`.

You need one reference model. For example:

```python
base_model = honest_workers[0].model
```

Then instead of calling:

```python
base_model(x)
```

you call:

```python
from torch.func import functional_call

logits = functional_call(
    base_model,
    (params, buffers),
    (x,),
)
```

Here:

```python
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())
```

For one model, `params` looks like:

```python
{
    "layer1.weight": Tensor[...],
    "layer1.bias": Tensor[...],
    ...
}
```

For `K` clients, you want stacked params:

```python
{
    "layer1.weight": Tensor[K, ...],
    "layer1.bias": Tensor[K, ...],
    ...
}
```

Same idea for buffers.

---

## Phase 4: write stack/unstack helpers

You need helpers like this:

```python
def get_params_and_buffers(model):
    params = {
        name: p.detach().clone().requires_grad_(True)
        for name, p in model.named_parameters()
    }
    buffers = {
        name: b.detach().clone()
        for name, b in model.named_buffers()
    }
    return params, buffers


def stack_state(worker_group):
    params_list = []
    buffers_list = []

    for w in worker_group:
        params, buffers = get_params_and_buffers(w.model)
        params_list.append(params)
        buffers_list.append(buffers)

    stacked_params = {
        name: torch.stack([p[name] for p in params_list], dim=0)
        for name in params_list[0]
    }

    stacked_buffers = {
        name: torch.stack([b[name] for b in buffers_list], dim=0)
        for name in buffers_list[0]
    }

    return stacked_params, stacked_buffers
```

Then after local training:

```python
def unstack_params_into_workers(stacked_params, worker_group):
    with torch.no_grad():
        for i, w in enumerate(worker_group):
            state = w.model.state_dict()

            for name, param in w.model.named_parameters():
                param.copy_(stacked_params[name][i])

            # Usually buffers are unchanged unless BatchNorm or stateful layers exist.
```

For first implementation, avoid BatchNorm if possible. BatchNorm makes client-local buffers annoying because running means/vars should be updated separately per client.

---

## Phase 5: vectorize the loss

Define a one-client loss:

```python
from torch.func import functional_call, grad, vmap


def single_client_loss(params, buffers, model, loss_fn, x, y):
    logits = functional_call(model, (params, buffers), (x,))
    return loss_fn(logits, y)
```

Then make a batched gradient function:

```python
def make_batched_grad_fn(model, loss_fn):
    def loss_for_one_client(params, buffers, x, y):
        logits = functional_call(model, (params, buffers), (x,))
        return loss_fn(logits, y)

    grad_fn = grad(loss_for_one_client)

    batched_grad_fn = vmap(
        grad_fn,
        in_dims=(0, 0, 0, 0),
    )

    return batched_grad_fn
```

This means:

```text
params:  dict[name -> Tensor[K, ...]]
buffers: dict[name -> Tensor[K, ...]]
x:       Tensor[K, B, ...]
y:       Tensor[K, B]
```

and output:

```text
grads: dict[name -> Tensor[K, ...]]
```

Each client gets its own gradient.

---

## Phase 6: implement batched SGD manually

Do not use `torch.optim` at first.

Manual SGD is much easier with stacked parameters.

For plain SGD:

```python
def sgd_step(params, grads, lr, weight_decay=0.0):
    new_params = {}

    for name in params:
        grad = grads[name]

        if weight_decay != 0.0:
            grad = grad + weight_decay * params[name]

        new_params[name] = params[name] - lr * grad

    return new_params
```

If you use momentum, you need per-client momentum buffers:

```python
momentum[name]: Tensor[K, ...]
```

Then:

```python
def sgd_momentum_step(params, grads, momentum_buffers, lr, momentum, weight_decay=0.0):
    new_params = {}
    new_momentum_buffers = {}

    for name in params:
        grad = grads[name]

        if weight_decay != 0.0:
            grad = grad + weight_decay * params[name]

        buf = momentum_buffers.get(name)
        if buf is None:
            buf = torch.zeros_like(grad)

        buf = momentum * buf + grad
        new_params[name] = params[name] - lr * buf
        new_momentum_buffers[name] = buf

    return new_params, new_momentum_buffers
```

For initial implementation, I would temporarily set:

```yaml
momentum_worker: 0.0
```

Validate correctness first. Then add momentum.

---

# 5. Minimal skeleton of `BatchedLocalTrainer`

Something like:

```python
class BatchedLocalTrainer:
    def __init__(
        self,
        model,
        loss_fn,
        clients_per_batch: int,
        local_steps: int,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        device: str = "cuda",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.clients_per_batch = clients_per_batch
        self.local_steps = local_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device

        self.grad_fn = self._make_grad_fn()

    def _make_grad_fn(self):
        def loss_for_one_client(params, buffers, x, y):
            logits = functional_call(self.model, (params, buffers), (x,))
            return self.loss_fn(logits, y)

        return vmap(
            grad(loss_for_one_client),
            in_dims=(0, 0, 0, 0),
        )

    def train_workers(self, workers):
        for group in chunks(workers, self.clients_per_batch):
            self.train_group(group)

    def train_group(self, group):
        params, buffers = stack_state(group)
        params = {k: v.to(self.device) for k, v in params.items()}
        buffers = {k: v.to(self.device) for k, v in buffers.items()}

        momentum_buffers = {
            k: torch.zeros_like(v)
            for k, v in params.items()
        }

        losses_per_worker = torch.zeros(len(group), device=self.device)

        for _ in range(self.local_steps):
            x, y = get_stacked_client_batch(group, self.device)

            grads = self.grad_fn(params, buffers, x, y)

            # Optional: compute losses separately for logging
            # You can add this later.

            if self.momentum > 0:
                params, momentum_buffers = sgd_momentum_step(
                    params,
                    grads,
                    momentum_buffers,
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            else:
                params = sgd_step(
                    params,
                    grads,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

        unstack_params_into_workers(params, group)
```

The missing piece is:

```python
get_stacked_client_batch(group, device)
```

Something like:

```python
def get_stacked_client_batch(group, device):
    xs = []
    ys = []

    for w in group:
        x, y = w.next_train_batch()
        xs.append(x.to(device, non_blocking=True))
        ys.append(y.to(device, non_blocking=True))

    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)

    return x, y
```

You may need to adapt this to your actual dataloader format.

---

# 6. Integrate with your current `run_experiment`

Your current hot path is inside:

```python
for step in range(cfg.effective_rounds + 1):
    tracker.evaluate_step(step, honest_workers)
    tracker.record_train_loss(step, honest_workers)
    if step < cfg.effective_rounds:
        prev_weights = [w.pull(None).detach().clone() for w in honest_workers]
        for w in honest_workers:
            w.train()
        tracker.record_gradient_norms(step, honest_workers)
        ...
```

Replace with:

```python
batched_trainer = BatchedLocalTrainer.from_config(
    cfg=cfg,
    reference_model=honest_workers[0].model,
    device=device,
)
```

Then:

```python
if step < cfg.effective_rounds:
    prev_weights = [w.pull(None).detach().clone() for w in honest_workers]

    batched_trainer.train_workers(honest_workers)

    tracker.record_gradient_norms(step, honest_workers)

    h_weights = [w.pull(None) for w in honest_workers]
    h_deltas = [
        current - previous
        for current, previous in zip(h_weights, prev_weights, strict=True)
    ]

    for byz in byz_workers:
        byz.inform(h_weights, step)

    _step_dynamic(...)
```

Do **not** touch `_step_dynamic` at first. It handles sampling, rewards, aggregation, and diagnostics. It is not the first bottleneck, and it mutates sampler state.

---

# 7. Correctness validation plan

This is the part juniors often skip and then summon the debugging demon.

Do not start by benchmarking. First prove that batched training is semantically equivalent.

## Test 1: one client per batch

Set:

```yaml
clients_per_batch: 1
```

The batched trainer should produce nearly the same results as the old sequential `w.train()`.

Expected:

```text
weights after one round: close
losses: close
accuracy curves: close
```

Use:

```python
torch.testing.assert_close(old_weights, new_weights, rtol=1e-5, atol=1e-6)
```

Some nondeterminism is expected on GPU, but it should be small.

## Test 2: no momentum, no weight decay

Start with:

```yaml
momentum_worker: 0.0
weight_decay: 0.0
```

Get correctness first.

## Test 3: one local step

Start with:

```yaml
nb_local_steps: 1
```

Then increase to the real value.

## Test 4: small number of clients

Try:

```yaml
nodes: 4
nb_honests: 4
clients_per_batch: 2
```

Then scale up.

## Test 5: compare deltas

After training:

```python
old_delta = old_w.pull(None) - old_prev
new_delta = new_w.pull(None) - new_prev
```

Compare deltas per client.

This matters because your reward strategy may use update cosine similarity. In `_step_dynamic`, if the reward strategy is `UpdateCosineSimilarityReward`, it scores using `h_deltas` . If deltas change unexpectedly, your bandit dynamics change.

---

# 8. Performance validation plan

Once correctness is okay, benchmark.

Track:

```text
round time
local training time
GPU utilization
GPU memory
accuracy curve
final metrics
```

Add timing around only local training:

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()

batched_trainer.train_workers(honest_workers)

if torch.cuda.is_available():
    torch.cuda.synchronize()
train_time = time.perf_counter() - t0
```

Try:

```text
clients_per_batch = 1
clients_per_batch = 2
clients_per_batch = 4
clients_per_batch = 8
clients_per_batch = 16
clients_per_batch = 32
```

Expected behavior:

```text
K=1: roughly old speed, maybe slower because vmap overhead
K=2/4/8: likely speedup
K too large: memory pressure, slower, or OOM
```

The best value is empirical.

---

# 9. Important implementation traps

## Trap 1: BatchNorm

If your small models use BatchNorm, vmap becomes annoying because BatchNorm updates running stats.

Options:

```text
Best:
    avoid BatchNorm in these small FL models

Okay:
    freeze BatchNorm buffers

More complex:
    vmap buffers and update per-client running stats correctly
```

For FL simulations with small models, I would use LayerNorm, GroupNorm, or no normalization.

## Trap 2: Dropout/randomness

If your model has dropout, each client needs independent randomness.

For first version:

```python
model.eval()
```

is not acceptable if you want training with dropout.

Better:

```text
disable dropout initially
validate deterministic models first
then add randomness handling
```

`vmap` has randomness modes, but do not start there.

## Trap 3: Dataloader output shapes

`torch.stack(xs)` requires identical shapes.

If the last batch is smaller, use:

```python
drop_last=True
```

for train loaders.

Otherwise one client may emit `[17, ...]` and another `[32, ...]`, and stacking explodes.

## Trap 4: optimizer semantics

Manual SGD must match your old optimizer.

Check whether `DynamicWorker.train()` uses:

```text
SGD?
momentum?
weight decay?
gradient clipping?
learning rate decay?
```

Your config includes learning rate decay, weight decay, momentum, and maybe clipping-related fields in `WorkerConfig` .

Implement in this order:

```text
1. SGD only
2. weight decay
3. momentum
4. learning rate decay
5. gradient clipping, if needed
```

Do not implement all at once.

## Trap 5: flattened weights

Your code often calls:

```python
w.pull(None)
```

and later stacks pulled weights for consensus drift and pairwise distances .

So after batched training, `w.pull(None)` must still return the updated model weights exactly as before.

That means your batched trainer should update the actual `w.model` parameters, not just return separate tensors and forget to write them back.

---

# 10. How to handle `last_train_loss` and `last_gradient_norm`

Your tracker expects:

```python
w.last_train_loss
w.last_gradient_norm
```

because it records:

```python
losses = [float(getattr(w, "last_train_loss", np.nan)) for w in honest_workers]
```

and:

```python
norms = [w.last_gradient_norm for w in honest_workers]
```



So the batched trainer must set them.

For gradient norm:

```python
def per_client_grad_norm(grads):
    # grads: dict[name -> Tensor[K, ...]]
    total = None

    for g in grads.values():
        sq = g.flatten(start_dim=1).pow(2).sum(dim=1)
        total = sq if total is None else total + sq

    return total.sqrt()
```

Then:

```python
grad_norms = per_client_grad_norm(grads)

for i, w in enumerate(group):
    w.last_gradient_norm = float(grad_norms[i].detach().cpu())
```

For train loss, write a vectorized loss function too:

```python
def make_batched_loss_fn(model, loss_fn):
    def loss_for_one_client(params, buffers, x, y):
        logits = functional_call(model, (params, buffers), (x,))
        return loss_fn(logits, y)

    return vmap(loss_for_one_client, in_dims=(0, 0, 0, 0))
```

Then inside training:

```python
losses = batched_loss_fn(params, buffers, x, y)
```

At the end:

```python
for i, w in enumerate(group):
    w.last_train_loss = float(last_losses[i].detach().cpu())
```

---

# 11. Recommended implementation order

I would do this exact sequence.

## PR 1: refactor without changing behavior

Add:

```python
def _train_honest_workers_sequential(honest_workers):
    for w in honest_workers:
        w.train()
```

Replace the loop with:

```python
_train_honest_workers_sequential(honest_workers)
```

No behavior change. This gives you a clean seam.

---

## PR 2: expose worker batch access

Inside `DynamicWorker`, add:

```python
def next_train_batch(self):
    ...
```

It should return the same kind of batch that `train()` uses.

Make `train()` itself use `next_train_batch()` internally.

No behavior change.

---

## PR 3: add `BatchedLocalTrainer` with `clients_per_batch=1`

Implement the functional training path, but only test with:

```yaml
clients_per_batch: 1
```

At this point, performance may be worse. That is fine. You are testing correctness.

---

## PR 4: support `clients_per_batch > 1`

Now try:

```yaml
clients_per_batch: 2
clients_per_batch: 4
```

Compare metrics.

---

## PR 5: add momentum/weight decay/lr decay parity

Once basic SGD works, match the old optimizer semantics.

---

## PR 6: benchmark and tune

Only now start caring about speed.

---



I would modify `run_experiment` like this:

```python
local_trainer = None

if cfg.optimization.local_training_backend == "batched":
    local_trainer = BatchedLocalTrainer.from_config(
        cfg=cfg,
        reference_model=honest_workers[0].model,
        device=device,
    )
```

Then replace:

```python
for w in honest_workers:
    w.train()
```

with:

```python

local_trainer.train_workers(honest_workers)
```

Everything after that remains the same:

```python
tracker.record_gradient_norms(step, honest_workers)
h_weights = [w.pull(None) for w in honest_workers]
h_deltas = [
    current - previous
    for current, previous in zip(h_weights, prev_weights, strict=True)
]
```

That keeps your Byzantine logic, reward logic, sampler diagnostics, drift tracking, and result writing untouched.

---

# 14. Mental model

The final system should look like this:

```text
FL round
│
├── evaluate workers                         unchanged
├── save train losses                        unchanged
├── pull previous weights                    unchanged
│
├── local client training                    changed
│   ├── group clients into chunks of K
│   ├── stack params
│   ├── for local step:
│   │     ├── get one batch per client
│   │     ├── stack batches
│   │     ├── vmap forward/loss/grad
│   │     └── batched SGD update
│   └── write params back into workers
│
├── compute h_weights/h_deltas               unchanged
├── inform Byzantine workers                 unchanged
├── dynamic neighbor sampling/reward/agg     unchanged
├── nonfinite checks                         unchanged
└── save snapshot                            unchanged
```

That is the senior-engineer version: isolate the hot path, preserve semantics, add a correctness harness, then optimize.

