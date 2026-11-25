# Fine-Tuning

## Vision

For DINOv2 joint training, head-only linear probing, and classifier-then-backbone training on vision8/14/20, use the [DINOv2 guide](dinov2.md). The OpenCLIP strategies, forward modes, and regularizers below have a separate configuration path.

Vision fine-tuning uses YAML files in `src/merge_and_rebase/finetune/configs/`.

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/vision.yaml \
  --datasets CIFAR10,CIFAR100,EuroSAT
```

Use `--suite vision8` to select a named benchmark suite. Outputs default to `src/checkpoints/finetune/<model>/<pretrained>/<task>/` and include checkpoints, summaries, and append-only event logs.

Available strategies are `full`, `linear_probe`, `peft_lora`, and `subspace`. Forward modes, including `linearized_ntk`, are configured through the strategy configuration. The main vision presets include `vision.yaml`, `vision-peft.yaml`, `vision-ntk.yaml`, and two-stage variants.

### Fine-tuning methods

A method combines three blocks of `common.strategy` (plus an optional `common.regularization`):

| Key | Values |
|-----|--------|
| `strategy.name` | `full`, `linear_probe`, `peft_lora`, `subspace` |
| `strategy.forward_mode` | `standard`, `linearized_ntk` |
| `strategy.params.parameterization` | `weights` (default), `delta`; implied by `subspace` |
| `strategy.params.trainable_params` | `all_trainable` (default), `regularized_only` |
| `regularization.name` | `distillation`, `kfac_ggn`, `ekfac_ggn`, `composite` |

Every method uses the same entrypoint. Only the config changes:

```bash
python -m merge_and_rebase.finetune.train_vision --vision-config <config.yaml> --suite vision8
```

**Standard.** Non-linear fine-tuning of all weights (`full`) or of LoRA adapters (`peft_lora`), trained with cross-entropy. Presets: `vision.yaml`, `vision-peft.yaml`, `vision-vitl.yaml`, and `vision-muon.yaml` (Muon optimizer). The `vision-two-stage*.yaml` variants first fine-tune the text embeddings (`strategy.text_embeddings_finetune`) and then the image encoder.

**Linearized (NTK).** With `forward_mode: linearized_ntk`, the model is replaced by its first-order Taylor expansion around the pretrained weights θ₀: f(x; θ₀ + τ) ≈ f(x; θ₀) + J_θ f(x; θ₀)·τ. Training only happens in the tangent space, which makes the task vectors τ more disentangled when you merge them. Presets: `vision-ntk.yaml`, `vision-peft-ntk.yaml`. The cost is about one JVP per forward pass, so the presets use `batch_size: 16` with `accumulate_grad_batches: 8`, which keeps the effective batch size at 128.

**TAK (`configs/TAK/`).** Linearized fine-tuning with `parameterization: delta` (the trainable parameters are the displacement τ from θ₀) and a `kfac_ggn` regularizer. The regularizer penalizes τ under a K-FAC approximation of the Gauss-Newton/Fisher matrix, estimated on the other tasks (`train_percent` of their data, cached in `cache_dir`). The penalty strength is set by `reg_lambda`. There is one file per backbone and suite:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/TAK/vision-tak-kfac-vitb32-vision8.yaml
```

**Distillation.** The `distillation` regularizer adds an MSE (or another loss) between student and teacher activations at the configured `locations`, for example `image_features`. The teacher is either `frozen` (pretrained or `initialization.checkpoint`) or `online`: it is trained alongside the student, with its own `strategy`, `train`, and optional nested `regularization`. With `along_path.enabled`, the student is evaluated at θ₀ + α·τ, where α is sampled from `alpha_range`. Under `sampling: curriculum`, the interval starts at α = 1 and widens during training. Under `sampling: uniform`, α is drawn uniformly from the full range. This keeps the model well-behaved when the task vector is rescaled.

**DELTA (`configs/DELTA/Full FT`, `configs/DELTA/LoRA`).** A `composite` of two regularizers:

1. `distillation` from an online *linearized* teacher, itself regularized with `ekfac_ggn`.
2. An `ekfac_ggn` penalty on the student, with ImageNet21KP as the reference dataset.

The student stays non-linear (`forward_mode: standard`, `parameterization: delta`) and is distilled along the path. The LoRA variant saves only the adapters (`save_format: peft`). The directory names contain spaces, so quote the path:

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config "src/merge_and_rebase/finetune/configs/DELTA/Full FT/vision-delta-vitb32-vision8.yaml"
```

**Subspace (`configs/SUBSPACE/`).** Each eligible matrix is reparameterized as
W = W₀ + U·C·Vᵀ, with U and V frozen bases fixed before training and C the only
trainable tensor. The delta is therefore rank ≤ k with its row and column
spaces pinned, by construction rather than by penalty: where TAK penalizes
curvature-weighted drift and DELTA distils linearized behaviour, this removes
the unwanted directions from the hypothesis space instead of discouraging them.
At k = 64 on ViT-B/32 that is a few hundred thousand trainable parameters
against 85M.

The bases come from gradient statistics at θ₀. For each matrix, the Grams
A = E[GᵀG] and S = E[GGᵀ] are accumulated on the task's own data and again on a
reference set that is not one of the tasks being merged, and the bases are the
top-k solutions of the generalized eigenproblem

    A_task·v = λ·(A_ref + δ·(tr A_ref / n)·I)·v

so the directions kept are the ones the task wants *relative to* those the
pretrained model already uses generically. The ridge is scaled by the mean
eigenvalue of the denominator, so δ is scale-free; it is not cosmetic, since
where the reference energy is near zero the ratio diverges and the top
eigenvectors would be estimation noise.

The reference loader is built the same way the `kfac_ggn` and `ekfac_ggn`
regularizers build theirs, so `reference_datasets` takes the same task names.
The preset uses `ImageNet1KWds`, the ungated `clip-benchmark/wds_imagenet1k`
mirror, because `ImageNet1K` points at the gated `ILSVRC/imagenet-1k` and
`ImageNet21KP` is a large download; `ImageNet21KP` remains available and is
what DELTA and TAK use. The reference statistics use a label-free probe
rᵀf(x) with a fresh random r per batch, which is why a reference set whose
label space the task head cannot score still works — and why the mirror's
small split is enough, since a reference estimate reads a few thousand images.

| `strategy.params.subspace` | Default | Meaning |
|-----|-----|-----|
| `k` | 64 | Subspace dimension per side, clamped to each tensor's short side |
| `delta` | 0.01 | Ridge on the reference Gram, as a fraction of its mean eigenvalue |
| `batches` | 30 | Gradient batches for the task statistics |
| `reference_batches` | `batches` | Gradient batches per reference dataset |
| `reference_datasets` | — | Reference task names; required unless `no_reference` |
| `one_sided` | `false` | Restrict only the input side, so the delta is C·Vᵀ |
| `no_reference` | `false` | Ablation: use the identity instead of the reference Grams |
| `include_non_matrices` | `false` | Also train the non-2-D tensors in the trainable set |
| `cache_dir` | `<out_dir>/subspace_bases` | Where bases are cached and reused |
| `rebuild` | `false` | Recompute even when a cached file exists |

```bash
python -m merge_and_rebase.finetune.train_vision \
  --vision-config src/merge_and_rebase/finetune/configs/SUBSPACE/vision-subspace-vitb32-vision8.yaml
```

Two details worth knowing before running it. The preset uses `lr: 1.0e-4`
against `1.0e-5` for `full`: AdamW moves each coordinate by about the learning
rate regardless of gradient scale, so ‖Δ‖ grows like lr·√(#parameters) and
matching the delta scale of a full fine-tune from a few hundred thousand cores
needs roughly twenty times the rate. And `weight_decay: 0.0`, because decaying
the cores pulls the delta towards zero, which is a different regularizer from
decaying weights towards zero.

Bases are cached per task and per setting, so a resubmitted run pays the
statistics stage once. The run refuses to start if the reparameterization does
not reproduce θ₀ exactly, and the delta is folded into the weights before the
checkpoint is written, so an expert stays an ordinary state dict that every
merger consumes unchanged. `vision-subspace-noref-vitb32-vision8.yaml` is the
ablation that drops the reference set: with the identity as the denominator the
generalized problem collapses into the ordinary one and the bases become the
top-k eigenvectors of the task Grams alone, which isolates what the reference
set contributes.

The TAK and DELTA configs already set `datasets_order` for their suite (`vision8`, `vision14`, `vision20`), so `--suite` is optional. Use `--reference-suite` or `--reference-datasets` to change the regularizer's reference tasks. Use `--force-recompute` to rebuild the Fisher/K-FAC caches.

## Text

`strategy.name: subspace` is available here too, through
`finetune/subspace_text.py`. The method is the same -- every eligible matrix is
reparameterised as W = W₀ + U·C·Vᵀ with bases fixed before training -- but a
text model forces three differences, each of which changes what is measured:

- **The probe is not on the logits.** The vision reference statistics
  differentiate rᵀf(x) on a 512-dimensional embedding; a sequence-classification
  head emits three logits, and a random direction in three dimensions samples
  the Jacobian far too coarsely. The probe runs on the last hidden state,
  mean-pooled over the non-padding tokens.
- **The vocabulary embedding stays out.** It is shared by every expert and on
  T5 tied to the output head, so it belongs with the frozen remainder.
  Exclusion is by module type, not only by name.
- **The classification head trains normally.** Unlike CLIP's frozen zero-shot
  head it is randomly initialised, so there is no W₀ to stay near; the
  repository also extracts heads per task rather than merging them.

**Heads, when training one task per job.** `train_text` accumulates the heads
of the tasks it trained and writes `heads.pt` with a full overwrite, so a run
of one task per job leaves a file holding a single task and the last job to
finish wins. The merge needs one head per task, since each task has its own
label space and heads are never merged. Rebuild the file from the checkpoints
afterwards:

```bash
python -m merge_and_rebase.finetune.collect_heads \
  src/checkpoints/subspace_text/google-t5__t5-base \
  --checkpoint subspace_best_ep.pt
```

It refuses to write a partial set when `--tasks` is given, because a merge over
five of six heads is a wrong result rather than a partial one.

The reference corpus is generic text rather than one of the tasks being merged:
`reference_dataset` defaults to `wikitext` / `wikitext-103-raw-v1`. Options are
`k`, `delta`, `batches`, `reference_batches`, `reference_dataset`,
`reference_config`, `reference_split`, `reference_size`, `max_length`,
`one_sided`, `no_reference`, `cache_dir` and `rebuild`.

```bash
python -m merge_and_rebase.finetune.train_text \
  --text-config src/merge_and_rebase/finetune/configs/SUBSPACE/text-subspace-t5-base-nli6.yaml \
  --tasks rte
```

The preset raises the learning rate from 5e-6 to 5e-5 for the same reason as
the vision preset: AdamW moves each coordinate by about the learning rate
regardless of gradient scale, so ‖Δ‖ grows like lr·√(#parameters) and the cores
are about two orders of magnitude fewer than the matrices they parameterise.


```bash
python -m merge_and_rebase.finetune.train_text \
  --text-config src/merge_and_rebase/finetune/configs/text-peft.yaml \
  --suite nli6
```

Text configurations support full fine-tuning, linear probing, LoRA adapters, task heads, and PEFT export.

## Regularization

Regularizers are available in `train_vision` only; `train_text` does not read a `regularization` block. Set `common.regularization.name` to exactly one registered regularizer. The training loss per step is

```text
loss = task_loss + regularizer.apply(...)
```

| Name | Penalty |
|------|---------|
| `distillation` | Feature/logit matching against a frozen or online teacher, optionally along the path θ₀ + α·τ. |
| `kfac_ggn` | Quadratic penalty on τ under a K-FAC approximation of the GGN, estimated on reference tasks. |
| `ekfac_ggn` | Same as `kfac_ggn` with the eigenvalue-corrected (EK-FAC) approximation. |
| `composite` | Sum of several of the above (see below). |

For `kfac_ggn` and `ekfac_ggn`, the penalty scale is `reg_lambda`, with `full_block_scaler` and `projection_scaler` weighting the individual curvature terms. Curvature statistics are computed once on `train_percent` of each reference task and cached in `cache_dir`. The cache is reused until a relevant parameter changes or `force_recompute` / `--force-recompute` is set.

Reference tasks are resolved in this order: `--reference-datasets`, `--reference-suite`, `regularization.reference_datasets`, `regularization.reference_suite`, then the tasks of `--suite` (or of the training run). The task being trained is always excluded. Training on a single dataset requires an explicit reference selection.

### Composite regularizer

`composite` lets one run combine several regularizers, as DELTA does. Each entry in `regularizers` is a complete regularizer config with its own `name`:

```yaml
regularization:
  name: composite
  regularizers:
    - name: distillation
      locations: [{student: image_features, teacher: image_features, loss: mse, weight: 1.0}]
      teacher: {mode: online, ...}
    - name: ekfac_ggn
      reg_lambda: 500.0
      reference_datasets: [ImageNet21KP]
```

Semantics (`finetune/regularizers/composite.py`):

- **Loss.** The child losses are summed without an extra weight. Weight each child through its own parameters (`reg_lambda`, per-location `weight`).
- **Preparation.** Each child runs its own `finalize_model` and `prepare`, in list order, so every child keeps its own cache and reference tasks.
- **Batch overrides.** At most one child may change the training batch (for example, `distillation` with `along_path` sampling α). Two overriding children raise an error.
- **Optimizers.** Optimizers owned by children, such as an online teacher's, are collected and stepped together with the student's.
- **Checkpoints.** Extra checkpoint payloads from children are merged. Duplicate keys raise an error.

Nesting is allowed: a `distillation` teacher can carry its own `regularization` block, which is how the DELTA teacher is trained with `ekfac_ggn`.

## Logging

All entrypoints write local structured logs. Configure optional Weights & Biases logging with:

```yaml
logging:
  use_wandb: false
  project: null
  entity: null
  tags: []
  mode: online
  local_log_dir: null
  log_every_n_steps: 50
```

See [methods](methods.md) for the runtime behavior of merge, transport, and post-merge methods.
