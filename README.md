# PhD metric-learning experiments

This repository has one experiment implementation and one JSON schema. The
runner reads configs from `experiments/configs/`, split definitions from
`experiments/manifests/`, and implementation code from the `experiment_*`
modules. Configs in any other schema are rejected; there is no legacy training
path.

## Environment

Use Python 3.13.6 and install the pinned environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training and evaluation require a CUDA GPU. Pretrained weights must be available
locally when a config requests them. Pass `--models-dir` to select the root for
TorchVision and Hugging Face assets.

## One run

Run one manifest config with an explicit output directory. Explicit paths keep
data, weights, and results fixed and prevent accidental overwrite:

```bash
python train.py \
  --config experiments/configs/phd08_metric_dev.json \
  --data-root /path/to/PHD08_dataset \
  --models-dir /path/to/models \
  --output experiments/runs/phd08_metric_dev/seed1
```

To run the configured test protocol after training, add `--run-test`. Resume a
run with `--resume <run-dir>/last.pt`. Pretrained-only evaluation uses the same
config and protocol:

```bash
python train.py \
  --config experiments/configs/clinc150_sentence_transformer_frozen.json \
  --data-root /path/to/clinc150 \
  --models-dir /path/to/models \
  --output experiments/runs/clinc150_frozen/validation.json \
  --evaluate-pretrained --split validation
```

## Experiment matrices

Use the batch runner for the predeclared PHD08, Omniglot, CUB-200-2011, and
CLINC150 matrices. `--dry-run` writes and displays the planned jobs without
starting training:

```bash
python scripts/run_experiment_batch.py \
  --series p1 --phase development \
  --data-root /path/to/PHD08_dataset \
  --models-dir /path/to/models \
  --seeds 1,2,3,4,5 --dry-run
```

The PHD08 queue also supports `p2` through `p5`; other series are listed in
[`experiments/RUNBOOK.md`](experiments/RUNBOOK.md). The worker records the exact
config, command, checkpoint, logs, seed, and output directory for each job.

## Canonical experiment config

Every config declares its dataset and split manifest, task, model, objective,
sampler, training, validation/test protocol, evaluation geometry, and hardware.
The supported metric objectives include `catml`, `triplet`, `contrastive`,
`supcon`, `multiple_negatives`, `multi_similarity`, and `proxy_anchor`.

Sampler settings are explicit: `positive` is `random` or `apm`; `negative` is
`random`, `semi_hard`, `batch_hard`, or `auto_clusters`. Auto-clustering builds a
class graph from training embeddings. All methods draw class-balanced P×K
batches, so they use the same batch protocol when compared in a matrix.

Do not change a split manifest, dataset inventory, objective, sampler, model, or
evaluation distance within a comparison series. Create a distinct config and
record the change as a separate experiment. The runner checks manifest hashes
and class disjointness before training.

## Dataset preparation and protocols

Manifests and their generators are in `experiments/manifests/` and
`scripts/prepare_manifests.py`. Parquet conversion/restoration tools are in
`scripts/`. Dataset layouts, exact split rules, and the research protocols are
documented in the [experiment guide](experiments/README.md), the
[runbook](experiments/RUNBOOK.md), and the four
[research plans](EXPERIMENT_PLAN_PHD08.md),
[Omniglot](EXPERIMENT_PLAN_OMNIGLOT.md), [CUB](EXPERIMENT_PLAN_CUB.md), and
[CLINC150](EXPERIMENT_PLAN_CLINC150.md).

## Checks

Run the batch-planning regression checks with:

```bash
python -m unittest tests.test_batch_integrity
```
