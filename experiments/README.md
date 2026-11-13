# Experiment runner

This package contains the manifests, experiment configs, and internal adapters
for the CLINC150, CUB-200-2011, Omniglot, PHD08, and KorSynthetic plans. Launch
them through the repository's existing `train.py`; it keeps the legacy
`common` config path and dispatches manifest configs to the integrated
training utilities. The experiment implementation lives in root-level modules
and is exposed through the existing `dataset.py`, `loss.py`, `model.py`,
`mining_methods.py`, `metrics.py`, and `train_utils.py` APIs.

For the launch order, config-to-plan map, matrix generation, and exact commands,
see the [Russian runbook](RUNBOOK.md).

## Blackwell environment

The supported starter environment is Python 3.11, PyTorch 2.15.0,
TorchVision 0.29.2, and the stable CUDA 13.2 wheels. Install the matched
framework wheels first, then the experiment dependencies:

```bash
python3.11 -m venv .venv-blackwell
source .venv-blackwell/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.15.0 torchvision==0.29.2 \
  --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements-experiments.txt
```

The host driver must support CUDA 13.2; NVIDIA lists 595.45.04 as the Linux
GA driver floor. The PyTorch CUDA wheel includes the user-space runtime, so a
separate toolkit install is only needed if you later build custom CUDA
extensions. [PyTorch 2.15 lists CUDA 13.2 as its stable wheel](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[TorchVision 0.29.2 supports PyTorch 2.14 and later](https://github.com/pytorch/vision/releases),
and [NVIDIA's CUDA 13.2 release notes list the driver requirement](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/).

Check the cluster environment before a long sweep:

```bash
nvidia-smi
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, torch.cuda.is_bf16_supported())'
```

The configs use BF16, pinned memory, deterministic worker seeding per epoch,
and channels-last image tensors. Persistent workers are an opt-in because
they retain augmentation RNG state across epochs/resumes. P×K batch defaults
are starting values, not claims that they saturate a 204 GB card. Increase `classes_per_batch`,
`samples_per_class`, and evaluation batch size only after measuring throughput
and confirming the same batch protocol across compared runs. `torch.compile`
is intentionally left out of the baseline to keep compile time and graph
specialization from changing the initial runtime comparison.

Retrieval keeps the gallery embeddings on the GPU and computes query scores in
chunks. This fits the planned PHD08 class-disjoint gallery comfortably at the
configured embedding sizes; if another dataset or backbone exceeds available
memory, reduce evaluation batch size first or add a CPU/memmap gallery backend
as a separately recorded evaluation mode.

## Data manifests

Manifests generated from the datasets currently present under
`/home/arseniy/data` are stored in `experiments/manifests/`. Regenerate one
after changing the dataset or split seed:

```bash
python scripts/prepare_manifests.py --dataset cub200 --root /home/arseniy/data/CUB-200-2011 --output experiments/manifests/cub200.json --seed 20261113
python scripts/prepare_manifests.py --dataset omniglot --root /home/arseniy/data/omniglot --output experiments/manifests/omniglot.json --seed 20261113
python scripts/prepare_manifests.py --dataset phd08 --root /home/arseniy/data/PHD08_dataset --kor-root /home/arseniy/data/kor_synthetic --output experiments/manifests/phd08.json --seed 20261113
python scripts/prepare_manifests.py --dataset clinc150 --root /home/arseniy/data/clinc150 --output experiments/manifests/clinc150.json --seed 20261113
python scripts/prepare_manifests.py --dataset kor_synthetic --root /home/arseniy/data/kor_synthetic --phd-root /home/arseniy/data/PHD08_dataset --output experiments/manifests/kor_synthetic.json --seed 20261113
```

The CUB manifest follows `images.txt` and reports the eight extra JPGs that
are present outside that metadata. CLINC text duplicates are kept in the
official splits; the manifest records that policy. The KorSynthetic Hangul
protocol includes 11,172 syllables and excludes the 18 non-Hangul symbols in
the supplied alphabet. PHD08 class-disjoint splits are reproducible by class,
but the manifest explicitly does **not** claim source/family-disjointness;
that metadata is not present in the local tree.

The default PHD08 manifest records the expected 2,187 images/class but does
not walk all 5.1M files. Add `--deep-scan` to its preparation command if a full
per-class image-count audit is needed; run that I/O-heavy scan once.

PHD08 and KorSynthetic use the same direct label rule: folder `N` corresponds
to `kor_101_mfcr.json` alphabet entry `N`. No correspondence table is needed.
Manifest generation checks that each of PHD08's 2,350 folder indices resolves
to a unique Hangul symbol and that the corresponding KorSynthetic folder
exists. P5 evaluation applies the same index rule at runtime and checks the
alphabet and class-label fingerprints recorded in the manifests.

## Run a development experiment

Development configs evaluate only the reserved validation classes. They do
not run the configured test protocol automatically:

```bash
python train.py \
  --config experiments/configs/phd08_metric_dev.json \
  --data-root /home/arseniy/data/PHD08_dataset \
  --output experiments/runs/phd08/catml_random_seed20261113
```

The synthetic-to-PHD08 P5 configs train a KorSynthetic classifier and evaluate
it on the 2,350 index-aligned PHD08 classes. The 2350-class config trains on that
subset; the `full` config trains on all Hangul syllables. To start either
protocol, pass both dataset roots and explicitly enable the fixed test run:

```bash
python train.py \
  --config experiments/configs/phd08_synthetic_2350.json \
  --data-root /home/arseniy/data/kor_synthetic \
  --eval-data-root /home/arseniy/data/PHD08_dataset \
  --output experiments/runs/p5/kor2350_to_phd08 \
  --run-test
```

The P5 test split is currently all PHD08 images in the 2,350 index-aligned
classes; classification processes those images in batches without retaining
all image references or predictions in memory. This is a declared transfer
protocol, not source-disjoint evaluation.

Available starter configs cover class-disjoint retrieval, closed-set
classification, few-shot episodes, CLINC OOS rejection, and Omniglot run-folder
one-shot evaluation. `kornet2` loads the current
`zoo/KorNet2.json` architecture. CLINC uses RoBERTa-base; Hugging Face weights
are pinned to a model revision in the configs and cached by Transformers, or
set `model.local_files_only=true` to require a local snapshot. TorchVision
weights are also named explicitly; CUB includes both the ResNet-50 and ViT-B/16
development lines. Pre-cache TorchVision weights on an internet-connected
machine or set `model.weights_path` to a matching full state dict on the
cluster.

The frozen SentenceTransformer baseline is inference-only:

```bash
python train.py \
  --config experiments/configs/clinc150_sentence_transformer_frozen.json \
  --data-root /home/arseniy/data/clinc150 \
  --split validation \
  --evaluate-pretrained \
  --output experiments/runs/clinc150/st-frozen/validation.json
```

Use `--split test` for its test retrieval only after settings are fixed.
`--calibrate-with-validation` is for threshold-based open-set evaluation. The
fine-tuned MiniLM comparison is `clinc150_sentence_transformer_finetune_dev.json`.

To expand a development config into a fixed seed/loss/selector matrix, for
example:

```bash
python scripts/run_matrix.py \
  --config experiments/configs/cub200_metric_dev.json \
  --data-root /home/arseniy/data/CUB-200-2011 \
  --output-dir experiments/jobs/cub_pilot \
  --seeds 1 \
  --losses catml \
  --selectors random \
  --examples-per-class 10 \
  --epochs 1 \
  --steps-per-epoch 2
bash experiments/jobs/cub_pilot/run.sh
```

For a config with a second evaluation dataset, also pass `--eval-data-root`;
the generated `train.py` commands will include it.

`run_matrix.py` writes resolved configs, `jobs.jsonl`, and a sequential `run.sh`;
the `bash` line starts training. Use a development matrix first. A final-stage
matrix (`--stage final`) is supported for retrieval dev configs: it uses the
manifest's final-train class set and fixed last epoch. Add `--run-test` only
after validation and settings are frozen. The
runner supports the in-repository objectives and selectors, N-way/K-shot
nearest-support or prototype evaluation, CLINC known/OOS threshold
calibration, and the existing Omniglot run-folder evaluator. PFML, Proxy-AN,
full ProtoNet episode training, and exact historical Omniglot task reproduction
remain external baselines/adapters, not silently substituted by this runner.
The CUB pilot above runs two train steps plus validation; it checks the pipeline
but is not a scientific result. Increase the budget only after it completes on
the target cluster.

For a final class-disjoint model after choosing settings on validation:

```bash
python scripts/run_matrix.py \
  --config experiments/configs/phd08_metric_dev.json \
  --data-root /home/arseniy/data/PHD08_dataset \
  --output-dir experiments/jobs/phd08_final_pilot \
  --seeds 1 \
  --losses catml \
  --selectors random \
  --examples-per-class all \
  --stage final \
  --run-test
```

`--run-test` is an explicit opt-in. It calibrates/selects using validation,
then evaluates the configured final test split once and writes the per-run
result beside the checkpoint.

## CATML definition used here

`CATMLLoss`, exported by the root `loss.py`, follows the formula in
`/home/arseniy/phd/Dissertation/catml_definition.tex`:

- `h` is the encoder output; triplet distances use coordinate-wise
  `softplus(h)`.
- The positive term is the unsquared positive distance; the ranking term is
  `softplus(d_ap - d_an + margin)`.
- The class-radius term uses `h`, not the softplus-transformed representation.
- The configured default is mean over triplets and per-sample class weights
  `m_c / B`, with current differentiable batch centers.

The dissertation notes that the historical experiment's exact reductions and
center-gradient mode still need confirmation from its original run code. The
runner therefore records this reduction and center mode in every resolved
config/checkpoint instead of labelling it an exact reproduction. A nonzero
`norm_epsilon` is a smoothed-norm variant and must be reported separately.

APM updates class probabilities from train-batch radii. The default small
unobserved-class floor prevents classes from becoming permanently
unsampleable before their first radius is observed; setting it to zero matches
the unsmoothed probability rule more closely but can starve unobserved classes.
Auto-clustering uses sampled train embeddings and only selects a neighbor that
is present in the current P×K batch, with random fallback. These are explicit
online adaptations of the documented selectors.

## Limitations to keep visible

- PHD08 currently has no source-family/font identifiers in the experiment
  manifest. Its 60/20/20 image split is not described as source-disjoint.
- N-way/K-shot episodes are sampled reproducibly from the manifest seed; the
  exact support/query IDs are saved as `episodes_validation.json` and
  `episodes_test.json` so baselines can share them. CLINC OOS thresholds are
  calibrated on validation and transferred to test using the selected
  checkpoint.
- The local Omniglot `all_runs` folder is accepted by the run-folder one-shot
  adapter, but it has not been confirmed against the exact official task-file
  protocol. Do not call it an exact literature reproduction yet.
- PFML, Proxy-AN, a fully trained ProtoNet baseline, and source-aware PHD08
  grouping are external/remaining baselines; the runner does not silently
  substitute another objective for them.
- Each training config defaults to a manageable development budget. The
  planned multi-seed full matrix is generated only after the pilot budget and
  cluster throughput have been fixed.
