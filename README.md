# siamese-nets

PyTorch training code for image embeddings with triplet and contrastive objectives.
The training entry point reads a JSON config:

```bash
python train.py -c configs/kor_0.json
```

Training currently requires a CUDA device. Checkpoints, plots, and a copy of the
resolved `common` config are written under `checkpoint_pt`.
Older top-level configs in `configs/` are translated to the current schema by
`train.py`; new experiments should use the `common` schema shown in the current
configs.

## Planned research experiments

For the CLINC150, CUB-200-2011, Omniglot, PHD08, and KorSynthetic plans, the
existing `train.py` entry point also accepts manifest-based experiment
configs. The new mode provides deterministic class manifests, P×K batches,
CATML/APM/Auto-clustering,
classification and retrieval metrics, and BF16 training options. The plan
files remain the scientific protocol; the runner records its resolved config
and manifest in each run folder.

Start with [experiments/README.md](experiments/README.md) for Blackwell setup,
data manifests, and launch commands. Starter configs are in
`experiments/configs/`; generated split manifests are in
`experiments/manifests/`.

## Loss and sampler settings

Set `common.loss_settings.type` to `triplet` or `contrastive`. Contrastive mode
selects a pair dataset automatically and expects a binary pair label: `0` for
same class and `1` for different classes. `gen_imp_ratio` is the probability
of generating a positive pair. For triplet mode, each item contains anchor,
positive, and negative images.

The sampler is configured independently for `train` and `test` under
`common.batch_settings`:

```json
{
  "train": {
    "elements_per_batch": 256,
    "iterations": 10,
    "gen_imp_ratio": 0.3,
    "positive_mode": "random",
    "negative_mode": "hard",
    "auto_clusters": {
      "inner_imp_prob": 0.2,
      "raw_clusters": 1000,
      "cluster_max_size": 200,
      "make_clust_on_ep": 1
    }
  }
}
```

Supported negative modes:

- `random`: choose a negative class uniformly, then choose an image from it.
- `hard` (also `batch_hard`): from the negative examples sampled in the current
  batch, use the closest valid negative embedding for each anchor. In pair mode,
  replace each negative pair's second image with the closest cross-class
  candidate in the batch.
- `semi_hard`: for triplets, choose a negative with
  `d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin`,
  falling back to the closest valid negative if none is available. For
  contrastive pairs, choose the closest negative inside the contrastive margin,
  falling back to the closest valid candidate.
- `auto_clusters`: periodically find nearby class centroids and sample
  negatives from the anchor class's cluster with probability `inner_imp_prob`;
  otherwise use random negatives. This is class-level mining, not per-image
  hard mining. It requires `save_ideals: true`.

Cluster settings may be nested under `auto_clusters` as shown above or provided
as legacy flat fields. Pair sampling and triplet sampling both support random
and cluster-based negatives. The `hard` modes select from candidates in the
current batch, so their candidate pool grows with the training batch size.

For a contrastive experiment, use for example:

```json
"loss_settings": { "type": "contrastive", "alpha_margin": 1.0 },
"batch_settings": {
  "train": {
    "elements_per_batch": 256,
    "iterations": 10,
    "gen_imp_ratio": 0.5,
    "positive_mode": "random",
    "negative_mode": "random"
  },
  "test": {
    "elements_per_batch": 256,
    "iterations": 10,
    "gen_imp_ratio": 0.5,
    "positive_mode": "random",
    "negative_mode": "random"
  }
}
```

## Dataset layout

`KorSynthetic` expects one directory per class containing the repository's
synthetic JSON image records. `Omniglot` expects the usual alphabet/character
directory hierarchy. `Common` expects a two-level hierarchy, where each class
directory contains its images. The class folders must contain at least two
images to be eligible for positive-pair sampling.

## Metrics and reproducibility

Triplet evaluation treats anchor-positive as the positive pair and both
anchor-negative and positive-negative as negative pairs. EER uses negative
Euclidean distance as the pair score and same-class as the positive label. Set `common.seed` to seed Python,
NumPy, and PyTorch random generators. Checkpoints store model, optimizer,
scheduler, and epoch state; set `common.file_to_start` to a saved `model.pt` to
resume training.
