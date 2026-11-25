"""Checks for global triplet generation and mining rules."""

import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiment_sampling import ClassGraph, TripletIndexBatchSampler, generate_triplet_batch, mine_global_negatives
from experiment_losses import CATMLLoss
from experiment_sampling import APMState


class GlobalMiningTests(unittest.TestCase):
    def setUp(self):
        self.bank = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.0],
            [1.5, 0.0],
            [5.0, 5.0],
        ])
        self.labels = torch.tensor([0, 0, 1, 2, 3])
        self.anchor = self.bank[[0]]
        self.positive = self.bank[[1]]
        self.anchor_label = self.labels[[0]]

    def test_global_hard_picks_nearest_different_class(self):
        selected = mine_global_negatives(
            self.anchor, self.positive, self.anchor_label, self.bank, self.labels,
            mode="global_hard", query_chunk_size=1, candidate_chunk_size=2,
        )
        self.assertEqual(selected.tolist(), [2])

    def test_global_semi_hard_picks_nearest_inside_margin(self):
        selected = mine_global_negatives(
            self.anchor, self.positive, self.anchor_label, self.bank, self.labels,
            mode="semi_hard", margin=2.0, query_chunk_size=1, candidate_chunk_size=2,
        )
        self.assertEqual(selected.tolist(), [3])

    def test_global_semi_hard_falls_back_to_global_hard(self):
        selected = mine_global_negatives(
            self.anchor, self.positive, self.anchor_label, self.bank, self.labels,
            mode="semi_hard", margin=0.1, query_chunk_size=1, candidate_chunk_size=2,
        )
        self.assertEqual(selected.tolist(), [2])

    def test_triplet_batch_keeps_positive_class_and_cluster_negative(self):
        refs_by_class = [[0, 1], [2, 3], [4, 5]]
        graph = ClassGraph(3, max_cluster_size=0)
        graph.neighbors[10].add(20)
        graph.neighbors[20].add(10)
        args = (
            64, refs_by_class, [0, 2, 4, 6], [{}, {}, {}], [0, 1, 2], [0, 1, 2],
            [10, 20, 30], graph, None,
        )
        random_mode = generate_triplet_batch(
            *args, negative_mode="random", directed_probability=1.0, margin=0.2,
            rng=random.Random(7), negative_rng=random.Random(8), np_rng=np.random.default_rng(9),
            device=torch.device("cpu"),
        )
        cluster_mode = generate_triplet_batch(
            *args, negative_mode="auto_clusters", directed_probability=1.0, margin=0.2,
            rng=random.Random(7), negative_rng=random.Random(8), np_rng=np.random.default_rng(9),
            device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(random_mode[:, :2], cluster_mode[:, :2])
        for anchor_row, positive_row, negative_row in cluster_mode:
            anchor_class = int(anchor_row // 2)
            positive_class = int(positive_row // 2)
            negative_class = int(negative_row // 2)
            self.assertEqual(anchor_class, positive_class)
            self.assertNotEqual(anchor_class, negative_class)
            if anchor_class in (0, 1):
                self.assertEqual(negative_class, 1 - anchor_class)

    def test_index_sampler_splits_logical_batches_into_microbatches(self):
        logical_batches = 0

        def generate():
            nonlocal logical_batches
            start = logical_batches * 8
            logical_batches += 1
            return np.arange(start, start + 24, dtype=np.int64).reshape(8, 3)

        sampler = TripletIndexBatchSampler(batch_count=3, batch_size=8, minibatch_size=4, generate_batch=generate)
        microbatches = list(sampler)
        self.assertEqual(len(microbatches), 6)
        self.assertTrue(all(len(microbatch) == 4 for microbatch in microbatches))
        self.assertEqual(logical_batches, 3)
        self.assertEqual(microbatches[0][0], (0, 1, 2))
        self.assertEqual(microbatches[2][0], (8, 9, 10))

    def test_index_sampler_works_as_dataloader_batch_sampler(self):
        class IndexDataset(Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, index):
                return index

        sampler = TripletIndexBatchSampler(
            batch_count=2,
            batch_size=4,
            minibatch_size=2,
            generate_batch=lambda: np.arange(12, dtype=np.int64).reshape(4, 3),
        )
        loader = DataLoader(IndexDataset(), batch_sampler=sampler, collate_fn=lambda items: items)
        batches = list(loader)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [(0, 1, 2), (3, 4, 5)])

    def test_apm_measures_radii_to_the_shared_centers(self):
        apm = APMState(["a", "b"], ema=0.5)
        centers = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        apm.update_shared_centers(
            torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([0, 0, 1]),
            centers,
            torch.tensor([True, True]),
        )
        np.testing.assert_allclose(apm.radii, [1.0, 1.0])
        self.assertTrue(apm.observed.all())

    def test_catml_uses_explicit_shared_centers(self):
        embeddings = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        labels = torch.tensor([0, 0, 1])
        centers = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 2.0])}
        objective = CATMLLoss(center_mode="current", rho=0.0, tau=0.0, xi=1.0)
        _, metrics = objective(
            embeddings, labels, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]),
            fixed_centers=centers,
        )
        expected = torch.tensor([1.0, 1.0, 1.0]).mean()
        torch.testing.assert_close(metrics["cluster"], expected)


if __name__ == "__main__":
    unittest.main()
