"""Tests for dataset indexing utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.dataset_index import build_index, load_index, save_index, split_index, summarize_index


def _touch(path: Path) -> None:
    """Create an empty file and its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class DatasetIndexTestCase(unittest.TestCase):
    """Verify the first data indexing step stays understandable and stable."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_root = self.root / "data" / "raw"
        self.category = "demo_category"

        _touch(self.dataset_root / self.category / "train" / "good" / "rgb" / "000.png")
        _touch(self.dataset_root / self.category / "train" / "good" / "xyz" / "000.tiff")

        _touch(self.dataset_root / self.category / "validation" / "good" / "rgb" / "001.png")
        _touch(self.dataset_root / self.category / "validation" / "good" / "xyz" / "001.tiff")

        _touch(self.dataset_root / self.category / "test" / "good" / "rgb" / "002.png")
        _touch(self.dataset_root / self.category / "test" / "good" / "xyz" / "002.tiff")
        _touch(self.dataset_root / self.category / "test" / "good" / "gt" / "002.png")

        _touch(self.dataset_root / self.category / "test" / "hole" / "rgb" / "003.png")
        _touch(self.dataset_root / self.category / "test" / "hole" / "xyz" / "003.tiff")
        _touch(self.dataset_root / self.category / "test" / "hole" / "gt" / "003.png")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_index_and_splits(self) -> None:
        """The index should reflect all sample types and normalized split names."""
        index_df = build_index(self.dataset_root, [self.category])

        self.assertEqual(len(index_df), 4)
        self.assertEqual(index_df["split"].tolist(), ["train", "val", "test", "test"])
        self.assertEqual(index_df["label"].tolist(), [0, 0, 0, 1])
        self.assertEqual(index_df["sample_id"].tolist(), ["000", "001", "002", "003"])
        self.assertEqual(index_df.loc[2, "mask_path"], "data/raw/demo_category/test/good/gt/002.png")
        self.assertEqual(index_df.loc[0, "mask_path"], "")

        split_tables = split_index(index_df)
        self.assertEqual(len(split_tables["train"]), 1)
        self.assertEqual(len(split_tables["val"]), 1)
        self.assertEqual(len(split_tables["test"]), 2)

    def test_save_load_and_summary(self) -> None:
        """Saved CSV files should reload cleanly and keep empty mask paths."""
        index_df = build_index(self.dataset_root, [self.category])
        output_path = self.root / "data" / "splits" / "index.csv"
        save_index(index_df, output_path)

        reloaded_df = load_index(output_path)
        train_row = reloaded_df.loc[reloaded_df["split"] == "train"].iloc[0]
        self.assertEqual(train_row["mask_path"], "")

        summary_df = summarize_index(index_df)
        counts = {
            (row.category, row.split, int(row.label)): int(row.count)
            for row in summary_df.itertuples(index=False)
        }
        self.assertEqual(counts[(self.category, "train", 0)], 1)
        self.assertEqual(counts[(self.category, "val", 0)], 1)
        self.assertEqual(counts[(self.category, "test", 0)], 1)
        self.assertEqual(counts[(self.category, "test", 1)], 1)


if __name__ == "__main__":
    unittest.main()
