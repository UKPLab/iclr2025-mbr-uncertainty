from typing import List

import datasets
from datasets import load_dataset
from .base import Seq2SeqDataset

class XSUM(Seq2SeqDataset, datasets.GeneratorBasedBuilder):

    def _map_to_common_format(self, sample):
        return {
            "dataset_id": "xsum",
            "input": sample["document"],
            "output": sample["summary"]
        }

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        splits = ["train", "validation", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        data = {split: [] for split in splits}

        for split in splits:
            dataset = load_dataset("xsum", split=split)
            for sample in dataset:
                data[split].append(self._map_to_common_format(sample))

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data):
        for idx, sample in enumerate(data):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
