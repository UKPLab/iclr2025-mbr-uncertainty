from typing import List

import datasets
from datasets import load_dataset
from .base import Seq2SeqDataset

class STSB(Seq2SeqDataset, datasets.GeneratorBasedBuilder):

    def _map_to_common_format(self, sample):
        return {
            "dataset_id": "stsb",
            "input": f"{sample['sentence1']} - {sample['sentence2']}",
            "output": sample["score"]
        }

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        dataset = load_dataset("sentence-transformers/stsb")

        splits = ["train", "validation", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        split_data = {split: [] for split in splits}

        for split in splits:
            for sample in dataset[split]:
                split_data[split].append(self._map_to_common_format(sample))

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": split_data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data):
        for idx, sample in enumerate(data):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
