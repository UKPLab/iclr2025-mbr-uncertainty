from typing import List

import datasets
from datasets import load_dataset
from tqdm import tqdm

from .base import Seq2SeqDataset

class IWSLT17(Seq2SeqDataset, datasets.GeneratorBasedBuilder):

    def _map_to_common_format(self, sample):
        return {
            "dataset_id": "iwslt2017",
            "input": sample["translation"]["en"],
            "output": sample["translation"]["de"]
        }

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        splits = ["train", "validation", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        data = {split: [] for split in splits}

        dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
        for split in splits:
            local_dataset = dataset[split]
            for sample in local_dataset:
                data[split].append(self._map_to_common_format(sample))

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data):
        for idx, sample in tqdm(enumerate(data)):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
