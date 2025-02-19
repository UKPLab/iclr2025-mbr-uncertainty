import dataclasses
import math

import numpy as np
# from evaluate import load_metric
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
from transformers import PretrainedConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
 DataCollatorForSeq2Seq, Trainer, DefaultDataCollator, Seq2SeqTrainer, DataCollatorForLanguageModeling, \
 AutoTokenizer, DataCollatorWithPadding, default_data_collator

from mbr.methods.base import Method
from mbr.methods.preprocessing.generation import CausalSeq2SeqPreprocessor, Seq2SeqPreprocessor
from mbr.trainer.trainer_seq2seq import CausalSeq2SeqTrainer, CausalSeq2SeqTrainerForIVON, Seq2SeqTrainerForIVON

class Seq2SeqMethod(Method):
    name = "seq2seq"
    peft_task_type = "SEQ_2_SEQ_LM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
            # load_metric(metric, cache_dir=self.model_args.cache_dir) for metric in ["sacrebleu"]
        ]

    def _ngram_stats(self, data, N):
        from nltk import ngrams
        """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
        ngram_freqs = {}  # ngrams with frequencies
        ngram_len = 0  # total number of ngrams
        for inst in data:
            for ngram in ngrams(inst, N):
                ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
                ngram_len += 1
        # number of unique ngrams
        uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
        return ngram_freqs, uniq_ngrams, ngram_len

    def _entropy(self, ngram_freqs):
        """Shannon entropy over ngram frequencies"""
        total_freq = sum(ngram_freqs.values())
        return -sum(
            [
                freq / total_freq * np.log2(freq / total_freq)
                for freq in ngram_freqs.values()
            ]
        )

    def compute_metrics(self, p):
        p.label_ids[p.label_ids == -100] = self.tokenizer.pad_token_id
        p.predictions[p.predictions == -100] = self.tokenizer.pad_token_id
        predictions_strings = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )
        reference_strings = [[ref] for ref in self.tokenizer.batch_decode(
            p.label_ids, skip_special_tokens=True)]

        results = {}

        # n-best list
        if len(predictions_strings) != len(reference_strings):
            return results

        for metric in self.metrics:
            results.update(
                metric.compute(
                    predictions=predictions_strings,
                    references=reference_strings
                )
            )

        return results


    def get_trainer_class(self):
        if "VON" in self.optimizer_args.optimizer_name:
            return Seq2SeqTrainerForIVON
        return Seq2SeqTrainer
        
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        if self.optimizer_args.load_peft_model:
            return AutoPeftModelForSeq2SeqLM
        else:
            return AutoModelForSeq2SeqLM

    def preprocess_features(self, features):
        processor = Seq2SeqPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def postprocess_predictions(self, p, dataset, scores=None):
        out = []
        p.label_ids[p.label_ids == -100] = self.tokenizer.pad_token_id
        reference_strings = self.tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
        model_class = self.get_model_class(self.config)

        p.predictions[p.predictions == -100] = self.tokenizer.pad_token_id
        decoded_predictions = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )

        decoded_inputs = self.tokenizer.batch_decode(
            [sample["input_ids"] for sample in dataset], skip_special_tokens=True
        )

        for idx, prediction in enumerate(decoded_predictions):
            input_idx = math.floor(idx / self.model_args.num_return_sequences)
            out.append({
                "sequence": prediction,
                "reference": reference_strings[input_idx],
                "source": decoded_inputs[input_idx]
            })

        if scores is not None:
            if len(scores) == len(out):
                for idx, score in enumerate(scores):
                    out[idx]["score"] = score

        return out

class CausalSeq2SeqMethod(Seq2SeqMethod):
    name="causal_seq2seq"
    peft_task_type = "CAUSAL_LM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
            # load_metric(metric, cache_dir=self.model_args.cache_dir) for metric in ["sacrebleu"]
        ]

        if not self.data_args.is_training:
            self.tokenizer.padding_side = "left"
    
    def preprocess_features(self, features):
        processor = CausalSeq2SeqPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def get_trainer_class(self):
        if "VON" in self.optimizer_args.optimizer_name:
            return CausalSeq2SeqTrainerForIVON
        else:
            return CausalSeq2SeqTrainer
        
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        if self.optimizer_args.load_peft_model:
            return AutoPeftModelForCausalLM
        else:
            return AutoModelForCausalLM

    def get_special_tokens(self):
        if self.tokenizer.chat_template is None:
            return ["<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"]
        else:
            return []

    def postprocess_predictions(self, p, dataset, scores=None):
        out = []
        p.label_ids[p.label_ids == -100] = self.tokenizer.pad_token_id
        reference_strings = self.tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
        model_class = self.get_model_class(self.config)

        p.predictions[p.predictions == -100] = self.tokenizer.pad_token_id
        decoded_predictions = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True
        )

        decoded_inputs = self.tokenizer.batch_decode(
            [sample["input_ids"] for sample in dataset], skip_special_tokens=True
        )

        print(decoded_inputs)

        for idx, prediction in enumerate(decoded_predictions):
            input_idx = math.floor(idx / self.model_args.num_return_sequences)
            out.append({
                "sequence": prediction[len(decoded_inputs[input_idx]):].strip(),
                "reference": reference_strings[input_idx],
                "source": decoded_inputs[input_idx]
            })

        if scores is not None:
            if len(scores) == len(out):
                for idx, score in enumerate(scores):
                    out[idx]["score"] = score

        return out
