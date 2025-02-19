import copy
import itertools

from transformers import GemmaTokenizerFast

from mbr.methods.preprocessing.base import Preprocessor

class Seq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features):
        sequences, labels = [], []
        for source, target in zip(features["input"], features["output"]):
            if source is not None and target is not None:
                tokenized_source = self.tokenizer(
                    self.model_args.prompt_prefix + source, 
                    max_length=self.model_args.max_input_length,
                    truncation=True
                )["input_ids"]
                tokenized_target = self.tokenizer(
                    target, 
                    max_length=self.model_args.max_output_length,
                    truncation=True
                )["input_ids"]
                sequences.append(tokenized_source)
                labels.append(tokenized_target)

        return sequences, labels


class CausalSeq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features):
        sequences, labels = [], []
        for source, target in zip(features["input"], features["output"]):
            if source is not None and target is not None:
                if "google/gemma-2b-it" in self.model_args.model_name_or_path or isinstance(self.tokenizer, GemmaTokenizerFast):
                    # hack to truncate to max_input_length..
                    source = self.tokenizer.batch_decode(
                        [self.tokenizer(
                            source,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.model_args.max_input_length
                        )["input_ids"]],
                        skip_special_tokens=True
                    )[0]
                    input_no_label = self.tokenizer.apply_chat_template([
                        {"role": "user",
                        "content": self.model_args.prompt_prefix + source},
                        ], add_generation_prompt=True)
                    input_label = self.tokenizer.apply_chat_template([
                        {"role": "user",
                        "content": self.model_args.prompt_prefix + source},
                        {"role": "model",
                        "content": target}
                    ])
                    if self.data_args.is_training:
                        input_ = input_label + [self.tokenizer.eos_token_id]
                        tokenized_target = input_.copy()
                    else:
                        input_ = input_no_label
                        tokenized_target = self.tokenizer(
                            target, 
                            max_length=self.model_args.max_input_length,
                            truncation=True,
                            add_special_tokens=True
                        )["input_ids"]
                    tokenized_source = input_
                else:
                    raise NotImplementedError

            sequences.append(tokenized_source)
            labels.append(tokenized_target)

        return sequences, labels
