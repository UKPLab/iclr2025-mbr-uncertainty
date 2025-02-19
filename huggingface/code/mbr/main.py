import dataclasses
import itertools
import json
import logging
import os
import subprocess
os.environ["TRANSFORMERS_CACHE"] = '/path/to/.cache/huggingface/transformers/'
os.environ["HF_HOME"] = '/path/to/.cache/huggingface'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
from enum import Enum

import numpy as np
import torch
import transformers
from peft import PeftConfig
from sacrebleu.metrics import BLEU
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    get_linear_schedule_with_warmup
)
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import is_main_process, PredictionOutput, get_last_checkpoint

from mbr.arguments import *
from mbr.methods.base import Method, TaskType
from mbr.methods.generation import CausalSeq2SeqMethod, Seq2SeqMethod
from mbr.optimizers.lr_schedulers import CosineAnnealingWarmupRestarts, \
 get_inverse_square_root_schedule_with_warmup
from mbr.optimizers.ivon import IVON
from utils import NumpyEncoder
import wandb
wandb.init(mode="disabled")

logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"
os.environ['WANDB_MODE'] = 'disabled'

method_classes = [
    CausalSeq2SeqMethod,
    Seq2SeqMethod,
]

optimizer_map = {
    "AdamW": torch.optim.AdamW,
    "IVON": IVON,
}


class GPUMemoryCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_step = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if torch.cuda.is_available():
            max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
            logging.info(f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
            state.log_history[-1]['gpu_memory'] = torch.cuda.max_memory_allocated()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in [1, 8, 64, 512]:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
                logging.info(
                    f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        super().on_step_end(args, state, control, **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prediction_step += 1
        if self.prediction_step in [1, 8, 64, 512]:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
                logging.info(
                    f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        super().on_prediction_step(args, state, control, **kwargs)


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


def get_config_class(model_args, optimizer_args):
    if optimizer_args.load_peft_model:
        return PeftConfig
    else:
        return AutoConfig

def get_lr_scheduler(optimizer, optimizer_args, training_args, dataset, data_args):
    max_steps = (len(dataset) * training_args.num_train_epochs) // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
    if data_args.is_training:
        if data_args.lr_scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                max_steps,
            )
        elif data_args.lr_scheduler == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                max_steps
            )
        else:
            raise NotImplementedError
    else:
        return None

def get_optimizer(model, optimizer_args, training_args, training_data):
    optimizer_class = optimizer_map[optimizer_args.optimizer_name]

    if optimizer_class == torch.optim.AdamW:
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=training_args.learning_rate,
            betas=(optimizer_args.beta1, optimizer_args.beta2),
            eps=optimizer_args.eps,
            weight_decay=training_args.weight_decay
        )
    elif optimizer_class == IVON:
        optimizer = IVON(
            [p for n, p in model.named_parameters() if p.requires_grad], 
            training_args.learning_rate,  
            mc_samples=optimizer_args.mc_samples, 
            beta1=optimizer_args.beta1,
            beta2=optimizer_args.beta2, 
            weight_decay=training_args.weight_decay,
            hess_init=optimizer_args.hess_init,
            clip_radius=optimizer_args.clip_radius,
            ess=optimizer_args.ess,
            hess_approx=optimizer_args.hessian_estimator,
            prior_mean=0.0 if not optimizer_args.use_prior_mean else [p for n, p in model.named_parameters()],
            rescale_lr=optimizer_args.rescale_lr
            )
    else:
        raise NotImplementedError()

    return optimizer

def get_tokenizer_class(config, model_args):
        return AutoTokenizer

def get_tokenizer_name(config, model_args):
    if model_args.tokenizer_name:
        return model_args.tokenizer_name
    else:
        return model_args.model_name_or_path


class RunMode(Enum):
    TRAIN = 1
    PREDICT = 2


def main(run_mode: RunMode):
    training_args_class = Seq2SeqTrainingArguments
    parser_arguments = (ModelArguments, DataTrainingArguments if run_mode ==
                                                                 RunMode.TRAIN else DataPredictionArguments,
                        OptimizerArguments,
                        training_args_class)
    parser = HfArgumentParser(parser_arguments)

    raw_args = sys.argv[1:]
    json_index = -1 if raw_args[-1].endswith(".json") and (len(
        raw_args) == 1 or not raw_args[-2].startswith('-') or '=' in raw_args[-2]) else 0
    if len(raw_args) > 0 and raw_args[json_index].endswith(".json"):
        with open(raw_args[json_index]) as fp:
            json_args_dict = json.load(fp)
        del raw_args[json_index]

        if run_mode == RunMode.TRAIN:
            train_parser = HfArgumentParser(training_args_class)
            training_args_dict = vars(train_parser.parse_args(
                raw_args + ['--output_dir', json_args_dict['output_dir']]))
            training_args_dict.update(json_args_dict)
            json_args_dict = training_args_dict

        model_args, data_args, optimizer_args, training_args = parser.parse_dict(
            json_args_dict, allow_extra_keys=True)
    else:
        model_args, data_args, optimizer_args, training_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(model_args.manual_seed)

    logging.info(data_args)

    logging.info(
        f"My rank is {training_args.local_rank} with {torch.cuda.device_count()} GPUs.")
    if training_args.local_rank != -1:
        torch.cuda.set_device(training_args.local_rank)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    _setup_logging(training_args)

    config_class = get_config_class(model_args, optimizer_args)

    config = config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    if model_args.num_labels is not None:
        config.num_labels = model_args.num_labels

    print(model_args.use_auth_token)

    tokenizer = get_tokenizer_class(config, model_args).from_pretrained(
        get_tokenizer_name(config, model_args),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method_class = next(
        (m for m in method_classes if m.name == model_args.method), None)
    if method_class is None:
        raise Exception(f"No method class for name {model_args.method}.")
    method_definition: Method = method_class(
        model_args, data_args, optimizer_args, config, tokenizer)

    # Set seed before initializing model.
    set_seed(model_args.manual_seed)
    torch.manual_seed(model_args.manual_seed)

    model = method_definition.get_model(run_mode, config).to(training_args.device)
    model.config.keys_to_ignore_at_inference = [
        "decoder_attentions"
    ]
    model.config.num_beams = model_args.generation_beam_size
    model.config.max_length = model_args.generation_max_len
    model.config.do_sample = model_args.generation_do_sample
    model.config.top_k = model_args.generation_top_k
    model.config.length_penalty = model_args.generation_length_penalty
    model.config.no_repeat_ngram_size = model_args.generation_no_repeat_ngram_size
    model.config.num_return_sequences = model_args.num_return_sequences
    
    model.config.dropout = model_args.dropout

    if run_mode == RunMode.TRAIN:
        extra_trainer_args = {
            'train_dataset': method_definition.get_train_dataset(),
            'eval_dataset': method_definition.get_validation_dataset(),
        }
    else:
        extra_trainer_args = {
            'eval_dataset': method_definition.get_test_dataset()
        }

    data_collator = method_definition.get_data_collator()
    trainer_class = method_definition.get_trainer_class()

    # if run_mode == RunMode.TRAIN:
    if run_mode == RunMode.TRAIN:
        optimizer = get_optimizer(model, optimizer_args, training_args, extra_trainer_args["train_dataset"])
        lr_scheduler = get_lr_scheduler(optimizer, optimizer_args, training_args, extra_trainer_args["train_dataset"], data_args)
    else:
        optimizer = get_optimizer(model, optimizer_args, training_args, extra_trainer_args["eval_dataset"])
        lr_scheduler = get_lr_scheduler(optimizer, optimizer_args, training_args, extra_trainer_args["eval_dataset"], data_args)  

    if data_args.ivon_s is not None:
        import copy
        model_class = method_definition.get_model_class(config)
        ivon_s = model_class.from_pretrained(data_args.ivon_s, is_trainable=True, use_auth_token=model_args.use_auth_token)
        full_hess = []
        for n, p in ivon_s.named_parameters():
            if p.requires_grad:
                full_hess.append(p.flatten())
        optimizer.param_groups[0]["hess"].data.copy_(torch.cat(full_hess))

    if model_args.sample_params:
        optimizer._sample_params()

    trainer: Trainer = trainer_class(
        model=model,
        args=training_args,
        tokenizer=method_definition.tokenizer,
        data_collator=data_collator,
        compute_metrics=method_definition.compute_metrics,
        optimizers=(optimizer, lr_scheduler),
        **extra_trainer_args,
    )

    trainer.mc_samples = optimizer_args.inference_mc_samples

    trainer.add_callback(GPUMemoryCallback())

    trainer.scores = []

    if run_mode == RunMode.TRAIN:
        # Check for existing checkpoint to continue the training
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        resume_from_checkpoint = last_checkpoint if last_checkpoint is not None else None
        # Start training
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint)

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))

        test_dataset = method_definition.get_test_dataset()
        results = trainer.predict(test_dataset)
        metrics = method_definition.compute_metrics(results)

        if trainer.args.save_strategy != "no":
            if isinstance(trainer.optimizer.optimizer, IVON):
                print("Saving IVON Hessian.")
                output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
                trainer.save_model(output_dir)
                lower_bound = 0
                hessian = trainer.optimizer.param_groups[0]["hess"]
                for n, p in trainer.model.named_parameters():
                    if p.requires_grad:
                        length = p.flatten().shape[0]
                        local_hessian = hessian[lower_bound:lower_bound+length]
                        local_hessian = local_hessian.reshape(p.shape)
                        p.data.copy_(local_hessian)
                        lower_bound += length
                output_dir = os.path.join(training_args.output_dir, "hessian")
                trainer.save_model(output_dir)

    elif run_mode == RunMode.PREDICT:
        test_dataset = method_definition.get_test_dataset()

        results = trainer.predict(test_dataset)
        results = method_definition.postprocess_predictions(
            results,
            test_dataset,
            scores=trainer.scores
        )

        if data_args.prediction_output_file is not None:
            with open(data_args.prediction_output_file, 'wt') as f:
                try:
                    json.dump(
                        dataclasses.asdict(results) if type(
                            results) == PredictionOutput else results,
                        f,
                        cls=NumpyEncoder,
                        ensure_ascii=False
                    )
                except:
                    json.dump({}, f)
