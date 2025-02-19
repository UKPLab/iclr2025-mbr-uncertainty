import sys

import numpy as np

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

import sisyphus.toolkit as tk
from ukp.fairseq.search import MBRJob, NeuralMBRJob
from ukp.huggingface.search import *
from ukp.huggingface.training import *
from ukp.huggingface.evaluation import *
from ukp.fairseq.evaluation import FairseqEvalJob


Path = tk.Path

code_root = "/path/to/iclr2025-mbr-uncertainty/huggingface/code/"

def TODO():
    print("Parameter undefined")

def eval_dataset(config, model, dataset, dataset_config_name, task_config, nbest_size, 
                 do_sample=True, f_string_prefix="", rerun=False, gpumem=24, per_device_eval_batch_size=2, time_rqmt=4):
    config["per_device_eval_batch_size"] = per_device_eval_batch_size
    config["use_peft"] = False
    config["load_peft_model"] = True
    config["generation_beam_size"] = nbest_size
    config["num_return_sequences"] = nbest_size
    config["generation_do_sample"] = do_sample

    search_data_config = {
        'dataset_name': os.path.join(code_root, f'mbr/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': task_config['test_split'],
    }

    if not rerun:
        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=32,
            time_rqmt=time_rqmt,
            gpumem=gpumem
        )
    else:
        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=60,
            time_rqmt=24,
            gpumem=gpumem,
            rerun=True
        )
    sampling_str = "ancestral_sampling" if do_sample else "beam_search"
    tk.register_output(f"{f_string_prefix}_{nbest_size}_{sampling_str}.json", search_job.out_search_file)
    predictions = PredictionDictToListJob(code_root, search_job.out_search_file, mem_rqmt=12).out_search_file

    return predictions

def calculate_metrics(predictions, dataset, dataset_config_name, task_config, task, nbest_size,
                     do_sample=True, f_string_prefix=""):
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'mbr/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': task_config['test_split'],
    }

    eval_job = CalculateMetricsJob(
        code_root,
        task,
        predictions,
        search_data_config,
        nbest=nbest_size,
        gpumem=24
    )
    sampling_str = "ancestral_sampling" if do_sample else "beam_search"
    tk.register_output(f"{f_string_prefix}_{nbest_size}_{sampling_str}.metrics.json", eval_job.out_metric_file)

def neural_mbr(predictions, dataset, dataset_config_name, task_config, task, nbest_size, refs, srcs,
               do_sample=True, f_string_prefix="", mem_rqmt=32, rerun=False, flatten=False, rerun_eval=False):
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'mbr/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': task_config['test_split'],
    }

    mbr_job = NeuralMBRJob(
        predictions,
        refs,
        nbest_size,
        src=srcs,
        mem_rqmt=mem_rqmt,
        flatten=flatten
    )

    eval_job = CalculateMetricsJob(
        code_root,
        task,
        mbr_job.out_hyp_file,
        search_data_config,
        gpumem=24
    )

    sampling_str = "ancestral_sampling" if do_sample else "beam_search"
    tk.register_output(f"{f_string_prefix}_{nbest_size}_{sampling_str}_rerun.mbr.metrics.json", eval_job.out_metric_file)

    return mbr_job.out_hyp_file

              

async def ivon_iwslt_gemma():
    srcs = "/path/to/iwslt17_srcs.json" # Path to a file containing a list of srcs
    refs = "/path/to/iwslt17_refs.json" #  Path to a file containing a list of references

    datset = "iwslt17"
        
    task_config = {
            "method": "causal_seq2seq",
            "test_split": "test",
            "validation_split": "test[:2%]",
            "num_epochs": 1,
            "train_split": "train"
        }
    
    models = {}

    all_predictions = {
        "eq10": [],
        "eq9": []
    }
    
    for seed in [1, 42, 64, 128]:

        use_peft = True

        dataset_config_name = "seq2seq"

        config = {
            "model_name_or_path": "google/gemma-2b-it",
            "predict_with_generate": True,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "cache_dir": TODO(),
            "max_input_length": 1024,
            "max_output_length": 128,
            "optimizer_name": "IVON",
            "weight_decay": 1e-6,
            "warmup_steps": 512,
            "min_lr": 0.0,
            "beta1": 0.9,
            "beta2": 0.99999,
            "hess_init": 3e-4,
            "ess": 1e7,
            "max_grad_norm": 1.0,
            "learning_rate": 3e-2,
            "clip_radius": 1e-3,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "use_peft": use_peft,
            "manual_seed": seed,
            "use_auth_token": TODO(),
            "prompt_prefix": "Translate from English to German: "
        }

        train_data_config = {
            'dataset_name': os.path.join(code_root, f'mbr/datasets/{dataset}.py'),
            'dataset_config_name': dataset_config_name,
            'dataset_train_split': task_config['train_split'],
            'dataset_val_split': task_config['validation_split'],
            'dataset_test_split': task_config['validation_split']
        }

        config["method"] = task_config["method"]

        train_job = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=task_config["num_epochs"],
            mem_rqmt=40,
            time_rqmt=24,
            gpumem=32
        )
        train_job.add_alias(f"{dataset}_ivon_{seed}")
        tk.register_output(f"{dataset}_ivon_{seed}", train_job.out_best_model)


        for nbest_size_ in [10, 20]:

            do_sample = True

            config["generation_top_k"] = 0

            predictions = eval_dataset(config, models[seed], dataset, dataset_config_name, task_config, 
                                    nbest_size_, f_string_prefix=f"iclr25/huggingface/ivon_iwslt17_{seed}", do_sample=do_sample,
                                    gpumem=24, per_device_eval_batch_size=1 if nbest_size_ > 20 else 2)
            if nbest_size_ == 20:
                all_predictions["eq10"].append(predictions)
            elif nbest_size_ == 10:
                all_predictions["eq9"].append(predictions)

    neural_mbr(all_predictions["eq9"], dataset, dataset_config_name, task_config, "mt", 10, refs, srcs,
                f_string_prefix=f"iclr25/huggingface/multimodal_ivon_iwslt17_eq9", do_sample=True, flatten=True, mem_rqmt=80)

    neural_mbr(all_predictions["eq10"], dataset, dataset_config_name, task_config, "mt", 20, refs, srcs,
                f_string_prefix=f"iclr25/huggingface/multimodal_ivon_iwslt17_eq10", do_sample=True, mem_rqmt=80)

async def py():
    await async_main()

async def async_main():
    await ivon_iwslt_gemma()

async def py():
    await async_main()
