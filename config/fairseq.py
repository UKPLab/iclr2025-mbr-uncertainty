import os
import sys

import numpy as np

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

import sisyphus.toolkit as tk
from ukp.fairseq.evaluation import FairseqEvalJob
from ukp.fairseq.search import FairseqSearchJob, MBRJob
from ukp.fairseq.training import FairseqTrainingJob

Path = tk.Path

async def ivon_iwslt14_base():
    for mc_samples in [2]:
        for optimizer in ["ivon"]:
            config = [
                "/path/to/data-bin/iwslt14.tokenized.de-en", # binarized data for iwslt14
                "--arch", "transformer", 
                "--share-decoder-input-output-embed",
                "--optimizer", optimizer,
                "--clip-norm", "1.0",
                "--lr", "0.15", 
                "--lr-scheduler", "inverse_sqrt",
                "--warmup-updates", "4000",
                "--clip-radius", "0.001",
                "--dropout", "0.2",
                "--weight-decay", "0.0001",
                "--criterion", "cross_entropy",
                "--max-tokens", "4096",
                "--eval-bleu",
                "--eval-bleu-args", '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}',
                "--eval-bleu-detok", "moses",
                "--eval-bleu-remove-bpe",
                "--best-checkpoint-metric", "bleu",
                "--maximize-best-checkpoint-metric",
                "--patience", "3",
                "--batch-size", "1024",
                "--ess", "1e8",
                "--hess-init", "0.1",
                "--seed", "1",
                "--mc-samples", str(mc_samples),
            ]

            train_job = FairseqTrainingJob(
                config
            )
            train_job.add_alias(f"{optimizer}_comparison_mc_{mc_samples}")
            tk.register_output(f"fairseq_example/{optimizer}_ivon_iwslt14_trafo_base_{mc_samples}", train_job.out_checkpoints_dir)

            model_path = os.path.join(train_job.out_checkpoints_dir.get_path(), "checkpoint_best.pt")

            config = [
                "/path/to/data-bin/iwslt14.tokenized.de-en", # binarized data for iwslt14
                "--path", model_path,
                "--batch-size", "128",
                "--beam", "4",
                "--nbest", "4",
                "--remove-bpe",
                "--sampling",
                "--lenpen", "0.6",
                "--sample-params",
                "--num-mc-samples", "1"
            ]

            search_job = FairseqSearchJob(config, train_job.out_checkpoints_dir)
            tk.register_output(f"example/{optimizer}_iwslt14_trafo_base_out.txt", search_job.out_hyp_file)

            config = [
                "/path/to/data-bin/iwslt14.tokenized.de-en", # binarized data for iwslt14
                "--path", model_path,
                "--batch-size", "128",
                "--beam", "4",
                "--nbest", "4",
                "--remove-bpe",
                "--sampling",
                "--lenpen", "0.6",
                "--sample-params",
                "--num-mc-samples", "1"
            ]

            search_job = FairseqSearchJob(config, train_job.out_checkpoints_dir)
            tk.register_output(f"example/{optimizer}_iwslt14_trafo_base_out.txt", search_job.out_hyp_file)

            eval_job = FairseqEvalJob(
                search_job.out_hyp_file,
                search_job.out_tgt_file,
                nbest=4
            )
            tk.register_output(f"example/{optimizer}ivon_iwslt14_trafo_base_out.metrics.json", eval_job.out_metrics_file)

            mbr_job = MBRJob(
                search_job.out_hyp_file,
                search_job.out_src_file,
                4
            )
            tk.register_output(f"example/{optimizer}ivon_iwslt14_trafo_base_out.mbr.txt", mbr_job.out_hyp_file)

            eval_job = FairseqEvalJob(
                mbr_job.out_hyp_file,
                search_job.out_tgt_file,
                nbest=1
            )
            tk.register_output(f"example/{optimizer}ivon_iwslt14_trafo_base_out.mbr.metrics.json", eval_job.out_metrics_file)

async def async_main():
    await ivon_iwslt14_base()

async def py():
    await async_main()
