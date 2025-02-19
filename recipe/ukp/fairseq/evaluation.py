import copy
import os
import shutil
import subprocess as sp
import copy
import json
import logging

import sacrebleu
# from comet import download_model, load_from_checkpoint
from sisyphus import *
from tqdm import tqdm

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)

class FairseqEvalJob(Job):
    """
    Evaluate a fairseq transformer model
    """
    __sis_hash_exclude__ = {
        'keep_only_best': False,
        'distributed': False,
        'sbatch_args': None,
        'selective_prediction_metrics': None,
        'risk_file': None,
        'raw_file': None,

    }

    def __init__(
        self,
        hyps,
        tgts,
        srcs=None,
        nbest=1,
        metrics=["bleu"],
        time_rqmt=4,
        mem_rqmt=32,
        cpu_rqmt=1,
        gpu_rqmt=1,
        gpumem=10,
        sbatch_args=None,
        keep_only_best=False,
        distributed=False,
        selective_prediction_metrics=None,
        raw_file=None,
        risk_file=None,
        **kwargs,
    ):
        """
        :param config:
        :param num_epochs:
        :param time_rqmt:
        :param mem_rqmt:
        :param cpu_rqmt:
        :param gpu_rqmt:
        """
        self.hyps = hyps
        self.metrics = metrics
        self.nbest = nbest
        self.tgts = tgts
        self.srcs = srcs

        self.keep_only_best = keep_only_best
        self.distributed = distributed

        self.risk_file = risk_file
        self.selective_prediction_metrics = selective_prediction_metrics
        self.raw_file = raw_file

        if gpu_rqmt > 1:
            sbatch_args = "-P multigpu"
        elif sbatch_args is None:
            sbatch_args = []

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        #     "gpumem": gpumem,
            "sbatch_args": sbatch_args,
        }
        if gpu_rqmt > 0:
            self.rqmt["gpumem"] = gpumem

        self.out_metrics_file = self.output_path("metrics.json", directory=False)

        self.python_exe = gs.PYTHON_EXE

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.python_exe),
            "/path/to/iclr2025-mbr-uncertainty/evaluation.py",
            "--hyps", self.hyps.get_path(),
            "--tgts", self.tgts if isinstance(self.tgts, str) else self.tgts.get_path(),
            "--outfile", self.out_metrics_file.get_path(),
            "--nbest", str(self.nbest),
            "--metrics", ";".join(self.metrics),
        ]
        if self.srcs is not None:
            run_cmd.extend(["--srcs", self.srcs if isinstance(self.srcs, str) else self.srcs.get_path()]),
        if self.risk_file is not None:
            run_cmd.extend([
                "--risk-file", self.risk_file.get_path()
            ])
        if self.selective_prediction_metrics is not None:
            run_cmd.extend([
                "--selective-prediction-metrics", ";".join(self.selective_prediction_metrics)
            ])
        if self.raw_file is not None:
            run_cmd.extend([
                "--raw-file", self.raw_file.get_path()
            ])
        return run_cmd

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def run(self):
        sp.check_call(self._get_run_cmd())

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
            hash_kwargs = copy.deepcopy(kwargs)
            excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
            for key in excluded_keys:
                if key in hash_kwargs:
                    del hash_kwargs[key]

            if 'kwargs' in hash_kwargs and (hash_kwargs['kwargs'] is None or len(hash_kwargs['kwargs']) == 0):
                del hash_kwargs['kwargs']

            return super().hash(hash_kwargs)

class EvaluateDiversityJob(Job):
    """
    Evaluate a fairseq transformer model
    """
    __sis_hash_exclude__ = {
        'keep_only_best': False,
        'distributed': False,
        'sbatch_args': None,
    }

    def __init__(
        self,
        hyps,
        metrics=["bleu"],
        time_rqmt=4,
        mem_rqmt=32,
        cpu_rqmt=1,
        gpu_rqmt=0,
        gpumem=12,
        sbatch_args=None,
        keep_only_best=False,
        distributed=False,
        **kwargs,
    ):
        """
        :param config:
        :param num_epochs:
        :param time_rqmt:
        :param mem_rqmt:
        :param cpu_rqmt:
        :param gpu_rqmt:
        """
        self.hyps = hyps
        self.metrics = metrics

        self.keep_only_best = keep_only_best
        self.distributed = distributed

        if gpu_rqmt > 1:
            sbatch_args = "-P multigpu"
        elif sbatch_args is None:
            sbatch_args = []

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        #     "gpumem": gpumem,
            "sbatch_args": sbatch_args,
        }
        if gpu_rqmt > 0:
            self.rqmt["gpumem"] = gpumem

        self.out_metrics_file = self.output_path("metrics.json", directory=False)

        self.python_exe = gs.PYTHON_EXE


    def run(self):
        import itertools
        import numpy as np
        import sacrebleu

        all_hyps = []
        bleus = []
        metrics = {}

        for hyp_file in self.hyps:
            with open(hyp_file.get_path(), "r") as f:
                local_hyps = json.load(f)
            local_hyps = [hyp.strip() for hyp in local_hyps]
            all_hyps.append(local_hyps)

        for combination in itertools.combinations(all_hyps, 2):
            combination_bleu = sacrebleu.corpus_bleu(combination[0], [combination[1]])
            bleus.append(combination_bleu.score)
        metrics["bleu_diversity"] = np.mean(bleus)

        with open(self.out_metrics_file.get_path(), "w") as f:
            json.dump(metrics, f)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
            hash_kwargs = copy.deepcopy(kwargs)
            excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
            for key in excluded_keys:
                if key in hash_kwargs:
                    del hash_kwargs[key]

            if 'kwargs' in hash_kwargs and (hash_kwargs['kwargs'] is None or len(hash_kwargs['kwargs']) == 0):
                del hash_kwargs['kwargs']

            return super().hash(hash_kwargs)