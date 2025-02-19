import copy
import os
import shutil
import subprocess as sp
import copy
import json
import logging

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)

class FairseqSearchJob(Job):
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
      config,
      model,
      resume_from_checkpoint=None,
      time_rqmt=4,
      mem_rqmt=32,
      cpu_rqmt=1,
      gpu_rqmt=1,
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
    self.config = config
    self.model = model

    self.keep_only_best = keep_only_best
    self.distributed = distributed

    self.resume_from_checkpoint = resume_from_checkpoint

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "sbatch_args": sbatch_args,
    }
    
    if gpu_rqmt > 0:
        self.rqmt["gpumem"] = gpumem

    self.out_hyp_file = self.output_path("hyps.txt", directory=False)
    self.out_src_file = self.output_path("src.txt", directory=False)
    self.out_tgt_file = self.output_path("tgt.txt", directory=False)
    self.out_raw_file = self.output_path("raw.txt", directory=False)

  def _get_run_cmd(self):
      run_cmd = [
          "fairseq-generate",
      ]
      if self.resume_from_checkpoint is not None:
          pass # TODO: implement
      run_cmd.extend(self.config)
      return run_cmd

  def create_files(self):
    
    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    with open(self.out_raw_file.get_path(), "w") as f:
      sp.check_call(self._get_run_cmd(), stdout=f)

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)
    yield Task("cleanup", mini_task=True)

  def cleanup(self):
    import json
    with open(self.out_raw_file.get_path(), "r") as f:
      lines = f.readlines()

    hyp_lines = [line[2:].split("\t") for line in lines if line.startswith("H-")]
    src_lines = [line[2:].split("\t") for line in lines if line.startswith("S-")]
    tgt_lines = [line[2:].split("\t") for line in lines if line.startswith("T-")]

    hyp_lines = sorted(hyp_lines, key=lambda item: item[0])
    src_lines = sorted(src_lines, key=lambda item: item[0])
    tgt_lines = sorted(tgt_lines, key=lambda item: item[0])

    hyps = [item[-1].strip() for item in hyp_lines]
    srcs = [item[-1].strip() for item in src_lines]
    tgts = [item[-1].strip() for item in tgt_lines]

    with open(self.out_hyp_file, "w") as f:
      json.dump(hyps, f)

    with open(self.out_src_file, "w") as f:
      json.dump(srcs, f)

    with open(self.out_tgt_file, "w") as f:
      json.dump(tgts, f)

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

class MBRJob(Job):
  """
  Evaluate a fairseq transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
    'selective_prediction': False
  }

  def __init__(
      self,
      hyps,
      refs,
      n,
      flatten=False,
      src=None,
      time_rqmt=24,
      mem_rqmt=32,
      cpu_rqmt=1,
      gpu_rqmt=1,
      gpumem=12,
      sbatch_args=None,
      keep_only_best=False,
      distributed=False,
      selective_prediction=False,
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
    self.flatten = flatten
    self.hyps = hyps
    self.n = n
    self.refs = refs
    self.src = src

    self.distributed = distributed

    self.selective_prediction = selective_prediction

    self.python_exe = gs.PYTHON_EXE

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "sbatch_args": sbatch_args,
    }

    if gpu_rqmt > 0:
        self.rqmt["gpumem"] = gpumem

    self.out_hyp_file = self.output_path("hyps.txt", directory=False)
    self.out_risks_file = self.output_path("risks.json", directory=False)

  def _get_run_cmd(self):
    run_cmd = [
        tk.uncached_path(self.python_exe),
        "/path/to/iclr2025-mbr-uncertainty/mbr/mbr.py",
        ";".join([hyp.get_path() for hyp in self.hyps]),
        "--refs", self.refs if isinstance(self.refs, str) else self.refs.get_path(),
        "-n", str(self.n),
        "--eval-metric", "bleu",
        "--out-hyp-file", self.out_hyp_file.get_path()
    ]
    if self.src is not None:
      run_cmd.extend([
          "--src", self.src
      ])
    if self.flatten:
      run_cmd.append("--flatten")
    if self.selective_prediction:
      run_cmd.extend([
          "--out-risk-file", self.out_risks_file.get_path()
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

class NeuralMBRJob(Job):
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
      refs,
      n,
      flatten=False,
      src=None,
      time_rqmt=24,
      mem_rqmt=32,
      cpu_rqmt=1,
      gpu_rqmt=1,
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
    self.flatten = flatten
    self.hyps = hyps
    self.n = n
    self.refs = refs
    self.src = src

    self.distributed = distributed

    self.python_exe = gs.PYTHON_EXE

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "sbatch_args": sbatch_args,
    }

    if gpu_rqmt > 0:
        self.rqmt["gpumem"] = gpumem

    self.out_hyp_file = self.output_path("hyps.txt", directory=False)

  def _get_run_cmd(self):
    run_cmd = [
        tk.uncached_path(self.python_exe),
        "/path/to/iclr2025-mbr-uncertainty/mbr/mbr.py",
        ";".join([hyp if isinstance(hyp, str) else hyp.get_path() for hyp in self.hyps]),
        "--refs", self.refs if isinstance(self.refs, str) else self.refs.get_path(),
        "-n", str(self.n),
        "--metric", "bertscore",
        "--eval-metric", "bleu",
        "--out-hyp-file", self.out_hyp_file.get_path()
    ]
    if self.src is not None:
      run_cmd.extend([
          "--src", self.src
      ])
    if self.flatten:
      run_cmd.append("--flatten")
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
