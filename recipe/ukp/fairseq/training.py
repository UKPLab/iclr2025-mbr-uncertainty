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

class FairseqTrainingJob(Job):
  """
  Train a fairseq transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      config,
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
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)

  def _get_run_cmd(self):
      run_cmd = [
          "fairseq-train",
          "--save-dir", self.out_checkpoints_dir.get_path()
      ]
      if self.resume_from_checkpoint is not None:
          pass # TODO: implement
      run_cmd.extend(self.config)
      if self.distributed:
        run_cmd = [
          tk.uncached_path(self.python_exe),
          '-m', 'torch.distributed.launch', '--nproc_per_node', str(self.rqmt['gpu']),
        ] + run_cmd[1:]
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
