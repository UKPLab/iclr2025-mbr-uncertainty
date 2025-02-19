import copy
import json
import os
import subprocess as sp
import sys

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed
from i6_core.tools.download import DownloadJob

Path = setup_path(__package__)

class CalculateMetricsJob(Job):
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      eval_task,
      prediction_file,
      search_data_config,
      nbest=1,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.eval_task = eval_task
    self.prediction_file = prediction_file
    self.search_data_config = search_data_config
    self.python_exe = gs.PYTHON_EXE
    self.nbest = nbest

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

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "evaluation.py"),
          '--eval_task', self.eval_task,
          '--out_metric_file', self.out_metric_file.get_path(),
          '--prediction_file', self.__get_path_or_str(self.prediction_file),
          '--search_data_config_path', self.out_config_file.get_path(),
          '--nbest', str(self.nbest)
      ]
      return run_cmd

  def create_files(self):
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(self.search_data_config, fp)

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

      return super().hash(hash_kwargs)

  def __get_path_or_str(self, path):
    if isinstance(path, str):
      return path
    else:
      return path.get_path()

class PredictionDictToListJob(Job):
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      prediction_file,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=0,
      python_exe=None,
      sbatch_args=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.prediction_file = prediction_file
    self.python_exe = gs.PYTHON_EXE

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": 12,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_search_file = self.output_path("search_output.json")

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "prediction_dict_to_list.py"),
          '--out_search_file', self.out_search_file.get_path(),
          '--prediction_file', self.__get_path_or_str(self.prediction_file),
      ]
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

      return super().hash(hash_kwargs)

  def __get_path_or_str(self, path):
    if isinstance(path, str):
      return path
    else:
      return path.get_path()

class RegressionMBRJob(Job):
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      prediction_file,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=0,
      python_exe=None,
      sbatch_args=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.prediction_files = prediction_file
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
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_metric_file = self.output_path("metrics.json")

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "mbr_regression.py"),
          '--out_metric_file', self.out_metric_file.get_path(),
          '--predictions', ";".join([self.__get_path_or_str(prediction_file) for prediction_file in self.prediction_files])
      ]
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

      return super().hash(hash_kwargs)

  def __get_path_or_str(self, path):
    if isinstance(path, str):
      return path
    else:
      return path.get_path()


class FactCCJob(Job):
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      prediction_file,
      search_data_config,
      nbest=1,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.prediction_file = prediction_file
    self.search_data_config = search_data_config
    self.python_exe = gs.PYTHON_EXE
    self.nbest = nbest

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

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "factcc.py"),
          '--out_metric_file', self.out_metric_file.get_path(),
          '--prediction_file', self.__get_path_or_str(self.prediction_file),
          '--search_data_config_path', self.out_config_file.get_path(),
          '--nbest', str(self.nbest)
      ]
      return run_cmd

  def create_files(self):
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(self.search_data_config, fp)

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

      return super().hash(hash_kwargs)

  def __get_path_or_str(self, path):
    if isinstance(path, str):
      return path
    else:
      return path.get_path()