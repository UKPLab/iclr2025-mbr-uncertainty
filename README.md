# Uncertainty-Aware Decoding with Minimum Bayes' Risk - ICLR 2025

[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/search/cs?searchtype=author&query=Daheim,+N)
[![License](https://img.shields.io/github/license/UKPLab/iclr2025-mbr-uncertainty)](https://opensource.org/licenses/Apache-2.0)

This is the repositoy for ``Uncertainty-Aware Decoding with Minimum Bayes' Risk'' (ICLR 2025).



> **Abstract:** Despite their outstanding performance in the majority of scenarios, contemporary language models still occasionally generate undesirable outputs, for example, hallucinated text. While such behaviors have previously been linked to uncertainty, there is a notable lack of methods that actively consider uncertainty during text generation. In this work, we show how Minimum Bayes’ Risk (MBR) decoding, which selects model generations according to an expected risk can be generalized into a principled uncertainty-aware decoding method. In short, we account for model uncertainty during decoding by incorporating a posterior over model parameters into MBR’s computation of expected risk. We show that this modified expected risk is useful for both choosing outputs and deciding when to abstain from generation and can provide improvements without incurring overhead. We benchmark different methods for learning posteriors and show that performance improves with prediction diversity.

Contact person: [Nico Daheim](mailto:nico.daheim@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Getting Started

The repository contains code to run uncertainty-aware MBR, as well as to train models using [huggingface transformers](https://github.com/huggingface/transformers) and [fairseq](https://github.com/facebookresearch/fairseq).
Both have been adapted in this repository to accustom training with variational learning using the [IVON optimizer](https://openreview.net/forum?id=cXBv07GKvk) for which we use [the official implementation](https://github.com/team-approx-bayes/ivon). Our MBR implementation is based on the [implementation](https://github.com/deep-spin/qaware-decode) of [``Quality-Aware Decoding for Neural Machine Translation''](https://aclanthology.org/2022.naacl-main.100.pdf).

If you are only interested in experiments based on huggingface, then running
```
pip install -r requirements.txt
```
will install all necessary packages.

When using fairseq, the following has to be run in addition:
```
cd fairseq/
pip install --editable ./
```

The experiments of the paper were organized using the workflow manager [Sisyphus](https://github.com/rwth-i6/sisyphus). If you would like to make use of it, too, then please run:
```
git clone git@github.com:rwth-i6/sisyphus.git
cd sisyphus/
pip install -r requirements.txt
cd ..
mkdir alias
mkdir output
mkdir work
```
Sisyphus will use the directories as follows:
  1. `alias`: It's possible to identify aliases for each job to identify it quickly (as a default, a hash is appended to the jobclass name as an identifier), and sisyphus adds a symlink to the job under the alias.
  2. `output`: `tk.register_output("name", job_class.file)` registers an output under the filename `name` in the output folder that symlinks to `job_class.file`
  3. `work`: All jobs will be placed here under their hash.

## Usage

### Running experiments using Sisyphus

Examples for training with Sisyphus are found in the `config/` folder.
Running either training using huggingface or fairseq on a Slurm cluster only requires
```
cd iclr2025-mbr-uncertainty
sisyphus/sis --config config config/huggingface.py
```
or
```
cd iclr2025-mbr-uncertainty
sisyphus/sis --config config config/fairseq.py
```

The examples will run finetuning with LoRA of GEMMA-2B on IWSLT17 and a from-scratch training of a Transformer-base model on IWSLT14, respectively.
The examples also show how to use our sequence-level MBR methods and single model MBR baselines.
Token-level posteriors can be used easily in fairseq according to the documentation in their [repo](https://github.com/facebookresearch/fairseq).

For each part of the pipeline, Sisyphus Jobs are defined that wrap python scripts for training, decoding, mbr, and evaluation.

### Using scripts

For training, there is an example configuration file in `scripts`. The file can be invoked via:

```
cd huggingface/code/
python3 train.py ../../scripts/train_config.json
```
The example will run a similar training to the config in `config/huggingface.py` and train GEMMA-2B on IWSLT17 using LoRA.

Similarly, decoding can be run by 
```
cd huggingface/code/
python3 predict.py ../../scripts/search_config.json
```
Here, the config files describe all relevant parameters, such as the model to be used, the dataset, a prompt, random seed, whether to sample during decoding, etc.

For MBR, the script in `mbr/mbr.py` can be used.
The arguments follow the implementation from [`qaware-decode`](https://github.com/deep-spin/qaware-decode) but there are two important changes:

The first argument is can be the path to one prediction file but also a semi-colon-separated concatenation of multiple paths to prediction files to perform uncertainty-aware decoding via model combination.

Then, using `--flatten` concatenates all these hypothesis sets for each sample, i.e. performs Eq. 9, while not passing the argument will calculate utilities individually for each sample and then sum them, i.e. perform Eq. 10.

For evaluation, the script in `huggingface/code/evaluation.py` can be used. Besides predictions, hypothesis set size, etc. the argument `eval_task` has to be passed which selects the metrics for the given task, for example, rouge for summarization.

### Expected results

After running the jobs in `config/huggingface.py`, the results should closely match our MBR results on IWSLT17 using GEMMA-2B in Table 1, where we average over 4 seeds.


### Code Structure

The code is mainly based on the concept of ''methods'' that are found in the `/code/mbr/methods/` folder which wrap all of the functionality needed to reproduce a certain method:
  1. Defining and loading Trainer and Data Collator classes
  2. Loading all datasets
  3. Defining and applying the preprocessing methods, defined in `/code/mbr/methods/preprocessing`

To understand how the method classes are structured it's best to check `code/mbr/methods/base.py` which defines a base class from which all methods inherit.

The main entry point for the code is `/code/mbr/main.py` that handles loading method classes, models, and running the Trainers.

## Cite

Please use the following citation:

```
@inproceedings{
daheim2025uncertaintyaware,
title={Uncertainty-Aware Decoding with Minimum Bayes' Risk},
author={Nico Daheim and Clara Meister and Thomas M{\"o}llenhoff and Iryna Gurevych},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=hPpyUv1XyQ}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 


The repo template is adapted from [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno).
