import copy
import json
import sys
import argparse
from functools import partial, wraps
from joblib import delayed, Parallel
from typing import List

import numpy as np
from tqdm import tqdm



class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def wrapped_partial(f, *args, **kwargs):
    return wraps(f)(partial(f, *args, **kwargs))


def build_metric_fn(
    metric_name: str,
    comet_dir: str = None,
    bleurt_dir: str = None,
    n_cpus=1,
    n_gpus=1,
    batch_size: int = 32,
    progress_bar: bool = True,
    only_sentence_level: bool = True,
):
    if metric_name == "comet":
        assert comet_dir is not None
        return partial(
            comet,
            comet_dir=comet_dir,
            comet_bsize=batch_size,
            progress_bar=progress_bar,
        )
    elif metric_name == "bleurt":
        assert bleurt_dir is not None
        return partial(bleurt, bleurt_dir=bleurt_dir, bleurt_bsize=batch_size)
    elif metric_name == "bleu":
        return partial(
            bleu,
            progress_bar=progress_bar,
            parallel=n_cpus,
            only_sentence_level=only_sentence_level,
        )
    elif metric_name == "bertscore":
        return partial(
            bertscore,
            progress_bar=progress_bar,
            parallel=n_cpus,
        )


def comet(
    hyps: List[str],
    refs: List[str],
    srcs: List[str],
    comet_dir: str = None,
    comet_model: str = "wmt20-comet-da",
    comet_bsize: int = 64,
    progress_bar: bool = True,
):
    from comet import download_model, load_from_checkpoint

    # download comet and load
    comet_path = download_model(comet_model, comet_dir)
    comet_model = load_from_checkpoint(comet_path)
    comet_input = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
    ]
    # sentence-level and corpus-level COMET
    return comet_model.predict(
        comet_input,
        batch_size=comet_bsize,
        sort_by_mtlen=True,
        progress_bar=progress_bar,
        num_workers=1
    )


def bleurt(
    hyps: List[str],
    refs: List[str],
    srcs: List[str] = None,
    bleurt_dir: str = None,
    bleurt_bsize: str = 64,
):
    from bleurt import score

    bleurt_scorer = score.LengthBatchingBleurtScorer(bleurt_dir)
    bleurt_scores = bleurt_scorer.score(
        references=refs,
        candidates=hyps,
        batch_size=bleurt_bsize,
    )
    assert type(bleurt_scores) == list
    return bleurt_scores, np.array(bleurt_scores).mean()

def bertscore(
    hyps: List[str],
    refs: List[str],
    srcs: List[str] = None,
    bleurt_dir: str = None,
    bleurt_bsize: str = 32,
    progress_bar: bool = True,
    parallel=False
):
    import bert_score
    score_fn = lambda *args: bert_score.corpus_bleu(*args)[-1]
    # iterator = (
    #     delayed(score_fn)(hyp, ref, model_type="microsoft/deberta-base-mnli", batch_size=bleurt_bsize) if parallel > 1 else bleu_fn(hyp, [ref])
    #     for hyp, ref in zip(hyps, refs)
    # )
    bert_scores = bert_score.score(hyps, refs, model_type="microsoft/deberta-base-mnli", batch_size=bleurt_bsize, verbose=True)[-1].cpu().numpy().tolist()
    assert type(bert_scores) == list
    return bert_scores, np.array(bert_scores).mean()


def bleu(
    hyps: List[str],
    refs: List[str],
    srcs: List[str] = None,
    progress_bar: bool = True,
    parallel: int = 1,
    only_sentence_level: bool = True,
):
    import sacrebleu

    bleu_fn = lambda *args: sacrebleu.sentence_bleu(*args).score
    iterator = (
        delayed(bleu_fn)(hyp, [ref]) if parallel > 1 else bleu_fn(hyp, [ref])
        for hyp, ref in zip(hyps, refs)
    )

    if parallel > 1 and progress_bar:
        iterator = ProgressParallel(
            total=len(hyps),
            n_jobs=parallel,
            batch_size=50000,
            pre_dispatch="16*n_jobs",
        )(iterator)
    elif progress_bar:
        iterator = tqdm(iterator, total=len(hyps))

    sentence_scores = list(iterator)

    corpus_score = None
    if not only_sentence_level:
        corpus_score = sacrebleu.corpus_bleu(hyps, [refs]).score

    return sentence_scores, corpus_score



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hyps",
        type=str,
        help="Files containing all hypothesis grouped per sentence, with ``num_samples*sentences``, separated by ';' ",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Number of hypothesis per sentence",
    )
    parser.add_argument(
        "--refs",
        default=None,
        type=str,
        help="File containing reference translations. If passed, will be used for evaluating the chosen hypothesis.",
    )
    parser.add_argument(
        "--metric",
        default="bleu",
        choices=["bleu", "comet", "bleurt", "bertscore"],
        help="Metric to use. Currently only bleu, comet and bleurt are supported. Check `qaware_decode/metrics.py` for more details.",
    )
    parser.add_argument(
        "--eval-metrics",
        default=["bleu", "comet"],
        choices=["bleu", "comet", "bleurt"],
        help="Metric(s) to evaluate the chosen hypothesis",
        nargs="+",
    )
    parser.add_argument(
        "--num-subsamples",
        type=int,
        default=None,
        help="Number of subsamples to use for MBR expectation",
    )
    parser.add_argument(
        "--comet-dir",
        default=".cache/qaware_decode/comet",
        help="Directory containing the comet models.",
    )
    parser.add_argument(
        "--bleurt-dir",
        default=".cache/qaware_decode/bleurt",
        help="Directory containing the bleurt models.",
    )
    parser.add_argument(
        "--n-cpus",
        default=1,
        type=int,
        help="number of cpus to use for cpu based metrics",
    )
    parser.add_argument(
        "-n-gpus",
        default=1,
        type=int,
        help="number of gpus to use for gpu based metrics",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="batch size for gpu-based metrics"
    )
    parser.add_argument(
        "--src",
        default=None,
        help="File containing source sentences. Only necessary if metric is comet",
    )
    parser.add_argument(
        "--save-mbr-utils",
        default=None,
        help="File to save utility scores, one per hypothesis",
    )
    parser.add_argument(
        "--out-hyp-file",
        default=None,
    )
    parser.add_argument(
        "--out-risk-file",
        default=None,
    )
    parser.add_argument('--flatten', default=False, action='store_true')
    parser.add_argument("--seed")
    return parser.parse_args()


def mbr_corpus(
    hyps: List[List[str]],
    metric: callable,
    srcs: List[str] = None,
    num_subsamples: int = None,
    aggregation: str = "mean",
    scores: List[List[float]] = None,
) -> List[List[float]]:
    """
    Computes per-sample MBR for a corpus. Returns the (negative) risk of each sample

    Args:
        hyps: list of hypotheses for each sample
        metric: metric to compute MBR
        srcs: source for each sample. only used for src-based metrics (comet)
        num_subsamples: number of subsamples to use for MBR
        aggregation: how to aggregate the subsamples. "mean" or "max"

    Returns:
        neg_risk: negative risk of each sample
    """

    # if srcs is not None:
        # assert len(hyps) == len(srcs), f"{len(hyps)} != {len(srcs)}"

    num_samples = len(hyps[0])
    use_subsampling = num_subsamples is not None and num_subsamples < num_samples

    # flattens the source
    cands = []
    refs = []
    dup_srcs = [] if srcs is not None else None
    for i, samples in enumerate(hyps):
        indices = (
            np.random.choice(num_samples, num_subsamples, replace=False)
            if use_subsampling
            else list(range(num_samples))
        )
        for cand in samples:
            for ref_id in indices:
                cands.append(cand)
                refs.append(samples[ref_id])
                if srcs is not None:
                    dup_srcs.append(srcs[i])

    flat_metric_matrixes, _ = metric(cands, refs, srcs=dup_srcs)

    # unflattens the metrics into a N*S*T tensor
    metric_matrixes = []
    for i, _ in enumerate(hyps):
        metric_matrixes.append([])
        for j in range(num_samples):
            metric_matrixes[i].append([])
            for k in range(num_subsamples if use_subsampling else num_samples):
                metric_matrixes[i][j].append(
                    flat_metric_matrixes[
                        i * num_samples * num_samples + j * num_samples + k
                    ]
                )

    metric_matrixes = np.array(metric_matrixes)

    if aggregation == "mean":
        neg_risks = metric_matrixes.mean(axis=2).tolist()
    elif aggregation == "weighted_mean":
        assert scores is not None
        # TODO implemented
        raise ValueError("weighted_mean not implemented")
    else:
        raise ValueError(f"aggregation {aggregation} not implemented")

    return neg_risks


def main():
    args = parse_args()

    hyp_files = args.hyps.split(";")

    # combine hypothesis sets while maintaining sample counts
    if args.flatten:

        hyps = []
        for n, hyp_file in enumerate(hyp_files):
            with open(hyp_file, encoding="utf-8") as hyp_f:
                flat_hyps = [line.strip() for line in json.load(hyp_f)]
                assert len(flat_hyps) % args.num_samples == 0

                # unflatten the hypotheses
                for i in range(0, len(flat_hyps) // args.num_samples):
                    if n == 0:
                        hyps.append([])
                    for j in range(args.num_samples):
                        hyps[i].append(flat_hyps[i * args.num_samples + j])

        srcs = None
        if args.src is not None:
            with open(args.src, encoding="utf-8") as src_f:
                srcs = [line.strip() for line in json.load(src_f)]

        metric = build_metric_fn(
            args.metric,
            comet_dir=args.comet_dir,
            bleurt_dir=args.bleurt_dir,
            n_cpus=args.n_cpus,
        )

        neg_risk = mbr_corpus(
            hyps,
            metric=metric,
            srcs=srcs,
            num_subsamples=args.num_subsamples,
        )

        if args.save_mbr_utils is not None:
            mbr_utils = open(args.save_mbr_utils, "w")

        predictions = []
        for sample_hyps, sample_utilities in zip(hyps, neg_risk):
            predictions.append(sample_hyps[np.argmax(sample_utilities)])
            if args.save_mbr_utils is not None:
                for util in sample_utilities:
                    print(f"mbr-util={util}", file=mbr_utils)

    # sum over individual per-model utilities
    else:

        # num_models, num_srcs, num_hyps
        all_hyps = []
        for n, hyp_file in enumerate(hyp_files):
            with open(hyp_file, encoding="utf-8") as hyp_f:
                all_hyps.append([])
                flat_hyps = [line.strip() for line in json.load(hyp_f)]
                assert len(flat_hyps) % args.num_samples == 0

                # unflatten the hypotheses
                for i in range(0, len(flat_hyps) // args.num_samples):
                    all_hyps[n].append([])
                    for j in range(args.num_samples):
                        all_hyps[n][i].append(flat_hyps[i * args.num_samples + j])

        srcs = None
        if args.src is not None:
            with open(args.src, encoding="utf-8") as src_f:
                srcs = [line.strip() for line in json.load(src_f)]

        metric = build_metric_fn(
            args.metric,
            comet_dir=args.comet_dir,
            bleurt_dir=args.bleurt_dir,
            n_cpus=args.n_cpus,
        )

        model_risks = []
        for model_hyps in all_hyps:
            model_risks.append(mbr_corpus(
                model_hyps,
                metric=metric,
                srcs=srcs,
                num_subsamples=args.num_subsamples,
            ))

        if args.save_mbr_utils is not None:
            mbr_utils = open(args.save_mbr_utils, "w")

        predictions = []

        utilities = {model_idx: [] for model_idx in range(len(all_hyps))}

        for model_idx, samples in enumerate(all_hyps):
            for sample_idx, hypotheses in enumerate(samples):
                utilities[model_idx].append({})
                for hypothesis_idx, hypothesis in enumerate(hypotheses):
                    utility = model_risks[model_idx][sample_idx][hypothesis_idx]
                    utilities[model_idx][sample_idx][hypothesis] = utility
                    
        final_utilities = [{} for sample_idx in range(len(all_hyps[0]))]

        for model_idx, samples in enumerate(all_hyps):
            for sample_idx, hypotheses in enumerate(samples):
                for hypothesis_idx, hypothesis in enumerate(hypotheses):
                    if hypothesis not in final_utilities[sample_idx]:
                        final_utilities[sample_idx][hypothesis] = utilities[model_idx][sample_idx][hypothesis]
                        for model_idx_ in range(len(utilities.keys())):
                            if model_idx != model_idx_:
                                for hypothesis_, utility in utilities[model_idx_][sample_idx].items(): 
                                    if hypothesis == hypothesis_:
                                        final_utilities[sample_idx][hypothesis] += utility

        predictions = []

        for sample in final_utilities:
            predictions.append(max(sample, key=sample.get))

    if args.out_hyp_file is not None:
        with open(args.out_hyp_file, "w") as f:
            f.write(json.dumps(predictions))

    print(predictions)

if __name__ == "__main__":
    main()
