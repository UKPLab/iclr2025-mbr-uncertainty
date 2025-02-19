import argparse
import json
from collections import Counter

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def _token_f1(prediction, reference):
    prediction_tokens = tokenizer.tokenize(prediction)
    reference_tokens = tokenizer.tokenize(reference)

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def _evaluate_mt(predictions, search_data):
    from comet import download_model, load_from_checkpoint

    dataset = load_dataset(
        search_data["dataset_name"], 
        search_data["dataset_config_name"],
        split=search_data["dataset_test_split"],
        trust_remote_code=True
    )

    references = [sample["output"] for sample in dataset]
    inputs = [sample["input"] for sample in dataset]

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    data = [{"src": src, "mt": hyp, "ref": tgt}
                for src, hyp, tgt in zip(inputs, predictions, references)]

    model_output = model.predict(data, batch_size=8, gpus=1)

    score = model_output.system_score

    metrics = {
        "sbleu": round(sacrebleu.corpus_bleu(predictions, [references]).score, 2),
        "sbleu_version": sacrebleu.__version__,
    }

    metrics["comet"] = round(score*100.0, 2)

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarities = []

    srcs = inputs
    hyps = predictions

    model = SentenceTransformer('sentence-transformers/LaBSE').to("cuda:0")
    for hyp, src in tqdm(zip(hyps, srcs)):
        hyp_embedding, src_embedding = model.encode([hyp, src], convert_to_tensor=True, show_progress_bar=False)
        similarity = cos(hyp_embedding, src_embedding)
        similarities.append(similarity.cpu().numpy().item())

    metrics["labse"] = round(np.mean(similarities), 4)
    
    return metrics

def _evaluate_summarization(predictions, search_data):
    dataset = load_dataset(search_data["dataset_name"], search_data["dataset_config_name"], split=search_data["dataset_test_split"], trust_remote_code=True)
    references = [sample["output"] for sample in dataset]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1s, rougels = [], []

    for prediction, reference in zip(predictions, references):
        score = scorer.score(prediction, reference)
        rouge1s.append(score["rouge1"].fmeasure)
        rougels.append(score["rougeL"].fmeasure)


    return {
        "rouge1": round(np.mean(rouge1s), 4),
        "rougeL": round(np.mean(rougels), 4)
    }


def run(args):
    with open(args.prediction_file, "r") as f:
        predictions = json.load(f)
    with open(args.search_data_config_path, "r") as f:
        search_data = json.load(f)

    predictions = predictions[::args.nbest]

    if args.eval_task == "mt":
        scores = _evaluate_mt(predictions, search_data)
    elif args.eval_task == "summarization":
        scores = _evaluate_summarization(predictions, search_data)
    else:
        raise NotImplementedError

    with open(args.out_metric_file, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task",
        type=str,
    )
    parser.add_argument(
        "--nbest",
        type=int,
    )
    parser.add_argument(
        "--out_metric_file",
        type=str,
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
    )
    parser.add_argument(
        "--search_data_config_path",
        type=str,
    )
    parsed_args = parser.parse_args()

    run(parsed_args)