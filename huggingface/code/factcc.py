import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

def batched_FactCC(text_l, summary_l, tokenizer, model):    
    input_dict = tokenizer(text_l, summary_l, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
    for k, v in input_dict.items():
        input_dict[k] = input_dict[k].to("cuda:0")
    with torch.no_grad():
        logits = model(**input_dict).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        return logits, preds

def run(args):

    model_path = 'manueldeprada/FactCC'

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to("cuda:0")

    with open(args.prediction_file, "r") as f:
        predictions = json.load(f)
    with open(args.search_data_config_path, "r") as f:
        search_data = json.load(f)

    predictions = predictions[::args.nbest]

    dataset = load_dataset(
        search_data["dataset_name"], 
        search_data["dataset_config_name"],
        split=search_data["dataset_test_split"]
    )

    preds = []
    texts = []
    claims = []

    all_preds = []
    batch_size = 128

    for summary in dataset:
        texts.append(summary["input"])

    for prediction in predictions:
        claims.append(prediction)

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_claims = claims[i:i+batch_size]
        _, preds = batched_FactCC(batch_texts, batch_claims, tokenizer, model)
        all_preds.extend(preds.tolist())

    factcc = np.mean([pred == 1.0 for pred in preds])
    metrics = {
        "factcc": factcc
    }

    with open(args.out_metric_file, "w") as f:
        json.dump(metrics, f)


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