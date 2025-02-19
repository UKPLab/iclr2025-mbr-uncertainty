import argparse
import itertools
import json
import numpy as np

def run(args):
    n = 20

    all_predictions = []
    for prediction_file in args.predictions.split(";"):
        with open(prediction_file, "r") as f:
            predictions = json.load(f)
            references = [float(i["reference"]) for i in predictions[::n]]
            predictions = [float(i["sequence"][:4]) for i in predictions]
            all_predictions.append(predictions)

    if args.flatten:
        n = len(args.predictions.split(";")) * n
        all_predictions = [list(itertools.chain.from_iterable(all_predictions))]

    model_means = []
    for i in range(0, len(all_predictions[0]) // n):
        local_means = []
        for model_predictions in all_predictions:
            local_means.append(np.mean(model_predictions[i*n:(i+1)*n]))
        model_means.append(local_means)

    final_predictions = []
    for sample in model_means:
        final_predictions.append(round(float(np.mean(sample)), 2))

    rmse = []
    
    for prediction, reference in zip(final_predictions, references):
        rmse.append((prediction - reference)**2)

    rmse = np.sqrt(np.mean(rmse))

    metrics = {
        "rmse": rmse
    }

    with open(args.out_metric_file, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_metric_file",
        type=str,
    )
    parser.add_argument(
        "--predictions",
        type=str,
    )
    parser.add_argument(
        "--flatten",
        default=False,
        action="store_true"
    )
    parsed_args = parser.parse_args()

    run(parsed_args)