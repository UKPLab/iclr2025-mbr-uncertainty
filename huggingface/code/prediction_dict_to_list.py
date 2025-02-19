import argparse
import json

def run(args):
    with open(args.prediction_file, "r") as f:
        predictions = json.load(f)

    predictions = [prediction["sequence"] for prediction in predictions]

    with open(args.out_search_file, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_search_file",
        type=str,
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
    )
    parsed_args = parser.parse_args()

    run(parsed_args)