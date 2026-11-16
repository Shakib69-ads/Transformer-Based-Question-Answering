
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data_loader import load_squad_dataset
from src.preprocessing import load_tokenizer
from src.model import load_model, predict_answer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
WRONG_PREDICTIONS_FILE = RESULTS_DIR / "wrong_predictions.csv"

ERROR_COLUMNS = [
    "question",
    "context",
    "expected",
    "predicted",
    "f1"
]


def normalize_answer(text):
    text = text.lower()
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def compute_exact_match(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_truth = normalize_answer(ground_truth)

    return int(normalized_prediction == normalized_truth)


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)

    return f1


def evaluate_model(
    model,
    tokenizer,
    device,
    num_samples=200,
    seed=42
):
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(
            "num_samples must be a positive integer."
        )

    dataset = load_squad_dataset()
    validation_data = dataset["validation"]

    if len(validation_data) == 0:
        raise ValueError("Validation dataset is empty.")

    num_samples = min(num_samples, len(validation_data))

    validation_data = validation_data.shuffle(seed=seed)
    validation_data = validation_data.select(range(num_samples))

    total_em = 0
    total_f1 = 0
    wrong_predictions = []

    for i, sample in enumerate(validation_data, start=1):
        ground_truths = sample["answers"]["text"]

        if not ground_truths:
            raise ValueError(
                f"No ground truth answers found for sample {i}."
            )

        prediction, _ = predict_answer(
            sample["question"],
            sample["context"],
            model,
            tokenizer,
            device
        )

        em = max(
            compute_exact_match(prediction, truth)
            for truth in ground_truths
        )

        f1 = max(
            compute_f1(prediction, truth)
            for truth in ground_truths
        )

        total_em += em
        total_f1 += f1

        if em == 0:
            wrong_predictions.append({
                "question": sample["question"],
                "context": sample["context"],
                "expected": ground_truths,
                "predicted": prediction,
                "f1": f1
            })

        if i % 20 == 0 or i == num_samples:
            print(f"Evaluated: {i}/{num_samples}")

    return {
        "exact_match": 100 * total_em / num_samples,
        "f1": 100 * total_f1 / num_samples,
        "wrong_predictions": wrong_predictions
    }


def main():
    tokenizer = load_tokenizer()
    model, device = load_model()

    num_samples = 200

    results = evaluate_model(
        model,
        tokenizer,
        device,
        num_samples=num_samples,
        seed=42
    )

    print("\nTotal Samples:", num_samples)
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print(f"F1 Score: {results['f1']:.2f}%")
    print(
        "Non-Exact Predictions:",
        len(results["wrong_predictions"])
    )

    for item in results["wrong_predictions"][:5]:
        print("\nQuestion:", item["question"])
        print("Expected:", item["expected"])
        print("Predicted:", item["predicted"])
        print(f"F1: {item['f1']:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        results["wrong_predictions"],
        columns=ERROR_COLUMNS
    )

    df.to_csv(
        WRONG_PREDICTIONS_FILE,
        index=False
    )

    print(
        f"\nWrong predictions saved to: "
        f"{WRONG_PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()
