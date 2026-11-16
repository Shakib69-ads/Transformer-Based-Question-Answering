
import gc
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.data_loader import load_squad_dataset
from src.model import predict_answer
from src.evaluation import compute_exact_match, compute_f1


# Model configurations
MODEL_CONFIGS = {
    "DistilBERT": "distilbert/distilbert-base-uncased-distilled-squad",
    "BERT-Cased": "deepset/bert-base-cased-squad2",
    "RoBERTa": "deepset/roberta-base-squad2"
}

NUM_SAMPLES = 200
SEED = 42


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_FILE = RESULTS_DIR / "model_comparison_results.csv"


def load_qa_model(model_name):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def evaluate_qa_model(model, tokenizer, device, validation_data):
    total_em = 0.0
    total_f1 = 0.0

    start_time = time.perf_counter()

    for index, example in enumerate(validation_data):
        question = example["question"]
        context = example["context"]
        expected_answers = example["answers"]["text"]

        predicted_answer, _ = predict_answer(
            question,
            context,
            model,
            tokenizer,
            device
        )

        em = max(
            compute_exact_match(predicted_answer, answer)
            for answer in expected_answers
        )

        f1 = max(
            compute_f1(predicted_answer, answer)
            for answer in expected_answers
        )

        total_em += em
        total_f1 += f1

        if (index + 1) % 20 == 0:
            print(
                f"Evaluated {index + 1}/{len(validation_data)}"
            )

    elapsed_time = time.perf_counter() - start_time
    sample_count = len(validation_data)

    return {
        "Exact Match (%)": round(
            total_em / sample_count * 100, 2
        ),
        "F1 Score (%)": round(
            total_f1 / sample_count * 100, 2
        ),
        "Inference Time (s)": round(
            elapsed_time, 2
        ),
        "Avg Time/Question (s)": round(
            elapsed_time / sample_count, 3
        )
    }


def main():
    # Ensure results folder exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SQuAD Validation Dataset...")

    dataset = load_squad_dataset()

    validation_data = (
        dataset["validation"]
        .shuffle(seed=SEED)
        .select(range(NUM_SAMPLES))
    )

    results = []

    for model_label, model_name in MODEL_CONFIGS.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {model_label}")
        print(f"Checkpoint: {model_name}")
        print(f"{'=' * 50}")

        tokenizer, model, device = load_qa_model(
            model_name
        )

        metrics = evaluate_qa_model(
            model,
            tokenizer,
            device,
            validation_data
        )

        result = {
            "Model": model_label,
            "Checkpoint": model_name,
            **metrics
        }

        results.append(result)

        print(f"\n{model_label} Results:")
        print(
            f"Exact Match: "
            f"{metrics['Exact Match (%)']:.2f}%"
        )
        print(
            f"F1 Score: "
            f"{metrics['F1 Score (%)']:.2f}%"
        )
        print(
            f"Inference Time: "
            f"{metrics['Inference Time (s)']:.2f} seconds"
        )

        # Save comparison results in results/
        pd.DataFrame(results).to_csv(
            COMPARISON_FILE,
            index=False
        )

        del model, tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "\n========== FINAL MODEL COMPARISON =========="
    )

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print(
        f"\nResults saved to: {COMPARISON_FILE}"
    )


if __name__ == "__main__":
    main()
