
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

ERRORS_FILE = RESULTS_DIR / "wrong_predictions.csv"
GRAPH_FILE = RESULTS_DIR / "error_distribution.png"


def load_errors():
    if not ERRORS_FILE.is_file():
        raise FileNotFoundError(
            f"Wrong predictions file not found: {ERRORS_FILE}\n"
            "Run 'python -m src.evaluation' first."
        )

    df = pd.read_csv(ERRORS_FILE)
    return df


def analyze_errors(df):
    completely_wrong = df[df["f1"] == 0]
    partial_match = df[(df["f1"] > 0) & (df["f1"] < 1)]
    full_token_match = df[df["f1"] == 1]
    low_f1 = df[df["f1"] < 0.5]

    print("\nError Classification:")
    print("Completely Wrong:", len(completely_wrong))
    print("Partial Match:", len(partial_match))
    print("Full Token Match:", len(full_token_match))
    print("Low F1 Predictions:", len(low_f1))


def plot_error_distribution(df):
    categories = [
        "Completely Wrong",
        "Partial Match",
        "Full Token Match"
    ]

    counts = [
        len(df[df["f1"] == 0]),
        len(df[(df["f1"] > 0) & (df["f1"] < 1)]),
        len(df[df["f1"] == 1])
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.bar(categories, counts)
    plt.title("Transformer QA Error Distribution")
    plt.xlabel("Error Category")
    plt.ylabel("Number of Predictions")
    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nGraph saved to: {GRAPH_FILE}")

    plt.show()


def main():
    df = load_errors()

    print("Total Non-Exact Predictions:", len(df))
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 Records:")
    print(df.head())

    analyze_errors(df)
    plot_error_distribution(df)

    for i, row in df.head(5).iterrows():
        print(f"\n{'=' * 70}")
        print(f"ERROR CASE {i + 1}")
        print("=" * 70)

        print("\nQUESTION:")
        print(row["question"])

        print("\nCONTEXT:")
        print(row["context"])

        print("\nEXPECTED ANSWER:")
        print(row["expected"])

        print("\nMODEL PREDICTION:")
        print(row["predicted"])

        print("\nF1 SCORE:")
        print(row["f1"])


if __name__ == "__main__":
    main()
