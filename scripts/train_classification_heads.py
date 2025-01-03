"""Train xgboost head on xCLIP or Video-LLaVa embeddings"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

LABELS = {
    "lab-actions": [
        "Add",
        "AnalyticalMeasurement",
        "CollectLayer",
        "MeasureLiquid",
        "MeasureSolid",
        "NoAction",
        "PhaseSeparation",
        "Stir",
    ],
    "lab-motion": [
        "circular development",
        "figure eight development",
        "no action",
        "puddling development",
        "rinsing",
    ],
}


@click.command()
@click.option("--video_embeddings_path", type=click.Path(exists=True))
@click.option("--output_folder", type=click.Path(exists=True))
@click.option("--dataset_name", default="lab-actions", type=str)
@click.option("--n_estimators", type=int, default=100)
@click.option("--max_depth", type=int, default=8)
@click.option("--learning_rate", type=float, default=0.5)
def main(
    video_embeddings_path: Path,
    output_folder: Path,
    dataset_name: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> None:
    """
    Trains an xgboost model on the video embeddings to predict video activity labels.

    Args:
        video_folder (Path): The folder containing the videos.
        output_folder (Path): The folder where the results of xgboost prediction will be saved.
    """

    # Load embeddings and labels
    xgboost_data = {}
    for split in ["training", "validation", "testing"]:
        print(
            f"Shape {torch.load(Path(video_embeddings_path, f'video_embeddings_{split}.pt'))}"
        )
        embeddings = torch.load(
            Path(video_embeddings_path, f"video_embeddings_{split}.pt")
        ).squeeze(1)
        labels = pd.read_csv(Path(video_embeddings_path, f"video_labels_{split}.csv"))
        xgboost_data[split] = {
            "embeddings": embeddings,
            "labels": list(labels["label"]),
        }
    logger.info("Embeddings and labels loaded.")

    # Init xgboost and train
    logger.info("Training xgboost on video embeddings ...")

    # Best hyperparams
    clr = GradientBoostingClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    clr.fit(xgboost_data["training"]["embeddings"], xgboost_data["training"]["labels"])
    labels_predict_test = clr.predict(xgboost_data["testing"]["embeddings"]).tolist()
    test_f1_score = f1_score(
        xgboost_data["testing"]["labels"], labels_predict_test, average="weighted"
    )
    logger.info(f"F1-score on test set for dataset {dataset_name}: {test_f1_score}.")

    # Save and plot results

    labels["prediction"] = labels_predict_test
    labels.to_csv(f"{output_folder}/video_predictions_testing.csv", index=False)

    with open(f"{output_folder}/metrics.json", "w") as f:
        json.dump({"f1_score": test_f1_score}, f)

    _ = ConfusionMatrixDisplay.from_predictions(
        xgboost_data["testing"]["labels"],
        labels_predict_test,
        cmap="Purples",
        display_labels=LABELS[dataset_name],
    )
    plt.title(f"XGB {dataset_name}")
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/XGB_{dataset_name}_confusion_matrix.png")


if __name__ == "__main__":
    main()
