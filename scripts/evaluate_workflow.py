"""Evaluate workflow prediction accuracy"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import json
import logging
from itertools import groupby
from pathlib import Path
from typing import List

import click
import pandas as pd
from Levenshtein import ratio
from natsort import natsorted

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def aggregate_lab_actions(actions: List[str]) -> List[str]:
    """
    Aggregate actions to obtain workflow with sequence of steps.

    Args:
        actions: List pf actions predicted for each video.

    Returns:
        workflow: list of steps in experimental workflow.
    """

    workflow = []
    if len(actions) < 1:
        logging.error(
            "No action predicted for workflow. Make sure labels and action predictions are set properly in input file"
        )
    else:
        actions = [s for s in actions if s != "No action"]
        workflow.append(actions[0])
        for i in range(len(actions[1:])):
            if actions[i + 1] != actions[i]:
                workflow.append(actions[i + 1].split(" ")[0])

    return workflow


def aggregate_lab_motions(motions: List[str]) -> List[str]:
    """
    Aggregate motions to obtain workflow with sequence of steps.

    Args:
        motions: List pf motions predicted for each video.

    Returns:
        workflow: list of steps in experimental workflow.
    """

    workflow = []
    if len(motions) < 1:
        logging.error(
            "No action predicted for workflow. Make sure labels and action predictions are set properly in input file"
        )
    else:
        motions = [motion for motion in motions if motion != "no action"]

        # Remove isolated motions
        filtered_motions = [
            group[0] for group in groupby(motions) if len(list(group[1])) > 1
        ]

        # Unique identifier for all devleopment steps
        workflow = [
            motion if "development" not in motion else "development"
            for motion in filtered_motions
        ]

    return workflow


@click.command()
@click.option("--output_folder", required=True, type=click.Path(exists=True))
@click.option("--dataset_name", default="lab-actions", type=str)
def main(output_folder: Path, dataset_name: str) -> None:
    """
    Evaluate experimental workflow prediction accuracy.

    Args:
        output_folder: Path to folder containing action prediction and labels per video.
        dataset_name: dataset type to be evaluated, 'lab-actions' or 'lab-motions'.
    """

    # Load and sort labels and predictions
    predictions = pd.read_csv(
        Path(output_folder, "video_predictions_testing.csv"), index_col=0
    )
    labels = pd.read_csv(Path(output_folder, "video_labels_testing.csv"), index_col=0)
    predictions = predictions.reindex(natsorted(predictions.index))
    labels = labels.reindex(natsorted(labels.index))

    video_list = predictions.index.to_list()
    predictions["workflow"] = [video.split("_")[0] for video in video_list]

    levensthein = []
    for workflow in list(set([video.split("_")[0] for video in video_list])):

        test_videos = [video for video in list(predictions[predictions["workflow"] == workflow].index) if video in list(labels.index)]

        if dataset_name == "lab-actions":
            predicted_workflow = aggregate_lab_actions(
                list(predictions[predictions["workflow"] == workflow]["prediction"])
            )
            actual_workflow = aggregate_lab_actions(
                list(
                    labels.loc[test_videos][
                        "label"
                    ]
                )
            )
        elif dataset_name == "lab-motions":
            predicted_workflow = aggregate_lab_motions(
                list(predictions[predictions["workflow"] == workflow]["prediction"])
            )
            actual_workflow = aggregate_lab_motions(
                list(
                    labels.loc[test_videos][
                        "label"
                    ]
                )
            )
        else:
            logging.error(
                f"No workflow evaluation method available for dataset type {dataset_name}. Evaluation is available only for 'lab-actions' and 'lab-motions'."
            )

        levensthein.append(
            ratio(",".join(predicted_workflow), ",".join(actual_workflow))
        )

    logging.info(f"Average Levensthein ratio is {sum(levensthein)/len(levensthein)}")

    with open(f"{output_folder}/metrics.json", "w+") as f:
        json.dump({"Levenshtein Ratio": sum(levensthein) / len(levensthein)}, f)
    return None


if __name__ == "__main__":
    main()
