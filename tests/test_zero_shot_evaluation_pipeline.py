"""Tests workflow aggregation method."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

from pathlib import Path
from scripts.evaluate_workflow import aggregate_lab_actions
import pandas as pd
from natsort import natsorted


def test_lab_actions_workflow_aggregation() -> None:
    """Function to test workflow aggregation."""

    data = pd.read_csv(
        Path("resources/example_predictions/video_predictions_testing.csv"),
        index_col="video",
    )
    data = data.reindex(natsorted(data.index))

    video_list = data.index.to_list()
    data["workflow"] = [video.split("_")[0] for video in video_list]

    workflow_list = []
    all_labels = []
    for workflow in list(set([video.split("_")[0] for video in video_list])):
        original_labels = list(data[data["workflow"] == workflow]["label"])
        all_labels.append(original_labels)

        predicted_workflow = aggregate_lab_actions(
            list(data[data["workflow"] == workflow]["label"])
        )
        workflow_list.append(predicted_workflow)

        assert len(original_labels) > len(predicted_workflow)
        for step in predicted_workflow:
            assert step in original_labels
        for i in range(len(predicted_workflow) - 1):
            assert predicted_workflow[i] != predicted_workflow[i + 1]
