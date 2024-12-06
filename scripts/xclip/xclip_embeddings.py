"""Extract laboratory video embeddings with xCLIP"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import logging
from pathlib import Path

import av
import click
import pandas as pd
import torch
from data import read_video_pyav, sample_frame_indices
from transformers import AutoModel, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_video_features(path_video: Path) -> torch.Tensor:
    """
    Extracts features from videos using xClip model.

    Args:
        path_video (Path): Paths to one video.

    Returns:
        torch.Tensor: A tensor containing the extracted features.
    """
    # Load model and video processor
    model_name = "microsoft/xclip-base-patch32"
    processor = AutoModel.from_pretrained(model_name)
    model = AutoProcessor.from_pretrained(model_name).to(device)

    # Get video features
    with av.open(path_video) as container:
        indices = sample_frame_indices(
            clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames
        )
        video = read_video_pyav(container, indices)
    inputs = processor(videos=[video], return_tensors="pt").to(device)
    video_features = model.get_video_features(**inputs)

    return video_features


@click.command()
@click.argument("video_folder", required=True, type=click.Path(exists=True))
@click.option("--labels_csv_file", required=True, type=click.Path(exists=True))
@click.argument("output_folder", required=True, type=click.Path(exists=True))
def main(video_folder: Path, labels_csv_file: Path, output_folder: Path) -> None:
    """
    Extracts features from videos using xClip model and saves them to a pickle file.

    Args:
        video_folder (Path): The folder containing the videos.
        labels_csv_file: Path to file containing action/motion label for each video clip.
        output_folder (Path): The folder where the extracted features will be saved.
    """
    logging.info("Loading data...")

    # Load data
    labels_df = pd.read_csv(labels_csv_file, index_col=0)

    embeddings = {}
    for split in ["training", "validation", "testing"]:
        embeddings[split] = []

    # Get embeddings and labels
    for sample in labels_df:
        video_name = sample["video"]
        video_split = sample["split"]
        embedding = get_video_features(video_folder / video_name)
        embeddings[video_split].append(embedding)

    # Save embeddings and labels
    for split in ["training", "validation", "testing"]:
        logging.info(f"Saving {split} embeddings...")
        torch.save(
            torch.vstack(embeddings[split]),
            f"{output_folder}/video_embeddings_{split}.pt",
        )

        # Save labels
        action_labels = labels_df[labels_df["split"] == split]
        logging.info(f"Saving {split} labels...")
        action_labels.to_csv(f"{output_folder}/video_labels_{split}.csv")

    logging.info(f"Embeddings and labels saved to {output_folder}.")

    return None


if __name__ == "__main__":
    main()
