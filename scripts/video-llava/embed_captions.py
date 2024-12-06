"""Embed captions generated with Video-LLaVa"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import logging
from pathlib import Path

import click
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

similarity_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


@click.command()
@click.option("--caption_csv_file", required=True, type=click.Path(exists=True))
@click.option("--labels_csv_file", required=True, type=click.Path(exists=True))
@click.option("--output_folder", required=True, type=click.Path(exists=True))
def main(caption_csv_file: Path, labels_csv_file: Path, output_folder: Path) -> None:
    """
    Embed video captions using Sentence Transformer model and save embeddings and labels.

    Args:
        caption_csv_file: Path to file containing video captions.
        labels_csv_file: Path to file containing action/motion label for each video clip.
        output_folder: Path to folder for saving embeddings and labels for each dataset split.
    """

    captions = pd.read_csv(caption_csv_file, header=None, index_col=0)
    labels = pd.read_csv(labels_csv_file, index_col=0)

    embeddings = {}
    for split in ["training", "validation", "testing"]:
        embeddings[split] = []

    # Preprocess videos and get embeddings
    logging.info("Embedding video captions with Sentence Transformer model...")
    for video in captions.index:
        caption = captions.loc[video, 1].split(".")[0]

        embedding = similarity_model.encode(caption, convert_to_tensor=True)

        split = labels.loc[video]["split"]
        embeddings[split].append(embedding)

    for split in ["training", "validation", "testing"]:
        # Save captions
        logging.info(f"Saving {split} embeddings...")
        torch.save(
            torch.vstack(embeddings[split]),
            f"{output_folder}/video_embeddings_{split}.pt",
        )

        # Save labels
        action_labels = labels[labels["split"] == split]
        logging.info(f"Saving {split} labels...")
        action_labels.to_csv(f"{output_folder}/video_labels_{split}.csv")

    logging.info(f"Embeddings and labels saved to {output_folder}.")

    return None


if __name__ == "__main__":
    main()
