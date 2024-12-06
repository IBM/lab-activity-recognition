"""Extract video embeddings with Video-LLaVa"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import logging
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd
import torch
from llava.constants import DEFAULT_X_TOKEN, X_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_X_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@torch.no_grad()
def caption_video_with_llava(
    videos_path: Path, model_path: Path
) -> Dict[str, List[str]]:
    """
    Extracts video embeddings using Video-LlaVa.

    Args:
        videos_path: Path to folder containing videos
        model_path: Path to folder containing pretrained model files

    Returns:
        video_captions: caption for each video in videos_path
    """

    disable_torch_init()

    # Load model and setup metadata
    inp = "What action is being performed?"

    load_4bit, load_8bit = True, False
    model_name = "Video-LLaVA-7B"
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, load_8bit, load_4bit, device=device
    )
    video_processor = processor["video"]
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_X_TOKEN["VIDEO"] + "\n" + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_X_token(
            prompt, tokenizer, X_TOKEN_INDEX["VIDEO"], return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    video_captions = {}
    for video in videos_path.iterdir():
        video_tensor = video_processor(str(video), return_tensors="pt")["pixel_values"]

        if type(video_tensor) is list:
            tensor = [
                video.to(model.device, dtype=torch.float16) for video in video_tensor
            ]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[tensor, ["video"]],
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = (
            tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip("</s>").strip()
        )
        print(outputs)
        video_captions[video] = [outputs]

    return video_captions


@click.command()
@click.option("--video_folder", required=True, type=click.Path(exists=True))
@click.option("--model_folder", required=True, type=click.Path(exists=True))
@click.option("--output_csv_file", required=True, type=click.Path())
def main(video_folder: Path, model_folder: Path, output_csv_file: Path) -> None:
    """
    Generate video captions using Video-LLaVa model.

    Args:
        video_folder: Path to folder containing videos to be captioned
        model_folder: Path to folder containing pretrained model files
        output_csv_file (Path): The folder where the extracted features will be saved
    """

    # Preprocess videos and get embeddings
    logging.info("Extracting video captions with pretrained VideoLlaVa model...")
    captions = caption_video_with_llava(Path(video_folder), model_folder)
    logging.info("Captioning completed.")

    # Save captions
    captions = pd.DataFrame.from_dict(captions).T
    captions.to_csv(output_csv_file, header=False)

    logging.info("Captions saved to file.")

    return None


if __name__ == "__main__":
    main()
