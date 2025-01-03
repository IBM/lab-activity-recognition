# lab-activity-recognition

This repository hosts the Python code to run the end-to-end pipeline for activity recognition in scientific laboratories, as presented in the paper [Activity recognition in scientific experimentation using multimodal visual encoding](https://doi.org/10.1039/D4DD00287C). The code can be used to obtain embeddings of egocentric videos recorded in a laboratory setting leveraging pre-trained vision-language models, and then to train a classification head for activity recognition among a set of predefined labels.

Two video datasets (`lab-actions` and `lab-motion`) were used for code development and testing, and are available for download on [Zenodo](https://doi.org/10.5281/zenodo.14235875).
To extract frozen video embeddings for both dataset types, pretrained [xCLIP](https://arxiv.org/abs/2207.07285) and [Video-LLaVa](https://arxiv.org/abs/2311.10122) vision-language models were used. 
Each model requires its own installation and the corresponding pipelines need to be run in separate environments due to the custom installation required for Video-LLaVa. For detailed information refer to the [official repository](https://github.com/PKU-YuanGroup/Video-LLaVA).

## 1a. Extract xCLIP video embeddings

The whole pipeline can be run executing the script `run_xclip.sh`, see below a step-by-step guide.

### Install environment
Create a virtual environment and install xCLIP requirements.

```console
python -m venv .venv_xclip/
source .venv_xclip/bin/activate
pip install -r requirements.txt
```

### Extract embeddings
Once the xCLIP environment is installed, run the extraction of embeddings:

```console

(venv_xclip) $ python scripts/src/xclip_embeddings.py --video_folder your/path/to/videos --output_folder your/path/to/save/outputs
```
The previous command will save video embeddings as `.pt` files already split between training and testing sets. It will also save the corresponding labels to the video embeddings.
For training a classification head with the extracted xCLIP embeddings, refer to the `Train classification heads and save predictions` section below.

## 1b. Extract Video-LLaVa video embedding 

### Install environment
Follow the instructions for cloning the Video-LLaVa repository and proceed with installation as instructed by the `Requiremenents and Installation` section of the [Video-LLaVa code repository](https://github.com/PKU-YuanGroup/Video-LLaVA/tree/bcda27b7eae70613623a9714cf1942711a71ee08).

Make sure the code repository is at commit `bcda27b7eae70613623a9714cf1942711a71ee08` by running:
```console
git reset --hard bcda27b7eae70613623a9714cf1942711a71ee08
```

Then, activate the virtual environment created as part of the installation and run: 
```console
pip install sentence_transformers==2.2.2
```

Finally, download the pretrained Video-LlaVa-7B model from [the model repository](https://huggingface.co/LanguageBind/Video-LLaVA-7B/tree/e16778e47e589512d7e69f964850c8cad775a335) and save all downloaded model files to a local folder.

To run video captioning and embeddings, copy both `video_captioning.py` and `embed_captions.py` scripts of this repo to the cloned Video-LLaVa repository. Then, follow the steps below.

### Extract embeddings

Note that inference with Video-LLaVa requires GPU availability. 

#### Video captioning

Run Video-LLaVa inference with CUDA (GPU support) for video captioning, and save captions to CSV file using:
```console
python scripts/video-llava/video_captioning.py --video_folder your/path/to/videos --output_csv_file /path/to/new/csv/file
```

#### Embed captions

Extract embedding from text captions using the `Sentence Transformer` model and save embeddings and corresponding video labels to `pt` file:
```console
python scripts/video-llava/embed_captions.py --caption_csv_file /path/to/csv/captions --labels_csv_file /path/to/csv/input/labels --output_folder /path/to/output/folder
```

The embedding and label files obtained in the `output_folder` will be then used for classification heads training, testing and evaluation as instructed in the main README.md file of this repository.


## 2. Train classification heads and save predictions

To train XGBoost model with video embeddings extracted either with the xCLIP or Video-LLaVa pipelines, run:

```console
python scripts/train_classification_heads.py --video_embeddings_path /path/to/embeddings/folder --output_folder /path/to/prediction/output/folder --dataset_name dataset_name
```

Additionally, the `--dataset_name` argument can be used to select the type of dataset: `lab-actions (default)` or `lab-motion`.
The XGBoost model learning parameters can be modified by providing the following arguments:

- `n_estimators, default=100`
- `max_depth, default=8`
- `learning_rate, default=0.5`

The model prediction files saved to `/path/to/prediction/output/folder` will be used for for workflow level prediction evaluation as described below.

## 3. Workflow level prediction

Add to your environment the `natsort` and `Levenshtein` packages required for evaluation by running:
```console
pip install natsort
pip install Levenshtein
```

Then, evaluate workflow level predictions based on Levenshtein ratio scores by running:
```console
python scripts/evaluate_workflow.py --output_folder /path/to/prediction/output/folder --dataset_name dataset_name
```

The default dataset type used for evaluation is `lab-actions`. Set the `--dataset_name` to change the evaluation dataset to `lab-motion`. 
