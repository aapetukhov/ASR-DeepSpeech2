# Automatic Speech Recognition (ASR) with PyTorch

Ссылка на отчёт: https://wandb.ai/aapetukhov-new-economic-school/asr_project/reports/ASR-DeepSpeech-HW--Vmlldzo5NzE3ODgz

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a project on ASR with all necessary scripts provided for training and evaluating the model. It is worth noting that with better GPUs than a single P100 more extended training time would have been available, so higher results would have been achieved. You can also use a different language model, I use the pruned one because of the resources constraints.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Train

To train a model, log in to wandb and run the following commands:

1. First, train the model with
   
```bash
python train.py -cn=deepspeech2
```

2. Then, train it with
```bash
python train.py -cn=deepspeech2_360_augs_kaggle
```

3. Then, for clean,
```bash
python train.py -cn=ds2_finetune_strong_augs
```

Or, for other,
```bash
python train.py -cn=ds2_large_finetune
```

Where all configs are from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

# How To Evaluate

Download the pretrained models and clean lexicon from [here](https://drive.google.com/drive/u/1/folders/1oBV3LEffGjLxUKjYma7bxH1XPqcjiCdb) and locate them in your directory. You can also do this by running this commands in command line, but be aware that the files are large. **You can download only the clean model because it achieved the highest scores for my grade, but the other model might also make a hit.**

0. To download:

```bash
# install gdown
pip install gdown

# download best clean model
gdown 1XpAuRCg8phPTJxmzPyvrAUpc02ZgQC0O

# download the best other model
gdown 197CiNFeESxA6Mo6S5tv-hV8xF528WUrm

# download pretrained LM
gdown 1hqkXgR-OENH3uoILTInHCNKbmQFm-5wr

# download the lexicon for the LM, the default one is wrong
gdown 1HhqKQgOE4O-mnTbTm9s1JHMFZQTGpyyf
```

1. To run inference **LOCALLY**:

To run inference on clean (evaluate the model or save predictions):

```bash
python inference.py -cn=inference_clean_local
```

```bash
python inference.py -cn=inference_clean_local '+datasets.test.audio_dir=<YOUR_AUDIO_DIR>' '+datasets.test.transcription_dir=<YOUR_TRANSCRIPTION_DIR>'

To run inference on other:

```bash
python inference.py -cn=inference_other_local '+datasets.test.audio_dir=<YOUR_AUDIO_DIR>' '+datasets.test.transcription_dir=<YOUR_TRANSCRIPTION_DIR>'
```

2. To run inference **ON KAGGLE**:

To run inference on clean (evaluate the model or save predictions) in KAGGLE:

```bash
python inference.py -cn=inference_clean '+datasets.test.audio_dir=<YOUR_AUDIO_DIR>' '+datasets.test.transcription_dir=<YOUR_TRANSCRIPTION_DIR>'
```

To run inference on other:

```bash
python inference.py -cn=inference_other '+datasets.test.audio_dir=<YOUR_AUDIO_DIR>' '+datasets.test.transcription_dir=<YOUR_TRANSCRIPTION_DIR>'
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
