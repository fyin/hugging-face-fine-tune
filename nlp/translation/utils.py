import importlib.resources as pkg_resources
from typing import Dict, Any

import torch
import yaml

from nlp.translation import config


def get_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == "cuda"):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print(
            "NOTE: If you have a GPU, consider using it for training. Go to https://pytorch.org/get-started/locally/ for instructions.")
        print(
            "      On a Windows machine, Ex, run: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print(
            "      On a Mac machine, Ex, run: conda install pytorch::pytorch torchvision torchaudio -c pytorch")

    device = torch.device(device)
    return device

def load_yaml_config(filename: str) -> Dict[str, Any]:
    if not filename:
        raise ValueError("Filename cannot be None or empty.")

    with pkg_resources.files(config).joinpath(filename).open("r") as f:
        return yaml.safe_load(f)

def get_base_config() -> Dict[str, Any]:
    base_config = load_yaml_config("base_config.yaml")
    return {
        "datasets": base_config["datasets"],
        "model": base_config["model"],
        "src_lang": base_config["translation"]["src_lang"],
        "tgt_lang": base_config["translation"]["tgt_lang"],
        "metric_eval_path": base_config["evaluation"]["metric"]["path"],
        "metric_eval_type": base_config["evaluation"]["metric"]["type"],
        "eval_strategy": base_config["evaluation"]["strategy"],
        "eval_steps": base_config["evaluation"]["steps"],
        "early_stopping_patience": base_config["evaluation"]["early_stopping_patience"],
        "save_strategy": base_config["save"]["strategy"],
        "save_steps": base_config["save"]["steps"],
        "save_total_limit": base_config["save"]["total_checkpoint_limit"],
        "model_checkpoint_dir": base_config["save"]["model_checkpoint_dir"],
        "tensorboard_log_dir": base_config["save"]["tensorboard_log_dir"],
        "load_best_model_at_end": base_config["save"]["load_best_model_at_end"],
        "predict_with_generate": base_config["save"]["predict_with_generate"],
        "bf16": base_config["bf16"],
        "dataloader_num_workers": base_config["dataloader"]["num_workers"],
    }