import torch
from transformers import pipeline

from nlp.translation.utils import get_device

def inference_by_pretrained_model_pipeline(model_checkpoint, raw_input):
    device = torch.device(get_device())
    inference_pipeline = pipeline("translation", model=model_checkpoint, device=device)
    return inference_pipeline(raw_input)


if __name__ == "__main__":
    model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
    raw_input = "Upgrade software to the recent release."
    output = inference_by_pretrained_model_pipeline(model_checkpoint, raw_input)
    print(output)