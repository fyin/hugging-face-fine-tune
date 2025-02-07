from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from nlp.translation.utils import get_base_config


def get_tokenizer(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
    return tokenizer

def get_model(model_checkpoint):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) # No need for device_map, trainer will handle it
    return model

def model_init():
    base_config = get_base_config()
    model = get_model(base_config['model'])
    return model