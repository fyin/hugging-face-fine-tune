from typing import Union
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from nlp.translation.utils import get_base_config


def get_raw_datasets(base_config):
    if base_config is None:
        raise ValueError("Base config cannot be None")
    raw_datasets = load_dataset(base_config['datasets'], lang1=base_config['src_lang'], lang2=base_config['tgt_lang'], trust_remote_code=True)
    return raw_datasets

def split_datasets(raw_datasets: DatasetDict) -> DatasetDict:
    split_datasets = raw_datasets["train"].train_test_split(train_size=0.8)
    split_datasets["validation"] = split_datasets.pop("test")
    return split_datasets

def get_tokenized_datasets(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        split_datasets: DatasetDict,
        max_length: int,
        base_config: dict) -> DatasetDict:
    tokenized_datasets = split_datasets.map(
    lambda datasets: _preprocess_function(tokenizer, datasets, base_config, max_length),
    batched=True,
    remove_columns=split_datasets["train"].column_names)
    return tokenized_datasets

def _preprocess_function(
        tokenizer: Union[PreTrainedTokenizer,PreTrainedTokenizerFast],
        datasets: DatasetDict,
        base_config: dict,
        max_length: int):
    sources = [row[base_config['src_lang']] for row in datasets["translation"]]
    targets = [row[base_config['tgt_lang']] for row in datasets["translation"]]
    model_inputs = tokenizer(sources, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

if __name__ == "__main__":
    base_config = get_base_config()
    raw_datasets = get_raw_datasets(base_config)
    print(f'type: {type(raw_datasets)}, columns: {raw_datasets.column_names}, len: {len(raw_datasets["train"])}')

    splited_datasets = split_datasets(raw_datasets)
    print(f'type: {type(splited_datasets)}, columns: {splited_datasets.column_names}, len: {len(splited_datasets["train"])}')
    print(f'column names: {splited_datasets["train"].column_names}')