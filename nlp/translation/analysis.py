import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
import numpy as np

from nlp.translation.dataset import get_raw_datasets
from nlp.translation.model import get_tokenizer, get_model
from nlp.translation.train import load_metric_evaluator, preprocess
from nlp.translation.utils import get_base_config, load_yaml_config

'''
Analyze the trained datasets to get the statistics of their tokenized length on source and target languages.
'''
def analyze_dataset():
    model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = get_tokenizer(model_checkpoint)
    vocab_size = tokenizer.vocab_size
    raw_datasets = get_raw_datasets(get_base_config())

    dataset_index = 300
    row = raw_datasets["train"]["translation"][dataset_index]
    print(row)

    total_count = len(raw_datasets["train"]["translation"])

    src_lang = "en"
    tgt_lang = "zh_CN"
    sources = [row[src_lang] for row in raw_datasets["train"]["translation"]]
    targets = [row[tgt_lang] for row in raw_datasets["train"]["translation"]]

    source_token = tokenizer(sources[dataset_index], add_special_tokens=True)
    print(f"source_token: {source_token}")
    decoded_source = tokenizer.decode(source_token["input_ids"])
    print(f"decoded source: {decoded_source}")

    token_lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in sources]
    median_length = int(np.median(token_lengths))
    max_length = int(np.max(token_lengths))
    mean_length = int(np.mean(token_lengths))
    standard_deviation = int(np.std(token_lengths))

    src_length_percentile_90 = int(np.percentile(token_lengths, 90))
    src_length_percentile_95 = int(np.percentile(token_lengths, 95))
    src_length_percentile_99 = int(np.percentile(token_lengths, 99))

    target_token = tokenizer(text_target=targets[dataset_index], add_special_tokens=True)
    print(f"target_token: {target_token}")
    decoded_target = tokenizer.decode(target_token["input_ids"])
    print(f"decoded target: {decoded_target}")

    target_lengths = [len(tokenizer(text_target=text, add_special_tokens=True)["input_ids"]) for text in targets]
    median_target_length = int(np.median(target_lengths))
    max_target_length = int(np.max(target_lengths))
    mean_target_length = int(np.mean(target_lengths))
    standard_deviation_target = int(np.std(target_lengths))

    tgt_length_percentile_90 = int(np.percentile(target_lengths, 90))
    tgt_length_percentile_95 = int(np.percentile(target_lengths, 95))
    tgt_length_percentile_99 = int(np.percentile(target_lengths, 99))

    return {
        "vocab_size": vocab_size,
        "total_dataset_count": total_count,
        "src_lang": {
            "median_length": median_length,
            "max_length": max_length,
            "mean_length": mean_length,
            "standard_deviation": standard_deviation,
            "src_length_percentile_90": src_length_percentile_90,
            "src_length_percentile_95": src_length_percentile_95,
            "src_length_percentile_99": src_length_percentile_99
        },
        "tgt_lang": {
            "median_length": median_target_length,
            "max_length": max_target_length,
            "mean_length": mean_target_length,
            "standard_deviation": standard_deviation_target,
            "tgt_length_percentile_90": tgt_length_percentile_90,
            "tgt_length_percentile_95": tgt_length_percentile_95,
            "tgt_length_percentile_99": tgt_length_percentile_99
        }
    }

'''
Evaluate pretrained model performance before fine tuning.
'''
def evaluate_pretrained_model():
    base_config = get_base_config()
    config_yaml = load_yaml_config("hparam_config.yaml")
    tokenizer = get_tokenizer(base_config['model'])
    model = get_model(base_config['model'])
    metric = load_metric_evaluator(base_config)
    token_max_lengths = config_yaml['token_max_length']['values']
    map(int, token_max_lengths)

    tokenized_datasets = preprocess(tokenizer, base_config, token_max_lengths[0])
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=64)

    model.eval()

    for batch in tqdm(eval_dataloader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        with torch.no_grad():
            pred_tokens = model.generate(input_ids,  attention_mask=batch["attention_mask"],  max_length=token_max_lengths[0])

        decoded_preds = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        print(f"Sample decoded_pred[0]: {decoded_preds[0:5]}")
        print(f"Sample decoded_label[0]: {decoded_labels[0:5]}")

        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute BLEU score
    result = metric.compute()
    return result

if __name__ == "__main__":
    stat_result = analyze_dataset()
    print(stat_result)
    # Statistics results:
    # {'vocab_size': 65001, 'total_dataset_count': 139666, 'src_lang': {'median_length': 5, 'max_length': 684, 'mean_length': 8, 'standard_deviation': 11, 'src_length_percentile_90': 15, 'src_length_percentile_95': 23}, 'tgt_lang': {'median_length': 8, 'max_length': 1171, 'mean_length': 11, 'standard_deviation': 14, 'tgt_length_percentile_90': 23, 'tgt_length_percentile_95': 30}}
    eval_result = evaluate_pretrained_model()
    print(f"BLEU Score: {eval_result['score']}")
    # BLEU Score: 27.479442109947506