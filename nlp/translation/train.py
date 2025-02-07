import time
from typing import Callable, Dict

import evaluate
import numpy as np
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from transformers import DataCollatorForSeq2Seq, EvalPrediction, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

from nlp.translation.dataset import get_raw_datasets, split_datasets, get_tokenized_datasets
from nlp.translation.model import get_tokenizer, get_model, model_init
from nlp.translation.utils import get_base_config, load_yaml_config


def preprocess(tokenizer, base_config, token_max_length):
    raw_datasets = get_raw_datasets(base_config)
    splited_datasets = split_datasets(raw_datasets)
    tokenized_datasets = get_tokenized_datasets(tokenizer, splited_datasets, token_max_length, base_config)
    return tokenized_datasets

def fine_tune_model(smoke_test: bool = False) -> None:
    base_config = get_base_config()
    hp_space = get_hp_space()
    tokenizer = get_tokenizer(base_config['model'])
    metric = load_metric_evaluator(base_config)

    model = get_model(base_config['model'])

    tokenized_datasets = preprocess(tokenizer, base_config, hp_space['token_max_length'])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    compute_metrics = compute_metrics_fun(tokenizer, metric)


    training_args = Seq2SeqTrainingArguments(
        output_dir=base_config['model_checkpoint_dir'],
        torch_compile=True,
        eval_strategy=base_config['eval_strategy'],
        eval_steps=base_config['eval_steps'] if base_config['eval_strategy'] == "steps" else None,
        save_strategy=base_config['save_strategy'],
        save_steps=base_config['save_steps'],
        metric_for_best_model=base_config['metric_eval_type'],
        greater_is_better=True,
        save_total_limit=base_config['save_total_limit'],
        num_train_epochs=hp_space['num_train_epochs'],
        load_best_model_at_end=base_config['load_best_model_at_end'],
        predict_with_generate=base_config['predict_with_generate'],
        bf16=base_config['bf16'],
        dataloader_num_workers=base_config['dataloader_num_workers'] if base_config['dataloader_num_workers'] is not None else 0,
        dataloader_pin_memory = True,
    )

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Stops if no improvement in 5 evals
    )

    reporter = CLIReporter(
        metric_columns=["eval_bleu", "eval_loss", "epoch", "training_iteration"],
    )

    scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric="eval_bleu",
    mode = "max",
    max_t=100000, # maximum number of training iterations (or epochs) that a trial can run
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=lambda _: hp_space,
        resources_per_trial={"cpu": 4, "gpu": 1}, # dynamically get the number of cpus/gpus
        compute_objective=lambda metrics: metrics["eval_bleu"],
        n_trials=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        stop={"training_iteration": 1} if smoke_test else None,
        storage_path="~/ray_results/",
        name="tune_translation_asha",
        log_to_file=True,
    )

    print(f"Best hyperparameters: {best_run.hyperparameters}")


def compute_metrics_fun(tokenizer, metric) -> Callable[["EvalPrediction"], Dict]:
    def _compute_metrics(eval_preds: EvalPrediction) -> Dict:
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        print(f"Sample decoded_pred[0]: {decoded_preds[0:5]}")
        print(f"Sample decoded_label[0]: {decoded_labels[0:5]}")
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    return _compute_metrics

def evaluate_model(trainer, config):
    start_time = time.time()
    evaluate_result = trainer.evaluate(max_length=config['max_length'])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluating completed in {elapsed_time:.2f} seconds")
    print(evaluate_result)

"""
Set keep_in_memory=True to keep each sample in memory (by default it writes them on disk) to improve performance,
by reducing disk I/O overhead during evaluation.
"""
def load_metric_evaluator(config):
    metric = evaluate.load(path=config['metric_eval_path'], keep_in_memory=True)
    return metric

def get_hp_space():
    config_yaml = load_yaml_config("hparam_config.yaml")
    lr_low = float(config_yaml['learning_rate']['low'])
    lr_high = float(config_yaml['learning_rate']['high'])
    wd_low = float(config_yaml['optimizer']['adamw']['weight_decay']['low'])
    wd_high = float(config_yaml['optimizer']['adamw']['weight_decay']['high'])
    ae_low = float(config_yaml['optimizer']['adamw']['epsilon']['low'])
    ae_high = float(config_yaml['optimizer']['adamw']['epsilon']['high'])

    num_epochs = int(config_yaml['epochs']['value'])
    token_max_lengths = config_yaml['token_max_length']['values']
    map(int, token_max_lengths)

    train_batch_size_list = config_yaml['batch_size']['train']['values']
    map(int, train_batch_size_list)

    eval_batch_size_list = config_yaml['batch_size']['eval']['values']
    map(int, eval_batch_size_list)

    return {
        "learning_rate": tune.loguniform(lr_low, lr_high),
        "per_device_train_batch_size": tune.grid_search(train_batch_size_list),
        "per_device_eval_batch_size": tune.grid_search(eval_batch_size_list),
        "weight_decay": tune.loguniform(wd_low, wd_high),
        "adam_epsilon": tune.loguniform(ae_low, ae_high),
        "num_train_epochs": num_epochs,
        "token_max_length": token_max_lengths[0]
    }

if __name__ == "__main__":
    # model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
    # base_config = get_base_config()
    # tokenizer = get_tokenizer(model_checkpoint)
    # tokenized_datasets = preprocess(tokenizer, config['max_length'])
    # print(f'type: {type(tokenized_datasets)}, columns: {tokenized_datasets.column_names}, len: {len(tokenized_datasets["train"])}')

    ray.init(include_dashboard=False, num_cpus=4, num_gpus=1)
    fine_tune_model(False)