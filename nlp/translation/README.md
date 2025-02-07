# hugging-face-fine-tune/nlp/translation
A NLP fine tuning practice using [Huggingface datasets and transformers library](https://huggingface.co) for language translation from English to Chinese. 
Fine tune the pretrained model [Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) on the dataset [Helsinki-NLP/kde4](https://huggingface.co/datasets/Helsinki-NLP/kde4) which is technical/documentation-based, focusing on software localization.
 
## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n hugging-face-fine-tune python=3.11`
* Activate the environment `conda activate hugging-face-fine-tune`
* Install the dependencies `conda install --yes --file requirements.txt`
* Use conda-forge to install the following packages not available in conda channel. 
  * `conda install -c conda-forge accelerate`
  * `conda install -c conda-forge ray-train`

## Modules
* In package `nlp.translation`: 
    * `dataset`: Load dataset, split the dataset to train and validation ones, and tokenize the datasets using the tokenizer from the pretrained model.
    * `analysis`: Analyze the trained datasets to get the statistics of their tokenized length on source and target texts. Evaluate pretrained model performance before fine tuning.   
    * `model`: Initiate objects of tokenizer and model using the pretrained model checkpoint.
    * `train`: Use Seq2SeqTrainer to fine tune the pretrained model and evaluate the model for the configured evaluation strategy.
    * [Ray tune](https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html) is integrated with the trainer to automatically tune hyperparameters with an objective of maximum of ScareBlue score.
    * `inference`: Perform inference test on the pretrained model ([Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)) using [Huggingface pipeline](https://huggingface.co/learn/nlp-course/chapter2/2).

## Analyze and Preprocess Data
* Run script directly, `python3 -m nlp.translation.analysis` to calculate the token length statistics of the source and target texts in the dataset [Helsinki-NLP/kde4](https://huggingface.co/datasets/Helsinki-NLP/kde4)
  * Token length statistics results
  
 |lang |Median | Max | Mean | Standard Deviation |90th percentile | 95th percentile |99th percentile |
 | --- | --- | --- | --- | --- |--- | --- |--- |
 | en | 5 | 684 | 8 | 11 | 15 | 23 |56 |
 | zh_CN | 8 | 1171 | 11 | 14 | 23 | 30 |65|

## Training
  * Run script directly, `python3 -m nlp.translation.train` 
  * All training and evaluation processes are logged to Tensorboard. Each model epoch checkpoint is saved for later inference.  
    * It was trained on mps device. After 3 epochs, sacrebleu score is about 39. Use the pretrained model without fine tuning, the sacrebleu is 27.

## Visualization
* Use Tensorboard to visualize the weights, weight gradients, loss and accuracy during training process.
  * Run `tensorboard --logdir [your logging directory]` to start Tensorboard 
  * Open `http://localhost:6006` in your browser

## Inference
* Run script directly, `python3 -m nlp.translation.inference`

## References
* https://huggingface.co/learn/nlp-course/chapter2/2
* https://huggingface.co/learn/nlp-course/chapter7/3
* https://arxiv.org/pdf/1803.09820
* https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one
* https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html
