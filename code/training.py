import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
import sys
import os
import math
import argparse
import datasets
import evaluate

bleu = evaluate.load('sacrebleu')
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)


def data_len(data, tokenizer):
    token = []
    for sent in data:
        # Tokenize a sentence
        text = sent
        tokens = tokenizer.encode(text)
        token.append(tokens)
    # get the max length
    length = len(max(token, key=len))

    # check if it too long (because of memory and time limitation)
    if length > 512:
        length = 512

    return length

def encoding(input_seq, target_seq, max_source_length, max_target_length, tokenizer, task_prefix):
    # encode the input
    input_encoding = tokenizer(
        [task_prefix + sequence for sequence in input_seq],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    # input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

    # encode the target
    target_encoding = tokenizer(
        [sequence for sequence in target_seq],
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )

    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return input_encoding, labels 

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.encodings.items()}
    item['labels'] = self.labels[idx]
    return item

  def __len__(self):
    return len(self.labels)


def main():
    data_path = './datasets/'
    output_dir = 'output'
    log_dir = 'logs/'
    model_checkpoint = 'google/flan-t5-base'
    model_name = 'flan-t5-base'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_dir", type=str, help="Output directory.", default=output_dir)
    parser.add_argument("--log_dir", type=str, help="Log directory.", default=log_dir)
    parser.add_argument("--data_path", type=str, help="Dataset path.", default=data_path)
    parser.add_argument("--model_checkpoint", type=str, help="Model Checkpoint.", default=model_checkpoint)
    parser.add_argument("--model_name", type=str, help="Model Name.", default=model_name)
    args = parser.parse_args()
    
    output_dir = args.output_dir
    log_dir = args.log_dir

    # Dataset
    data_path = args.data_path
    df_train = pd.read_csv(data_path+"train.tsv", sep="\t").astype(str)
    df_eval = pd.read_csv(data_path+"eval.tsv", sep="\t").astype(str)

    train_input_seq = df_train.input_text.values
    train_target_seq = df_train.target_text.values

    eval_input_seq = df_eval.input_text.values
    eval_target_seq = df_eval.target_text.values

    model_checkpoint = args.model_checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Get data length
    max_train_input_len = data_len(train_input_seq, tokenizer)
    max_train_target_len = data_len(train_target_seq, tokenizer)
    max_eval_input_len = data_len(eval_input_seq, tokenizer)
    max_eval_target_len = data_len(eval_target_seq, tokenizer)

    task_prefix = "generate response: "
    train_set, train_labels = encoding(train_input_seq, train_target_seq, max_train_input_len, max_train_target_len, tokenizer, task_prefix)
    eval_set,  eval_labels= encoding(eval_input_seq, eval_target_seq, max_eval_input_len, max_eval_target_len, tokenizer, task_prefix)

    train_dataset = Dataset(train_set, train_labels)
    eval_dataset = Dataset(eval_set, eval_labels)

    batch_size = 2
    model_name = args.model_name
    training_args = Seq2SeqTrainingArguments(
        f"{model_name}-{output_dir}",
        evaluation_strategy ="epoch",
        learning_rate=3e-4,
        optim='adamw_torch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed= 42,
        weight_decay=0.01,
        save_total_limit=1, #num of checkpoint you want to save
        save_strategy="epoch", # the checkpoint save strategy, default 'steps'
        num_train_epochs=10,
        predict_with_generate=True,
        push_to_hub=False,
        logging_dir=log_dir,            # directory for storing logs
        logging_strategy='epoch',
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # compute the scores
        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True)
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        
        result = {'bleu' : round(bleu_score["score"], 4),
                'meteor' : round(meteor_score["meteor"], 4),
                'rougeL' : round(rouge_score["rougeL"],4)}
        result["gen_len"] = np.mean(prediction_lens)
        return result


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    best_ckpt_path = trainer.state.best_model_checkpoint
    model_path = os.path.join(f'{model_name}-{output_dir}', f"t5_best_ckpt_{output_dir}.pth")
    torch.save(best_ckpt_path, model_path)

    trainer.save_model(output_dir=f'{model_name}-{output_dir}/best-model')

if __name__ == "__main__":
    main()