import argparse
from random import randrange

import numpy as np
from tqdm import tqdm

import evaluate
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# https://www.philschmid.de/fine-tune-flan-t5-peft


def get_dataset(dataset_str):
    dataset = load_dataset("text",
        data_files={
            'dev': [f'data/{dataset_str}/dev.txt'],
            'train': [f'data/{dataset_str}/train.txt'],
            'test': [f'data/{dataset_str}/test.txt']
        }
    )
    
    def preprocess_sample(sample):
        # Split line into a sample's input 'text' and labeled 'quads'
        sample['text'], sample['quads'] = sample['text'].split('####')
        sample['quads'] = eval(sample['quads'])
        # Construct paraphrased versions of the quads for the language model to use as the target.
        s2o = {
            'positive': 'great',
            'neutral':  'ok',
            'negative': 'bad'
        }
        paras = []
        for quad in sample['quads']:
            a, c, s, o = quad
            os = s2o[s]
            if a == 'NULL':
                a = 'it'
            paras.append(f'{c} is {os} because {a} is {o}')
        sample['quads_paraphrased'] = ' [SSEP] '.join(paras)
        return sample
        
    dataset = dataset.map(preprocess_sample)
    return dataset
    

def do_train(dataset_str, base_model_str):
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_str)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_str, device_map='auto')

    dataset = get_dataset(dataset_str)
    
    # Tokenize the train/test dataset to determine the max length necessary to pad to.
    # See http://huggingface.co/docs/transformers/pad_truncation

    train_test_dataset = concatenate_datasets([dataset['train'], dataset['test']])
    
    tokenized_inputs = train_test_dataset.map(lambda x: tokenizer(x['text']),
                                              batched=True,
                                              remove_columns=['text', 'quads', 'quads_paraphrased'])
    input_lengths = [len(x) for x in tokenized_inputs['input_ids']]
    max_source_length = max(input_lengths)

    tokenized_targets =  train_test_dataset.map(lambda x: tokenizer(x['quads_paraphrased']),
                                                batched=True,
                                                remove_columns=['text', 'quads', 'quads_paraphrased'])
    target_lengths = [len(x) for x in tokenized_targets['input_ids']]
    max_target_length = max(target_lengths)

    def pre_tokenization(sample):
        model_inputs = tokenizer(sample['text'], max_length=max_source_length, padding='max_length')
        labels = tokenizer(text_target=sample['quads_paraphrased'], max_length=max_target_length, padding='max_length')
        
        # Replace all tokenizer.pad_token_id in the labels with -100 to ignore padding in the loss.
        labels['input_ids'] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']]

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Tokenize the dataset for real this time with a function that takes care of padding.
    tokenized_dataset = dataset.map(pre_tokenization, batched=True, remove_columns=['text', 'quads', 'quads_paraphrased'])

    # Use DataCollator that will take care of padding our inputes and labels.
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           model=model,
                                           label_pad_token_id=-100,
                                           pad_to_multiple_of=8)
    
    output_dir=f'{base_model_str}-{dataset_str}-training-output'
    
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                             auto_find_batch_size=True,
                                             learning_rate=3e-4,
                                             num_train_epochs=20,
                                             optim='adamw_torch',  # Use PyTorch AdamW to avoid deprecation warning
                                             logging_dir=f'{output_dir}/logs',
                                             logging_strategy='steps',
                                             logging_steps=500,
                                             save_strategy='no')
    # Does trainer set model.train() mode?
    trainer = Seq2SeqTrainer(model=model,
                             args=training_args,
                             data_collator=data_collator,
                             train_dataset=tokenized_dataset['train'])
    
    # Disable cache while training..
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    
    # Save trained model & tokenizer
    model_str = f'{base_model_str}-{dataset_str}'
    trainer.model.save_pretrained(model_str)
    tokenizer.save_pretrained(model_str)
    # Save tokenized test dataset for use in evaluating the model
    tokenized_dataset['test'].save_to_disk(f'{model_str}/test_tokenized')


def do_inference(dataset_str, base_model_str):
    model_str = f'{base_model_str}-{dataset_str}'
    
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_str, device_map='auto')
    model.eval()
    
    dataset = get_dataset(dataset_str)
    
    # Run inference on a random sample
    sample = dataset['test'][randrange(len(dataset['test']))]
    input_ids = tokenizer(sample['text'], return_tensors='pt').input_ids.cuda()
    #outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=1.0)
    outputs = model.generate(input_ids=input_ids)
    print(f'input: {sample["text"]}')
    print(f"output:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")


def do_eval(dataset_str, base_model_str):
    model_str = f'{base_model_str}-{dataset_str}'
    
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_str, device_map='auto')
    model.eval()
    
    test_dataset = load_from_disk(f'{model_str}/test_tokenized').with_format('torch')
    
    predictions, references = [], []
    for batch in tqdm(test_dataset):
        outputs = model.generate(input_ids=batch['input_ids'].unsqueeze(0).cuda(), max_new_tokens=256)
        #outputs = model.generate(input_ids=batch['input_ids'].unsqueeze(0).cuda())
        preds = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        # decode eval sample
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)
        labels = tokenizer.decode(labels, skip_special_tokens=True)
        predictions.append(preds)
        references.append(labels)
        
    metric = evaluate.load("rouge")
    results = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"Rogue1: {results['rouge1'] * 100:2f}%")
    print(f"rouge2: {results['rouge2'] * 100:2f}%")
    print(f"rougeL: {results['rougeL'] * 100:2f}%")
    print(f"rougeLsum: {results['rougeLsum'] * 100:2f}%")

    # Translate back to quad
    
    def extract_quads(paras):
        quads = []
        paras = [p.strip() for p in paras.split('[SSEP]')]
        for p in paras:
            try:
                c_os, a_o = p.split(' because ')
                c, os = c_os.split(' is ')
                a, o = a_o.split(' is ')
                if a.lower() == 'it':
                    a = 'NULL'
            except:
                a, c, os, o = '', '', '', ''
            quads.append((a, c, os, o))
        return quads
                
    pred_quads, ref_quads = [], []
    for i in range(len(predictions)):
        pred_quad = extract_quads(predictions[i])
        ref_quad = extract_quads(references[i])
        pred_quads.append(pred_quad)
        ref_quads.append(ref_quad)
    
    def compute_f1_scores(pred_pt, gold_pt):
        """
        Function to compute F1 scores with pred and gold quads
        The input needs to be already processed
        """
        # number of true postive, gold standard, predictions
        n_tp, n_gold, n_pred = 0, 0, 0

        for i in range(len(pred_pt)):
            n_gold += len(gold_pt[i])
            n_pred += len(pred_pt[i])

            for t in pred_pt[i]:
                if t in gold_pt[i]:
                    n_tp += 1

        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        scores = {'precision': precision, 'recall': recall, 'f1': f1}
        return scores
    
    results = compute_f1_scores(pred_quads, ref_quads)
    print(f"F1:        {results['f1'] * 100:2f}%")
    print(f"Precision: {results['precision'] * 100:2f}%")
    print(f"Recall:    {results['recall'] * 100:2f}%")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='rest15', type=str, required=True, help='[rest15|rest16]')
    parser.add_argument('--base-model', default='t5-small', type=str, required=True, help='[t5-small|t5-base|t5-large]')
    
    parser.add_argument('--train', action='store_true', help='Run training')

    # Following arguments uses trained models
    parser.add_argument('--inf', action='store_true', help='Run inference on random sample')
    parser.add_argument('--eval', action='store_true', help='Evaluate model')

    args = parser.parse_args()
    
    if args.train:
        do_train(args.dataset, args.base_model)

    if args.inf:
        do_inference(args.dataset, args.base_model)

    if args.eval:
        do_eval(args.dataset, args.base_model)
