import sys 
import locale
import pandas as pd
import random
import numpy as np
import torch
import re
import unicodedata
import typing as tp
import gc
from tqdm.auto import tqdm, trange
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from sacremoses import MosesPunctNormalizer
import sacrebleu
import os
import argparse
import shutil

# Function to set preferred encoding
def set_encoding():
    def gpe(x=None):
        return "UTF-8"

    locale.getpreferredencoding = gpe

# Function to read data from file paths into DataFrames
def read_data(paths):

    df_train = pd.read_excel(paths["train"])
    df_dev = pd.read_excel(paths["dev"])
    df_test = pd.read_excel(paths["test"])

    df_train.rename(columns={'MUNDARI': 'hne', 'HINDI': 'hin'}, inplace=True)
    df_dev.rename(columns={'MUNDARI': 'hne', 'HINDI': 'hin'}, inplace=True)
    df_test.rename(columns={'MUNDARI': 'hne', 'HINDI': 'hin'}, inplace=True)

    return df_train, df_dev, df_test

def create_model_save_path(base_save_path, index: int) -> str:
    """Create a unique directory path for saving the model and tokenizer."""
    save_dir = os.path.join(base_save_path, f"model_{index}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# Function to preprocess text
def preprocess_text(text):
    mpn = MosesPunctNormalizer(lang="hin")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean


def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


replace_nonprint = get_non_printing_char_replacer(" ")


# Function to initialize the model and tokenizer
def initialize_model_and_tokenizer(model_path):
    tokenizer = NllbTokenizer.from_pretrained(model_path, legacy_behaviour=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.cuda()
    return tokenizer, model


# Function to create optimizer
def create_optimizer(model):
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    return optimizer


# Function to cleanup GPU memory
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# Function to get batch pairs for training
def get_batch_pairs(batch_size, data):
    LANGS = [('hin', 'hin_Deva'), ('hne', 'hne_Deva')]
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data) - 1)]
        xx.append(preprocess_text(item[l1]))
        yy.append(preprocess_text(item[l2]))
    return xx, yy, long1, long2

def write_scores_to_file(filename, iteration, scores_dev):
    with open(filename, 'a') as f:  # Open the file in append mode
        f.write(f"{iteration} : Dev Scores = {scores_dev}\n")

# Function to train the model
def train_model(model, tokenizer, df_train, optimizer, batch_size=16, max_length=128, warmup_steps=1000,
                training_steps=200001, base_save_path='./models', early_stopping_steps=1000, top_n_models=10):
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    losses = []
    best_hne_score = 0
    no_improvement_steps = 0
    best_models = []

    model.train()
    x, y, loss = None, None, None
    cleanup()

    tq = trange(len(losses), training_steps)
    for i in tq:
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size, df_train)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to("cuda")
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to("cuda")
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

        if i % 500 == 0:
            print(i, np.mean(losses[-1000:]))

        if i % 500 == 0 and i > 0:
            # save_path = create_model_save_path(base_save_path, i)
            # model.save_pretrained(save_path)
            # tokenizer.save_pretrained(save_path)
            df_dev['hin_translated'] = batched_translate(df_dev.hne, src_lang='hne_Deva', tgt_lang='hin_Deva',model=model,tokenizer=tokenizer)
            df_dev['hne_translated'] = batched_translate(df_dev.hin, src_lang='hin_Deva', tgt_lang='hne_Deva',model=model,tokenizer=tokenizer)
            scores_dev = evaluate_translations(df_dev)
            # Write the scores to a file
            write_scores_to_file(base_save_path+'scores.txt', i, scores_dev)
            hne_score = sum(scores_dev[3:])/3
            if hne_score > best_hne_score:
                best_hne_score = hne_score
                no_improvement_steps = 0
                save_path = create_model_save_path(base_save_path, i)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved model and tokenizer to {save_path}")
                best_models.append(save_path)
                # Keep only the best top_n_models models
                if len(best_models) > top_n_models:
                    # Remove the oldest model if there are more than top_n_models
                    oldest_model = best_models.pop(0)
                    shutil.rmtree(oldest_model)
                    print(f"Removed model and tokenizer at {oldest_model}")
            else:
                no_improvement_steps += 500

            if no_improvement_steps >= early_stopping_steps:
                print(f"Stopping early due to no improvement for {early_stopping_steps} steps.")
                break

    save_path = create_model_save_path(base_save_path, i)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Final model and tokenizer saved to {save_path}")
    df_dev['hin_translated'] = batched_translate(df_dev.hne, src_lang='hne_Deva', tgt_lang='hin_Deva',model=model,tokenizer=tokenizer)
    df_dev['hne_translated'] = batched_translate(df_dev.hin, src_lang='hin_Deva', tgt_lang='hne_Deva',model=model,tokenizer=tokenizer)
    scores_dev = evaluate_translations(df_dev)
    write_scores_to_file(os.path.join(base_save_path, 'scores.txt'), i, scores_dev)
    return losses


# Function to translate text
def translate(text, src_lang='hne_Deva', tgt_lang='hin_Deva', a=32, b=3, max_input_length=1024, num_beams=4, model=None,
              tokenizer=None, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)


# Function to batch translate texts
def batched_translate(texts, batch_size=8, model=None, tokenizer=None, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in trange(0, len(texts2), batch_size):
        results.extend(translate(texts2[i: i + batch_size], model=model, tokenizer=tokenizer, **kwargs))
    return [p for i, p in sorted(zip(idxs, results))]


def evaluate_translations(df, num_samples=5):
    bleu_calc = sacrebleu.BLEU()
    chrf2_calc = sacrebleu.CHRF(word_order=2)
    chrf_calc = sacrebleu.CHRF(word_order=1)

    # Extract the lists of translated texts and references
    hin_translated_texts = df['hin_translated'].tolist()
    hin_references = df['hin'].tolist()
    hne_translated_texts = df['hne_translated'].tolist()
    hne_references = df['hne'].tolist()

    # Compute scores
    hin_bleu_score = bleu_calc.corpus_score(hin_translated_texts, [hin_references])
    hin_chrf_score = chrf_calc.corpus_score(hin_translated_texts, [hin_references])
    hin_chrf2_score = chrf2_calc.corpus_score(hin_translated_texts, [hin_references])
    hne_bleu_score = bleu_calc.corpus_score(hne_translated_texts, [hne_references])
    hne_chrf_score = chrf_calc.corpus_score(hne_translated_texts, [hne_references])
    hne_chrf2_score = chrf2_calc.corpus_score(hne_translated_texts, [hne_references])

    # Print scores
    print("\nMundari to Hindi Translations:")
    print("BLEU:", hin_bleu_score.score)
    print("ChrF (word order 1):", hin_chrf_score.score)
    print("ChrF++ (word order 2):", hin_chrf2_score.score)

    print("\nHindi to Mundari Translations:")
    print("BLEU:", hne_bleu_score.score)
    print("ChrF (word order 1):", hne_chrf_score.score)
    print("ChrF++ (word order 2):", hne_chrf2_score.score)

    # Print random samples
    print("\nRandom Samples:")
    indices = random.sample(range(len(df)), num_samples)
    for i in indices:
        print(f"\nSample {i + 1}:")
        print(f"Original Hindi: {df['hin'].iloc[i]}")
        print(f"Translated Hindi: {df['hin_translated'].iloc[i]}")
        print(f"Original Mundari: {df['hne'].iloc[i]}")
        print(f"Translated Mundari: {df['hne_translated'].iloc[i]}")
    
    # Return scores in a list
    return [hin_bleu_score.score, hin_chrf_score.score, hin_chrf2_score.score,
            hne_bleu_score.score, hne_chrf_score.score, hne_chrf2_score.score]

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Path to the Hindi training file.')
    parser.add_argument('--dev', type=str, required=True, help='Path to the Hindi development file.')
    parser.add_argument('--test', type=str, required=True, help='Path to the Hindi test file.')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--training_steps', type=int, default=200001, help='Total number of training steps')
    parser.add_argument('--base_save_path', type=str, default='./models/', help='Base path for saving models.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model directory.')
    parser.add_argument('--early_stopping_steps', type=int, default=1000, help='Number of steps for early stopping')
    parser.add_argument('--top_n_models', type=int, default=10, help='Number of best models to save')

    args = parser.parse_args()

    set_encoding()
    paths = {
        'train': args.train,
        'dev': args.dev,
        'test': args.test,
    }

    df_train, df_dev, df_test = read_data(paths)
    tokenizer, model = initialize_model_and_tokenizer(args.model_path)
    optimizer = create_optimizer(model)
    losses = train_model(model, tokenizer, df_train, optimizer, warmup_steps=args.warmup_steps, training_steps=args.training_steps, base_save_path=args.base_save_path,
                early_stopping_steps=args.early_stopping_steps, top_n_models=args.top_n_models)
    df_test['hin_translated'] = batched_translate(df_test.hne, src_lang='hne_Deva', tgt_lang='hin_Deva', model=model, tokenizer=tokenizer)
    df_test['hne_translated'] = batched_translate(df_test.hin, src_lang='hin_Deva', tgt_lang='hne_Deva', model=model, tokenizer=tokenizer)
    evaluate_translations(df_test)
    df_test.to_csv('df_test.csv', index=False)