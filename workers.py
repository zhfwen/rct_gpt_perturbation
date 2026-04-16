import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def combine_shap_ig(args):
    df_sv, df_ig, id = args
    output = {
        'id': [],
        'dataset': [],
        'sv_value': [],
        'sv_token': [],
        'sv_base_value': [],
        'ig_value': [],
        'ig_token': [],
        'ig_base_value': [],
    }

    df_sv1 = df_sv[df_sv['id'] == id]
    df_ig1 = df_ig[df_ig['id'] == id]

    df_sv1 = df_sv1[1:]
    df_ig1 = df_ig1[~df_ig1['token'].isin(['[PAD]', '[CLS]', '[SEP]'])]

    df_sv1 = df_sv1[:len(df_ig1)]

    output['id'].extend(df_sv1['id'])
    output['dataset'].extend(df_sv1['dataset'])
    output['sv_value'].extend(df_sv1['value'])
    output['sv_token'].extend(df_sv1['feature'])
    output['sv_base_value'].extend(df_sv1['base_value'])
    output['ig_value'].extend(df_ig1['logit'])
    output['ig_token'].extend(df_ig1['token'])
    output['ig_base_value'].extend(df_ig1['baseline_logit'])

    return output


def shap_worker(args):
    file = args
    features = []
    values = []
    base_values = []
    ids = []
    datasets = []

    with open(Path('sv_results') / file, "rb") as f:
        sv = pickle.load(f)
    sv_pos = sv[:, :, 1]
    i_min = int(file.split('_')[2])
    i_max = int(file.split('_')[3].replace(".pkl", "")) + 1

    df = pd.read_csv("dataset.txt", sep='\t', quoting=csv.QUOTE_NONE)
    df = df[i_min:i_max]
    if len(df) != len(sv_pos):
        raise ValueError("length does not match")

    for i in tqdm(range(len(sv_pos))):
        sv_instance = sv_pos[i]
        features.extend(sv_instance.data)
        values.extend(sv_instance.values)
        ids.extend([df['id'][i]] * len(sv_instance.data))
        datasets.extend([df['dataset'][i]] * len(sv_instance.data))
        base_values.extend([sv_instance.base_values] * len(sv_instance.data))

    return features, values, base_values, ids, datasets


def add_gpt(args):
    id, df_i, ids = args

    output = {key: [] for key in list(df_i.columns) + ['gpt_index_value', 'gpt_index_word_value']}

    if id not in ids:
        df_i['gpt_index_value'] = np.nan
        df_i['gpt_index_word_value'] = np.nan
    else:
        values = {
            'gpt_index_value': [],
            'gpt_index_word_value': [],
        }

        for directory, column in [("gpt_index_masking", "gpt_index_value"), ("gpt_index_word_masking", "gpt_index_word_value")]:
            with open(Path(directory) / 'results' / f"{id}.json", 'r') as json_file:
                r = json.load(json_file)
                for index in range(len(df_i)):
                    try:
                        values[column].append(float(r[str(index)]))
                    except:
                        values[column].append(np.nan)

            df_i[column] = values[column]

    for col in df_i.columns:
        output[col].extend(list(df_i[col]))

    return output


def compute_aopc_separate(args):
    id, df, col = args

    df[col].fillna(0, inplace=True)

    df['index'] = list(range(len(df)))
    df['index'] = df['index'].astype(int)

    df_pos = df[df[col] > 0].sort_values(by=col, ascending=False).reset_index(drop=True)
    df_neg = df[df[col] < 0].sort_values(by=col, ascending=True).reset_index(drop=True)

    token_list = df["sv_token"].tolist()
    original_prob = single_mask_and_predict(token_list, [])

    masks_pos = [list(df_pos['index'])[:m + 1] for m in range(len(df_pos))]
    masks_neg = [list(df_neg['index'])[:m + 1] for m in range(len(df_neg))]

    outputs_pos, _, _ = mask_and_predict(token_list, masks_pos)
    outputs_neg, _, _ = mask_and_predict(token_list, masks_neg)

    pc_values_pos = [original_prob - prob for _, _, prob in outputs_pos]
    pc_values_neg = [prob - original_prob for _, _, prob in outputs_neg]

    return pc_values_pos, df_pos['sv_token'], pc_values_neg, df_neg['sv_token']


def compute_aopc_abs(args):
    id, df, col = args

    df[col].fillna(0, inplace=True)

    df['index'] = list(range(len(df)))
    df['index'] = df['index'].astype(int)

    df_sorted = df.copy()
    df_sorted[f"abs_{col}"] = df_sorted[col].abs()
    df_sorted = df_sorted.sort_values(by=f"abs_{col}", ascending=False).reset_index(drop=True)

    token_list = df["sv_token"].tolist()
    original_prob = single_mask_and_predict(token_list, [])

    masks = [list(df_sorted['index'])[:m + 1] for m in range(len(df_sorted))]
    outputs, _, _ = mask_and_predict(token_list, masks)

    pc_values = [np.abs(original_prob - prob) for _, _, prob in outputs]
    return np.mean(pc_values)


def compute_aopc(args):
    id, df, col = args

    df[col].fillna(0, inplace=True)

    df['index'] = list(range(len(df)))
    df['index'] = df['index'].astype(int)

    df_sorted = df.copy()
    df_sorted[f"abs_{col}"] = df_sorted[col].abs()
    df_sorted = df_sorted.sort_values(by=f"abs_{col}", ascending=False).reset_index(drop=True)

    token_list = df["sv_token"].tolist()
    original_prob = single_mask_and_predict(token_list, [])

    masks = [list(df_sorted['index'])[:m + 1] for m in range(len(df_sorted))]
    outputs, _, _ = mask_and_predict(token_list, masks)

    pc_values = [original_prob - prob for _, _, prob in outputs]
    return np.mean(pc_values)


def compute_aopc_logodds(args):
    id, df, col = args

    df[col].fillna(0, inplace=True)

    if len(df) == 0:
        return np.nan, np.nan

    df['index'] = list(range(len(df)))
    df['index'] = df['index'].astype(int)

    df_sorted = df.copy()
    df_sorted[f"abs_{col}"] = df_sorted[col].abs()
    df_sorted = df_sorted.sort_values(by=f"abs_{col}", ascending=False).reset_index(drop=True)

    token_list = df["sv_token"].tolist()
    original_prob = single_mask_and_predict(token_list, [])

    pc_values = []
    aopc_values = []
    log_odds_values = []

    masks = [list(df_sorted['index'])[:m + 1] for m in range(len(df))]
    outputs, _, _ = mask_and_predict(token_list, masks)

    for _, _, prob in outputs:
        pc_values.append(np.abs(original_prob - prob))
        aopc_values.append(np.mean(pc_values))
        log_odds_values.append(np.abs(np.log(prob / original_prob)))

    return aopc_values, log_odds_values


def single_mask_and_predict(list_of_tokens, masking_list):
    outputs, _, _ = mask_and_predict(list_of_tokens, [masking_list])
    return outputs[0][2]


def mask_and_predict(list_of_tokens, list_of_lists):
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("model", model_max_length=512, padding='max_length', truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained("model")

    classifier = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        padding='max_length',
        max_length=512,
    )

    def forward(input):
        model.eval()
        with torch.no_grad():
            tokenized_input = tokenizer(input, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
            input_ids = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            logits = logits.numpy()
            return logits, probs

    masked_texts = []
    masked_tokens = []
    raw_outputs = []
    for li in list_of_lists:
        tokens = copy.deepcopy(list_of_tokens)
        tokens = [str(t) for t in tokens]
        mt = []
        for i in li:
            if i < 0 or i >= len(tokens):
                print(f"Index {i} skipped.")
                continue
            mt.append(tokens[i])
            tokens[i] = '[MASK]'

        masked_tokens.append(mt)
        raw_outputs.append(forward("".join(tokens)))
        masked_texts.append("".join(tokens))

    outputs = []
    for ro in raw_outputs:
        outputs.append([float(ro[0][0][1]), float(ro[0][0][0]), float(ro[1][0][1])])

    return outputs, masked_tokens, masked_texts


def get_aopc_curve_values(file: str):
    spacing = np.linspace(0, 1, 100)

    df_aopc = pd.read_csv(file)
    df_aopc['step_percent'] = (df_aopc['index'] + 1) / len(df_aopc)

    try:
        interpolated = np.interp(spacing, df_aopc['step_percent'], df_aopc['pc_value'])
    except Exception as e:
        return None, file
    return interpolated, file


def save_force_plot(args):
    index, pmid = args

    sv_dir = Path('sv_results')
    for file in sv_dir.iterdir():
        file_name_split = file.name.split('_')
        si, ei = int(file_name_split[2]), int(file_name_split[3].replace('.pkl', ''))

        if si <= index <= ei:
            sv_index = index - si
            with open(file, 'rb') as f:
                sv_probs = pickle.load(f)

            output_path = Path("peer_review_congress") / "force_plots" / f"{pmid}.html"
            with open(output_path, 'w') as f:
                f.write(shap.plots.text(sv_probs[sv_index, :, 1], display=False))

            return pmid, shap.plots.text(sv_probs[sv_index, :, 1], display=False)

    return pmid, ""


def get_force_plot_html(index):
    sv_dir = Path('sv_results')
    for file in sv_dir.iterdir():
        file_name_split = file.name.split('_')
        si, ei = int(file_name_split[2]), int(file_name_split[3].replace('.pkl', ''))

        if si <= index <= ei:
            sv_index = index - si
            with open(file, 'rb') as f:
                sv_probs = pickle.load(f)

            return index, shap.plots.text(sv_probs[sv_index, :, 1], display=False)

    return index, ""
