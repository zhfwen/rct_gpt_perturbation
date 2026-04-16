import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def main(i_min, i_max, n_steps=30):
    i_min = int(i_min)
    i_max = int(i_max)
    n_steps = int(n_steps)
    print(f"Running integrated gradient with index {i_min} to {i_max - 1}.")

    tokenizer = AutoTokenizer.from_pretrained("model", model_max_length=512, padding='max_length', truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained("model")
    device = torch.device("cuda")

    classifier = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_all_scores=True,
    )

    baseline = tokenizer("", truncation=True, max_length=512, return_tensors="pt", padding="max_length").to(device)

    with torch.no_grad():
        baseline_logits = model(input_ids=baseline['input_ids'], attention_mask=baseline['attention_mask']).logits.cpu()

    print("Baseline logits:", baseline_logits[:, 1])

    embedding_layer = model.get_input_embeddings()
    baseline_embeds = embedding_layer(baseline['input_ids'])

    def forward_func(input_embeds, attention_mask):
        return model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits

    output = {
        "id": [],
        "dataset": [],
        "token": [],
        "logit": [],
        "baseline_logit": [],
    }

    df = pd.read_csv("dataset.txt", sep='\t', quoting=csv.QUOTE_NONE)
    df = df[i_min:i_max]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        inputs = tokenizer(row['text'], truncation=True, max_length=512, return_tensors="pt", padding="max_length").to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_embeds = embedding_layer(input_ids)

        ig = IntegratedGradients(forward_func)
        attributions, delta = ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=1,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        attributions = attributions.sum(dim=-1).squeeze(0).cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        output['id'].extend([row["id"]] * len(tokens))
        output['dataset'].extend([row["dataset"]] * len(tokens))
        output['token'].extend(tokens)
        output['logit'].extend(attributions)
        output['baseline_logit'].extend([float(baseline_logits[:, 1][0])] * len(tokens))

    pd.DataFrame(output).to_csv(Path("ig_results") / f"ig_pos_{i_min}_{i_max - 1}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        raise SystemExit(f'Usage: {sys.argv[0]} i_min i_max [n_steps=30]')

    main(*sys.argv[1:])
