import csv
import pickle
import sys
from pathlib import Path

import pandas as pd
import shap
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def main(i_min, i_max, logits):
    i_min = int(i_min)
    i_max = int(i_max)
    logits = str(logits)

    logits_mapper = {
        "0": "probs",
        "1": "logits",
    }

    silent = False if sys.platform == "win32" else True
    tokenizer = AutoTokenizer.from_pretrained(
        "model", model_max_length=512, padding="max_length", truncation=True
    )
    model = AutoModelForSequenceClassification.from_pretrained("model")
    device = torch.device("cuda")

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_all_scores=True,
    )

    explainer_probs = shap.Explainer(
        shap.models.TransformersPipeline(classifier, rescale_to_logits=False),
        silent=silent,
    )
    explainer_logits = shap.Explainer(
        shap.models.TransformersPipeline(classifier, rescale_to_logits=True),
        silent=silent,
    )

    df = pd.read_csv("dataset.txt", sep="\t", quoting=csv.QUOTE_NONE)
    df = df[i_min:i_max]
    print(len(df))
    print(f"Running explainer with index {i_min} to {i_max - 1} and logits {logits}.")

    if logits == "0":
        sv = explainer_probs(df["text"])
    elif logits == "1":
        sv = explainer_logits(df["text"])
    else:
        raise ValueError("logits must be 0 or 1.")

    tag = f"{logits_mapper[logits]}_{i_min}_{i_max - 1}"

    sv_dir = Path("sv_results")

    with open(sv_dir / f"sv_{tag}.pkl", "wb") as f:
        pickle.dump(sv, f)

    sv_pos = sv[:, :, 1]
    with open(sv_dir / f"sv_pos_{tag}.pkl", "wb") as f:
        pickle.dump(sv_pos, f)

    ids = []
    datasets = []
    features = []
    values = []
    base_values = []

    for i in tqdm(range(len(df))):
        sv_instance = sv_pos[i]
        n = len(sv_instance.data)
        ids.extend([df["id"][i]] * n)
        datasets.extend([df["dataset"][i]] * n)
        features.extend(sv_instance.data)
        values.extend(sv_instance.values)
        base_values.extend([sv_instance.base_values] * n)

    pd.DataFrame(
        {
            "id": ids,
            "dataset": datasets,
            "feature": features,
            "value": values,
            "base_value": base_values,
        }
    ).to_csv(sv_dir / f"sv_pos_{tag}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(f"Usage: {sys.argv[0]} i_min i_max logits")

    if sys.argv[3] not in ["0", "1"]:
        raise SystemExit("logits must be 0 or 1.")

    main(sys.argv[1], sys.argv[2], sys.argv[3])
