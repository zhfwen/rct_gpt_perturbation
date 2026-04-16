import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import workers
from tqdm import tqdm


def main(start_index: int, num=1000):
    warnings.filterwarnings("ignore")

    df_results = pd.read_csv(Path("results") / "feature_attributions.csv")
    dfs_by_id = {key: value for key, value in df_results.groupby("id")}
    cols = ["ig_value", "sv_value"]

    end_index = min(start_index + num, len(dfs_by_id))
    out_dir = Path("results") / "pc_per_token" / f"{start_index}_{end_index - 1}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Indexes:")
    print(start_index, end_index - 1)
    ids = list(sorted(dfs_by_id.keys()))[start_index:end_index]

    # for each instance
    for id in tqdm(ids):
        # for each explainer
        for col in cols:
            pc_pos, tokens_pos, pc_neg, tokens_neg = workers.compute_aopc_separate(
                (id, dfs_by_id[id], col)
            )

            # pc values represent cumulative predictions with 0, 1, 2, ... tokens masked
            pd.DataFrame(
                {
                    "index": list(range(len(pc_pos))),
                    "token": tokens_pos,
                    "pc_value": pc_pos,
                }
            ).to_csv(out_dir / f"{id}_{col}_pos.csv", index=False)

            pd.DataFrame(
                {
                    "index": list(range(len(pc_neg))),
                    "token": tokens_neg,
                    "pc_value": pc_neg,
                }
            ).to_csv(out_dir / f"{id}_{col}_neg.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]))
