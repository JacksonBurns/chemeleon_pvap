import pickle
from pathlib import Path

import pandas as pd


with open(Path("preprocessed_data/broad_data_training_info.pickle"), "rb") as file:
    data = pickle.load(file)

train_df = data[3]
train_df["split"] = "train"

test_df = data[4]
test_df["split"] = "test"

df = pd.concat((train_df, test_df))
df = df.drop(columns=["TARGET_log_norm_vp(pa)"])
df = df.rename(columns={"SMILES": "smiles"})

df.to_csv("kruger_pvap.csv", index=False)
