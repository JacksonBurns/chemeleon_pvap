import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from mordred import descriptors, Calculator
from rdkit.Chem import MolFromSmiles

from kruger_descriptors import broad_descriptors, confined_descriptors, geckoq_descriptors

warnings.filterwarnings("ignore", r".*DataFrame\.map.*", FutureWarning)

calc = Calculator(descriptors, ignore_3D=True)

seed = 42
seed_everything(seed)

mp_path = Path("chemeleon_mp.pt")
if not mp_path.exists():
    urlretrieve(
        r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        mp_path.name,
    )

for fname in (Path("./kruger_confined_pvap.csv"), Path("./kruger_pvap.csv"), Path("./geckoq.csv")):
    if "kruger" in fname.stem:
        # kruger datasets
        df = pd.read_csv(fname, index_col="smiles")
        _train_df = df[df["split"] == "train"]
        _test_df = df[df["split"] == "test"]
    else:
        # geckoq
        df = pd.read_csv(fname, index_col="SMILES")
        df = df.rename(columns={"pSat_Pa": "log_vp(pa)"})
        df.loc[:, "log_vp(pa)"] = np.log10(df["log_vp(pa)"])
        _train_df = df.sample(n=22_778 + 5_695, random_state=seed)
        _test_df = df[~df.index.isin(_train_df.index)]  # 3164
    # mordred
    # descs = calc.pandas(map(MolFromSmiles, df.index), nproc=64, nmols=len(df), quiet=False).fill_missing().to_numpy(dtype=float)
    # kruger descriptors
    if "confined" in fname.stem:
        descs = np.array(confined_descriptors(df.index), dtype=float)
    elif "geckoq" in fname.stem:
        descs = np.array(geckoq_descriptors(df.index), dtype=float)
    else:
        descs = np.array(broad_descriptors(df.index), dtype=float)
    
    descriptors_lookup = {df.index[i]: descs[i, :] for i in range(len(df))}
    n_descs = descs.shape[1]
    # or, no descriptors at all
    # descriptors_lookup = {df.index[i]: None for i in range(len(df))}
    # n_descs = 0

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = kfold.split(_train_df.index)

    maes = []
    for fold_num, split in enumerate(splits):
        train_df = _train_df.iloc[split[0]]
        val_df = _train_df.iloc[split[1]]
        test_df = _test_df.copy(deep=True)

        scaler = MinMaxScaler((0, 1))
        train_df.loc[:, "log_vp(pa)"] = scaler.fit_transform(train_df[["log_vp(pa)"]])
        val_df.loc[:, "log_vp(pa)"] = scaler.transform(val_df[["log_vp(pa)"]])
        test_df.loc[:, "log_vp(pa)"] = scaler.transform(test_df[["log_vp(pa)"]])

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        target_columns = ["log_vp(pa)"]
        train_data = [
            data.MoleculeDatapoint.from_smi(smi, y, x_d=descriptors_lookup[smi])
            for smi, y in zip(train_df.index, train_df[target_columns].to_numpy())
        ]
        val_data = [
            data.MoleculeDatapoint.from_smi(smi, y, x_d=descriptors_lookup[smi])
            for smi, y in zip(val_df.index, val_df[target_columns].to_numpy())
        ]
        test_data = [
            data.MoleculeDatapoint.from_smi(smi, y, x_d=descriptors_lookup[smi])
            for smi, y in zip(test_df.index, test_df[target_columns].to_numpy())
        ]
        train_dset = data.MoleculeDataset(train_data, featurizer)
        val_dset = data.MoleculeDataset(val_data, featurizer)
        test_dset = data.MoleculeDataset(test_data, featurizer)
        extra_features_scaler = train_dset.normalize_inputs(key="X_d")
        val_dset.normalize_inputs(scaler=extra_features_scaler)
        test_dset.normalize_inputs(scaler=extra_features_scaler)
        train_loader = data.build_dataloader(train_dset, batch_size=256)
        val_loader = data.build_dataloader(val_dset, shuffle=False)
        test_loader = data.build_dataloader(test_dset, shuffle=False)
        ffn = nn.RegressionFFN(
            input_dim=mp.output_dim + n_descs,
            hidden_dim=2_048,
            n_layers=2,
            output_transform=torch.nn.Sigmoid(),
        )
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)
        checkpointing = ModelCheckpoint(
            "checkpoints", "best-{epoch}-{val_loss:.2f}", "val_loss", mode="min", save_last=True
        )
        early_stopping = EarlyStopping("val_loss", patience=3)
        logger = TensorBoardLogger(save_dir="logs", default_hp_metric=False, name=None)
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=True,
            max_epochs=50,
            callbacks=[checkpointing, early_stopping],
            deterministic=True,
            gradient_clip_val=0.1,
        )
        trainer.fit(mpnn, train_loader, val_loader)
        mpnn = models.MPNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=mpnn)
        preds = torch.cat(trainer.predict(mpnn, test_loader), dim=0)
        mae = mean_absolute_error(
            scaler.inverse_transform(test_df[target_columns]).flatten(),
            scaler.inverse_transform(preds.detach().numpy()).flatten(),
        )
        print(f"Fold {fold_num} MAE: {mae:.3f}")
        maes.append(mae)

    with open(Path("./results_descriptors_kruger.log"), "a") as logfile:
        logfile.write(f"""
Input Data: {fname.stem}

Raw Scores for {len(maes)}-fold cross validation with random seed {seed}: {", ".join(map(str, maes))}

Mean: {np.mean(maes):.3f}
Standard Deviation: {np.std(maes):.3e}
Minimum Value: {np.min(maes):.3f}
Maximum Value: {np.max(maes):.3f}
            
""")
