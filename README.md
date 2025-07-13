# chemeleon_pvap

Applying the [`CheMeleon`](https://github.com/JacksonBurns/CheMeleon) property prediction foundation model on the vapor pressure dataset collated in ["Improved vapor pressure predictions using group contribution-assisted graph convolutional neural networks (GC2NN)"](https://doi.org/10.5194/egusphere-2025-1191).

## Retrieving the Data

The data needed for this application is provided in the repo as [`kruger_pvap.csv`](./kruger_pvap.csv).

To fetch the data from the original study, run:
```bash
wget https://edmond.mpg.de/api/v1/access/datafile/265398?gbrecs=true -O preprocessed_data.zip
unzip preprocessed_data.zip
```

Next set up a Python environment with `pandas`, `torch`, `torch-geometric`, and `scikit-learn`.
Then run the below script to extract the data.

Other dataset: https://etsin.fairdata.fi/dataset/f17d76a4-99e8-4567-b4ca-9969db30d786/data

## Running `CheMeleon`

Set up a python envionment with `chemprop[hpopt]>=2.2.0`, `tensorboard`, and `ipykernel`.

Fitting code for `CheMeleon` is in `main.ipynb`.

The hyperparameters were determined by running `chemprop hpopt` on the first 5,000 molecules from the training data using this command:
```bash
chemprop hpopt --data-path kruger_pvap_5k.csv --output-dir optimization --smiles-columns smiles --from-foundation chemeleon --target-columns 'log_vp(pa)' --task-type regression --loss-function mse --split-sizes 0.8 0.199 0.001 --search-parameter-keywords ffn_num_layers ffn_hidden_dim batch_size --grad-clip 0.1 --raytune-num-samples 64
```
