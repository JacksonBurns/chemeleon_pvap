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

## Running `CheMeleon`

Set up a python envionment with `chemprop>=2.2.0`, `tensorboard`, and `ipykernel`.

Fitting code for `CheMeleon` is in `main.ipynb`.
