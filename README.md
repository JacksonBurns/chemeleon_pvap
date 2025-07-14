# chemeleon_pvap

Applying the [`CheMeleon`](https://github.com/JacksonBurns/CheMeleon) molecular property prediction foundation model on the vapor pressure dataset collated in ["Improved vapor pressure predictions using group contribution-assisted graph convolutional neural networks (GC2NN)"](https://doi.org/10.5194/egusphere-2025-1191).

## Results

Below are the average Mean Absolute Error values on randomly witheld testing data for the two models in the study (adGC2NN and fdGC2NN) and `CheMeleon` on the three datasets benchmarked by Kruger et al.
In the original study, the mean and standard deviation for 5-fold cross validation were provided for the Confined and Broad datasets.
I used these to run a Tukey Honestly Significant Difference test - practically better models are shown in bold.
Although I can't run a Tukey HSD for GeckoQ, I've bolded `CheMeleon` and adGC2NN since the distance between them is quite small and the standard deviation of `CheMeleon` was similar to its other benchmarks (leading me to believe that adGC2NN would also be similar here to its other results, and thus likely statistically the same).

| Model     | Confined | Broad | GeckoQ |
|-----------|----------|-------|--------|
| adGC2NN   | __0.39__     | __0.68__  | __0.67__   |
| fdGC2NN   | 0.51     | 0.80  | 0.74   |
| `CheMeleon` | 0.46     | __0.70__  | __0.66__   |

The indicated `CheMeleon` results use `mordred` descriptors, Kruger et al. descriptors, and no descriptors, respectively.

The instructions below cover how to use the code here to run your own models, or to reproduce the results above.
Doing the latter should take <1 hour using any fairly modern GPU - I completed all of the `CheMeleon` training on a single GTX 2080 Ti within that timeframe.

## Retrieving the Data

### Kruger et al.

The data needed for this demonstration is provided in the repo as [`kruger_pvap.csv`](./kruger_pvap.csv) and [`kruger_confined_pvap.csv](./kruger_confined_pvap.csv).
Note that this is derived from the original data which is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and is available [here](https://doi.org/10.17617/3.GIKHJL).

To fetch the data from the original study, run:
```bash
wget https://edmond.mpg.de/api/v1/access/datafile/265398?gbrecs=true -O preprocessed_data.zip
unzip preprocessed_data.zip
```

Next set up a Python environment with `pandas`, `torch`, `torch-geometric`, and `scikit-learn`.
Then run the `munge.py` script to extract the data.

### GeckoQ

The data needed for this demonstration is provided in the repo as [`geckoq.csv`](./geckoq.csv).
Note that this is derived from the original data which is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and is available [here](https://doi.org/10.23729/dd0396b3-9017-40f2-ae4b-6876bf33dd08).

No manipulations are necessary exepct renaming the `Dataframe.csv` file to `geckoq.csv` when downloading it from the original source.

## Running `CheMeleon`

Set up a python envionment with `chemprop[hpopt]>=2.2.0` and `tensorboard`.

The code for fitting `CheMeleon` to these datasets is in `main.py` which can be run with `python main.py`.
Around line 50 you will find the code for deciding which descriptor set to use (if using them at all) - comment out the appropriate lines to get the various results, or try adding your own sets.
The code for calculating the descriptors used in the reference study is, like the data, available under the CC BY 4.0 license and has been adapted for use here.

The hyperparameters you see set in that script were determined by running `chemprop hpopt` on the first 5,000 molecules from the training data (no testing labels used, so not data leak!) using this command:
```bash
chemprop hpopt --data-path kruger_pvap_5k.csv --output-dir optimization --smiles-columns smiles --from-foundation chemeleon --target-columns 'log_vp(pa)' --task-type regression --loss-function mse --split-sizes 0.8 0.199 0.001 --search-parameter-keywords ffn_num_layers ffn_hidden_dim batch_size --grad-clip 0.1 --raytune-num-samples 64
```
