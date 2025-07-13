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
## Results

With `mordred`:

Input Data: kruger_confined_pvap

Raw Scores for 5-fold cross validation with random seed 42: 0.4439544163416308, 0.4534297114870276, 0.4699363803419394, 0.4349771268674359, 0.4935472583733895

Mean: 0.459
Standard Deviation: 2.072e-02
Minimum Value: 0.435
Maximum Value: 0.494
            

Input Data: kruger_pvap

Raw Scores for 5-fold cross validation with random seed 42: 0.7511134889752211, 0.7533391015887078, 0.8184298972068269, 0.7022272513394556, 0.7146345252231544

Mean: 0.748
Standard Deviation: 4.051e-02
Minimum Value: 0.702
Maximum Value: 0.818
            

Input Data: geckoq

Raw Scores for 5-fold cross validation with random seed 42: 0.6565943929270137, 0.6466161334721799, 0.6641487982953671, 0.671946224246149, 0.6756713190203265

Mean: 0.663
Standard Deviation: 1.050e-02
Minimum Value: 0.647
Maximum Value: 0.676
            
Without extra descriptors from `mordred`:

Input Data: kruger_confined_pvap

Raw Scores for 5-fold cross validation with random seed 42: 0.5146280238193128, 0.5446220330057916, 0.5628985285859646, 0.5221340897306196, 0.5364268174545426

Mean: 0.536
Standard Deviation: 1.701e-02
Minimum Value: 0.515
Maximum Value: 0.563
            

Input Data: kruger_pvap

Raw Scores for 5-fold cross validation with random seed 42: 0.7358986770907883, 0.7306242518761972, 0.7179460724916061, 0.7503600819910288, 0.7216483508621991

Mean: 0.731
Standard Deviation: 1.146e-02
Minimum Value: 0.718
Maximum Value: 0.750
            

Input Data: geckoq

Raw Scores for 5-fold cross validation with random seed 42: 0.6772633615467853, 0.6952168684232696, 0.7036234421631622, 0.6805339156767231, 0.6791406791479035

Mean: 0.687
Standard Deviation: 1.041e-02
Minimum Value: 0.677
Maximum Value: 0.704
            
