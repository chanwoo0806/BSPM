# Blurring-Sharpening Process Models for Collaborative Filtering

This repository provides an *efficient and clarified* PyTorch implementation of the Blurring-Sharpening Processing Model (BSPM). The official implementation of BSPM can be found [here](https://github.com/jeongwhanchoi/BSPM). The paper can be found [here](https://dl.acm.org/doi/abs/10.1145/3539618.3591645).


## Setup
```bash
conda create -n bspm python=3.10
conda activate bspm
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
conda install torchdiffeq==0.2.2 numpy==1.22.4 scipy==1.9.3 pyyaml tqdm
```

## Run
```bash
# Execution with command-line arguments
python main.py --dataset gowalla --sharp_solv rk4 --sharp_step 1 --sharp_time 2.5 --ideal_cutoff 448 --ideal_weight 0.2

# Execution with a configuration file
python main.py --config bspm_gowalla
```