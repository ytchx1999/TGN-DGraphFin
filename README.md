
# TGN-DGraphFin

This repo is the code of [TGN](https://arxiv.org/pdf/2006.10637.pdf) model on [Dgraph-Fin](https://dgraph.xinye.com/dataset) dataset.

<!-- # TGN: Temporal Graph Networks [[arXiv](https://arxiv.org/abs/2006.10637), [YouTube](https://www.youtube.com/watch?v=W1GvX2ZcUmY), [Blog Post](https://towardsdatascience.com/temporal-graph-networks-ab8f327f2efe)] 


#### Paper link: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) -->


## 1. Setup 

### 1.1 Environment

- Dependencies: 
```{bash}
python >= 3.7
torch == 1.8.2+cu102
pandas == 1.4.1
sklearn == 1.0.2
tqdm
...
```

- GPU: NVIDIA A100 (40GB)

- Params: 425,601

### 1.2 Dataset

The dataset [DGraph-Fin](https://dgraph.xinye.com/dataset) is a dynamic social network in finance, which can be download and placed in `./data/dgraphfin/dgraphfin.npz`.

## 2. Usage

### 2.1 Data Preprocessing

To run this code on the datasets, please first run the script to preprocess the data.

```bash
cd utils/
python3 preprocess_dgraphfin.py
```

This may costs 1.5h generating data for link prediction pretrain task and downstream node classification. More details can be found in `utils/preprocess_dgraphfin.py`.


### 2.2 Model Training and Inference

The `scripts/` folder contains training and inference scripts for models.

First, pretrain with link prediction to get node embeddings. Note that we don not use memory mechanism, because this may lead to CUDA out of memory in A100 (40GB) GPU. Therefore, we use this no memory version. 

```bash
cd scripts/
bash pretrain_dgraphfin_lp_no_mem.sh
```

Then, we use pretrained node embeddings and model for downstream binary node classification task in 10 runs.

```bash
cd scripts/
bash downstream_dgraphfin_nc_no_mem.sh
```

## 3. Results

Performance on **DGraphFin** (10 runs):

| Methods   | Train AUC  | Valid AUC  | Test AUC  |
|  :----  | ----  |  ---- | ---- |
| TGN-no-mem |  ±  |  ±  |  ±  |

## 4. Note
The implemention is based on [Temporal Graph Networks](https://github.com/twitter-research/tgn).


