
# TGN for anomaly detection in DGraph-Fin

This repo is the code of [TGN](https://arxiv.org/pdf/2006.10637.pdf) model on [DGraph-Fin](https://dgraph.xinye.com/dataset) dataset. ([DGraph-Fin Leaderboard](https://dgraph.xinye.com/leaderboards/dgraphfin))

Performance on **DGraphFin** (10 runs):

| Methods   |  Test AUC  | Valid AUC  |
|  :----  | ---- | ---- |
| TGN-no-mem |  0.7741 ± 0.0003 | 0.7591 ± 0.0008 |

<!-- TGN-no-mem achieves top-3 performance on DGraphFin (top-2 performence without extra data) until August 22, 2022 ([https://dgraph.xinye.com/leaderboards/dgraphfin](https://dgraph.xinye.com/leaderboards/dgraphfin)).  -->


## 1. Setup 

### 1.1 Environment

- Dependencies: 
```{bash}
python==3.8
torch==1.8.2+cu102
pandas==1.4.1
sklearn==1.0.2
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

## 3. Future Work

- Dynamic sampling based on imbalanced class
- Combine TGN with anomaly detection methods
- Improve memory in a more efficient way

## 4. Args 

```bash
  -d, --data                                Dataset name
  --bs                                      Batch_size
  --prefix                                  Prefix to name the checkpoints
  --n_degree                                Number of neighbors to sample
  --n_head                                  Number of heads used in attention layer
  --n_epoch                                 Number of epochs
  --n_layer                                 Number of network layers
  --lr                                      Learning rate
  --patience                                Patience for early stopping
  --n_runs                                  Number of runs
  --drop_out                                Dropout probability
  --gpu                                     Idx for the gpu to use
  --node_dim                                Dimensions of the node embedding
  --time_dim                                Dimensions of the time embedding
  --backprop_every                          Every how many batches to backprop
  --use_memory                              Whether to augment the model with a node memory
  --embedding_module                        {graph_attention,graph_sum,identity,time} Type of embedding module
  --message_function                        {mlp,identity} Type of message function
  --memory_updater                          {gru,rnn} Type of memory updater
  --aggregator                              Type of message aggregator
  --memory_update_at_end                    Whether to update memory at the end or at the start of the batch
  --message_dim                             Dimensions of the messages
  --memory_dim                              Dimensions of the memory for each user
  --different_new_nodes                     Whether to use disjoint set of new nodes for train and val
  --uniform                                 take uniform sampling from temporal neighbors
  --randomize_features                      Whether to randomize node features
  --use_destination_embedding_in_message    Whether to use the embedding of the destination node as part of the message
  --use_source_embedding_in_message         Whether to use the embedding of the source node as part of the message
  --dyrep                                   Whether to run the dyrep model
  --seed                                    Seed for all
  --edge_dim                                Dimensions of the node embedding
  --no_norm                                 Whether to use LayerNorm in MergeLayer
```

## 5. Note
The implemention is based on [Temporal Graph Networks](https://github.com/twitter-research/tgn).


