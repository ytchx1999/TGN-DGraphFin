cd "$(dirname $0)"

python3 ../train_supervised.py \
-d dgraphfin \
--n_degree 10 \
--bs 1024 \
--n_epoch 20 \
--n_layer 1 \
--lr 0.0001 \
--prefix tgn-attn \
--n_runs 10 \
--drop_out 0.1 \
--gpu 0 \
--node_dim 100 \
--time_dim 100 \
--message_dim 100 \
--memory_dim 100 \
--use_memory 

