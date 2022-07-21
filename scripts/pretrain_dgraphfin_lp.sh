cd "$(dirname $0)"

python3 ../train_self_supervised.py \
-d dgraphfin \
--n_degree 10 \
--bs 2048 \
--n_epoch 10 \
--n_layer 1 \
--lr 0.001 \
--prefix tgn-attn \
--n_runs 1 \
--drop_out 0.1 \
--gpu 7 \
--node_dim 64 \
--time_dim 64 \
--message_dim 64 \
--memory_dim 64 \
--use_memory 

