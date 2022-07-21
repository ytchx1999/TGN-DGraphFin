cd "$(dirname $0)"

python3 ../train_self_supervised.py \
-d dgraphfin \
--n_degree 20 \
--bs 512 \
--n_epoch 20 \
--n_layer 2 \
--lr 0.001 \
--prefix tgn-no-mem \
--n_runs 1 \
--drop_out 0.1 \
--gpu 7 \
--node_dim 100 \
--time_dim 100 \
--message_dim 100 \
--memory_dim 100 

