cd "$(dirname $0)"

python3 ../train_supervised.py \
-d dgraphfin \
--n_degree 20 \
--bs 100 \
--n_epoch 30 \
--n_layer 2 \
--lr 3e-4 \
--prefix tgn-no-mem-uniform-ds \
--n_runs 10 \
--drop_out 0.1 \
--gpu 0 \
--node_dim 128 \
--time_dim 128 \
--message_dim 128 \
--memory_dim 128 \
--pos_weight 1. \
--uniform 

