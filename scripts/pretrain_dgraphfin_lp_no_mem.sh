cd "$(dirname $0)"

python3 ../train_self_supervised.py \
-d dgraphfin \
--n_degree 20 \
--bs 200 \
--n_epoch 25 \
--n_layer 2 \
--lr 0.0001 \
--prefix tgn-no-mem-uniform-ds \
--n_runs 1 \
--drop_out 0.1 \
--gpu 0 \
--node_dim 128 \
--time_dim 128 \
--message_dim 128 \
--memory_dim 128 \
--seed 0 \
--uniform

