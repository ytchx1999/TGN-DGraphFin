import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch
from tqdm import tqdm


def preprocess(data_name):
    OUT_DF = '../data/{}/ml_{}.csv'.format(data_name, data_name)
    OUT_ALL_TRAIN_DF = '../data/{}/ml_{}_all_train.csv'.format(data_name, data_name)
    OUT_TRAIN_DF = '../data/{}/ml_{}_train.csv'.format(data_name, data_name)
    OUT_VAL_DF = '../data/{}/ml_{}_val.csv'.format(data_name, data_name)
    OUT_TEST_DF = '../data/{}/ml_{}_test.csv'.format(data_name, data_name)

    OUT_EDGE_FEAT = '../data/{}/ml_{}_edge.npy'.format(data_name, data_name)
    OUT_NODE_FEAT = '../data/{}/ml_{}_node.npy'.format(data_name, data_name)

    names = [f'{data_name}.npz']
    items = [np.load(f'../data/{data_name}'+'/'+name) for name in names]
    
    x = items[0]['x']
    y = items[0]['y']  # .reshape(-1,1)
    edge_index = items[0]['edge_index']
    edge_type = items[0]['edge_type']
    ts = items[0]['edge_timestamp'].astype(np.float32)
    train_mask = items[0]['train_mask']
    valid_mask = items[0]['valid_mask']
    test_mask = items[0]['test_mask']

    # x = torch.tensor(x, dtype=torch.float).contiguous()
    # y = torch.tensor(y, dtype=torch.int64)
    # edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    # edge_type = torch.tensor(edge_type, dtype=torch.float)
    # train_mask = torch.tensor(train_mask, dtype=torch.int64)
    # valid_mask = torch.tensor(valid_mask, dtype=torch.int64)
    # test_mask = torch.tensor(test_mask, dtype=torch.int64)

    src, dst = edge_index[:, 0], edge_index[:, 1]
    idx = np.arange(src.shape[0])
    label_src = np.zeros(src.shape[0])
    label_dst = np.zeros(dst.shape[0])

    train_map_src = {}
    val_map_src = {}
    test_map_src = {}
    train_ts_src = {}
    val_ts_src = {}
    test_ts_src = {}

    train_map_dst = {}
    val_map_dst = {}
    test_map_dst = {}
    train_ts_dst = {}
    val_ts_dst = {}
    test_ts_dst = {}

    all_train_idx_src = []
    all_train_idx_dst = []

    for it, (s, d, t) in tqdm(enumerate(zip(src, dst, ts))):
        label_src[it] = y[s]
        label_dst[it] = y[d]
        # src node
        if s in train_mask:
            all_train_idx_src.append(it)
            if (s in train_map_src) and train_ts_src[s] >= t:
                pass
            else:
                train_map_src[s] = it
                train_ts_src[s] = t
        elif s in valid_mask:
            if (s in val_map_src) and val_ts_src[s] >= t:
                pass
            else:
                val_map_src[s] = it
                val_ts_src[s] = t
        elif s in test_mask:
            if (s in test_map_src) and test_ts_src[s] >= t:
                pass
            else:
                test_map_src[s] = it
                test_ts_src[s] = t
        # dst node
        if d in train_mask:
            all_train_idx_dst.append(it)
            if (d in train_map_dst) and train_ts_dst[d] >= t:
                pass
            else:
                train_map_dst[d] = it
                train_ts_dst[d] = t
        elif d in valid_mask:
            if (d in val_map_dst) and val_ts_dst[d] >= t:
                pass
            else:
                val_map_dst[d] = it
                val_ts_dst[d] = t
        elif d in test_mask:
            if (d in test_map_dst) and test_ts_dst[d] >= t:
                pass
            else:
                test_map_dst[d] = it
                test_ts_dst[d] = t
    
    for node in train_mask:
        if (node in train_map_src) and (node in train_map_dst):
            t_src, t_dst = train_ts_src[node], train_ts_dst[node]
            if t_src >= t_dst:
                train_map_dst.pop(node)
            else:
                train_map_src.pop(node)

    for node in valid_mask:
        if (node in val_map_src) and (node in val_map_dst):
            t_src, t_dst = val_ts_src[node], val_ts_dst[node]
            if t_src >= t_dst:
                val_map_dst.pop(node)
            else:
                val_map_src.pop(node)
    
    for node in test_mask:
        if (node in test_map_src) and (node in test_map_dst):
            t_src, t_dst = test_ts_src[node], test_ts_dst[node]
            if t_src >= t_dst:
                test_map_dst.pop(node)
            else:
                test_map_src.pop(node)
    
    train_idx_mask_src = list(train_map_src.values())
    val_idx_mask_src = list(val_map_src.values())
    test_idx_mask_src = list(test_map_src.values())

    train_idx_mask_dst = list(train_map_dst.values())
    val_idx_mask_dst = list(val_map_dst.values())
    test_idx_mask_dst = list(test_map_dst.values())
    
    train_src = np.concatenate([src[train_idx_mask_src], dst[train_idx_mask_dst]])
    train_dst =  np.concatenate([dst[train_idx_mask_src], src[train_idx_mask_dst]])
    train_ts =  np.concatenate([ts[train_idx_mask_src], ts[train_idx_mask_dst]])
    train_idx =  np.concatenate([idx[train_idx_mask_src], idx[train_idx_mask_dst]])
    train_label =  np.concatenate([label_src[train_idx_mask_src], label_dst[train_idx_mask_dst]])
    
    all_train_src = np.concatenate([src[all_train_idx_src], dst[all_train_idx_dst]])
    all_train_dst =  np.concatenate([dst[all_train_idx_src], src[all_train_idx_dst]])
    all_train_ts =  np.concatenate([ts[all_train_idx_src], ts[all_train_idx_dst]])
    all_train_idx =  np.concatenate([idx[all_train_idx_src], idx[all_train_idx_dst]])
    all_train_label =  np.concatenate([label_src[all_train_idx_src], label_dst[all_train_idx_dst]])

    val_src =  np.concatenate([src[val_idx_mask_src], dst[val_idx_mask_dst]])
    val_dst =  np.concatenate([dst[val_idx_mask_src], src[val_idx_mask_dst]])
    val_ts =  np.concatenate([ts[val_idx_mask_src], ts[val_idx_mask_dst]])
    val_idx =  np.concatenate([idx[val_idx_mask_src], idx[val_idx_mask_dst]])
    val_label =  np.concatenate([label_src[val_idx_mask_src], label_dst[val_idx_mask_dst]])

    test_src =  np.concatenate([src[test_idx_mask_src], dst[test_idx_mask_dst]])
    test_dst =  np.concatenate([dst[test_idx_mask_src], src[test_idx_mask_dst]])
    test_ts =  np.concatenate([ts[test_idx_mask_src], ts[test_idx_mask_dst]])
    test_idx =  np.concatenate([idx[test_idx_mask_src], ts[test_idx_mask_dst]])
    test_label =  np.concatenate([label_src[test_idx_mask_src], label_dst[test_idx_mask_dst]])

    print(f'train nodes: {train_src.shape[0]}, val nodes: {val_src.shape[0]}, test nodes: {test_src.shape[0]}')
    print(f'all train shape: {all_train_src.shape[0]}')

    df = pd.DataFrame({
        'src': src,
        'dst': dst,
        'ts': ts,
        'label': label_src,
        'idx': idx
    })
    df.sort_values("ts", inplace=True)

    all_train_df = pd.DataFrame({
        'src': all_train_src,
        'dst': all_train_dst,
        'ts': all_train_ts,
        'label': all_train_label,
        'idx': all_train_idx
    })
    all_train_df.sort_values("ts", inplace=True)

    train_df = pd.DataFrame({
        'src': train_src,
        'dst': train_dst,
        'ts': train_ts,
        'label': train_label,
        'idx': train_idx
    })
    train_df.sort_values("ts", inplace=True)

    val_df = pd.DataFrame({
        'src': val_src,
        'dst': val_dst,
        'ts': val_ts,
        'label': val_label,
        'idx': val_idx
    })
    val_df.sort_values("ts", inplace=True)

    test_df = pd.DataFrame({
        'src': test_src,
        'dst': test_dst,
        'ts': test_ts,
        'label': test_label,
        'idx': test_idx
    })
    test_df.sort_values("ts", inplace=True)

    df.to_csv(OUT_DF)
    all_train_df.to_csv(OUT_ALL_TRAIN_DF)
    train_df.to_csv(OUT_TRAIN_DF)
    val_df.to_csv(OUT_VAL_DF)
    test_df.to_csv(OUT_TEST_DF)

    np.save(OUT_EDGE_FEAT, edge_type)
    np.save(OUT_NODE_FEAT, x)


def run(data_name, bipartite=True):
    Path("../data/").mkdir(parents=True, exist_ok=True)
    preprocess(data_name)


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name',
                    default='dgraphfin')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)
