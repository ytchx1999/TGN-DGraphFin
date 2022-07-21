import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    train_df = pd.read_csv('./data/{}/ml_{}_train.csv'.format(dataset_name, dataset_name))
    val_df = pd.read_csv('./data/{}/ml_{}_val.csv'.format(dataset_name, dataset_name))
    test_df = pd.read_csv('./data/{}/ml_{}_test.csv'.format(dataset_name, dataset_name))
    edge_type = np.load('./data/{}/ml_{}_edge.npy'.format(dataset_name, dataset_name))
    node_features = np.load('./data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    sources = graph_df.src.values
    destinations = graph_df.dst.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    sources = train_df.src.values
    destinations = train_df.dst.values
    edge_idxs = train_df.idx.values
    labels = train_df.label.values
    timestamps = train_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(train_df.src.values, train_df.dst.values, train_df.ts.values, train_df.idx.values, train_df.label.values)
    val_data = Data(val_df.src.values, val_df.dst.values, val_df.ts.values, val_df.idx.values, val_df.label.values)
    test_data = Data(test_df.src.values, test_df.dst.values, test_df.ts.values, test_df.idx.values, test_df.label.values)

    return full_data, node_features, edge_type, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_type = np.load('./data/{}/ml_{}_edge.npy'.format(dataset_name, dataset_name))
    node_features = np.load('./data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))

    sources = graph_df.src.values
    destinations = graph_df.dst.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # random.seed(2020)
    train_mask = (timestamps <= val_time)
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])
    
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return node_features, edge_type, full_data, train_data, val_data, test_data


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
