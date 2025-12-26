import os
import json
import glob
import pickle
import numpy as np
from datetime import datetime
# from email.utils import parsedate_to_datetime
import torch
from torch_geometric.data import InMemoryDataset, Data

import sys

from torch_geometric.utils import add_self_loops
from torch_sparse import coalesce

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.BertManager import BertManager


class GCNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_process=False):
        self.pre_process_flag = pre_process
        if transform is None:
            self.transform = self._to_undirected
        super(GCNDataset, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'CED_Dataset')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'CED_Processed')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return f"CED_data_features.pt"

    @property
    def preprocess_dir(self):
        return os.path.join(self.root, 'CED_Preprocessed')

    @property
    def preprocessed_file_names(self):
        return f"CED_data_preprocessed.pkl"

    def preprocess(self):
        # preprocess the json files to generate a feature matrix for text data of the graph

        # initialize
        x = []
        y = []
        edge_index = []
        node_graph_id = []
        global_node_id_counter = 0
        graph_id_counter = 0

        # load bert model
        manager = BertManager()
        manager.load()

        for rumor in [True, False]:
            graph_filepath = os.path.join(self.raw_dir, f"{'rumor' if rumor else 'non-rumor'}-repost")
            blog_filepath = os.path.join(self.raw_dir, "original-microblog")
            filenames = glob.glob(os.path.join(graph_filepath, "*.json"))

            for filename in filenames:
                # load data
                with open(filename, 'r', encoding='utf-8') as f:
                    dt = json.load(f)
                root_filename = os.path.join(blog_filepath, os.path.basename(filename))
                with open(root_filename, 'r', encoding='utf-8') as f:
                    root_info = json.load(f)

                # parse root node information
                # time
                timestamp = root_info['time']
                try:
                    root_time = datetime.fromtimestamp(0)
                    if type(timestamp) is int:
                        root_time = datetime.fromtimestamp(timestamp)
                    elif type(timestamp) is str:
                        # root_time = parsedate_to_datetime(root_info['time']).strftime('%Y-%m-%d %H:%M:%S')
                        root_time = datetime.strptime(timestamp, "%a %b %d %H:%M:%S %z %Y")
                except ValueError as e:
                    print(f"Error at BLog {filename}")
                    raise e
                # add to features
                feature_root = np.array([root_time.year, root_time.month, root_time.day, root_time.hour, root_time.minute, root_time.second])

                # text
                input_text = manager.tokenizer(
                    root_info["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
                with torch.no_grad():
                    outputs = manager.model(input_text)
                    embedding = outputs.last_hidden_state[:, 0, :]
                    embedding = embedding.numpy().squeeze()

                # feature array
                feature_root = np.concatenate([feature_root, embedding])
                x.append(torch.from_numpy(feature_root).float())

                # counter
                root_id = global_node_id_counter
                global_node_id_counter += 1
                node_graph_id.append(graph_id_counter)

                # other nodes
                node_name_to_id_mapping = dict()
                for node in dt:
                    # time
                    timestamp = dt['date']
                    try:
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    except ValueError as e:
                        # about 6 nodes with illegal format of timestamps, ignore the nodes
                        continue
                    feature_node = [timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second]

                    # mapping
                    node_name_to_id_mapping[node['mid']] = global_node_id_counter

                    # text
                    input_text = manager.tokenizer(
                        node["text"],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="max_length"
                    )
                    with torch.no_grad():
                        outputs = manager.model(input_text)
                        embedding = outputs.last_hidden_state[:, 0, :]
                        embedding = embedding.numpy().squeeze()

                    feature_node = np.concatenate([feature_node, embedding])
                    x.append(torch.from_numpy(feature_node).float())

                    # counter
                    global_node_id_counter += 1
                    node_graph_id.append(graph_id_counter)

                # label
                y.append(1 if rumor else 0)

                # edge index
                for node in dt:
                    # skip the nodes with problem
                    try:
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    except ValueError as e:
                        continue

                    node_idx = node_name_to_id_mapping[node['mid']]
                    parent_mid = node['parent']

                    if parent_mid == '':
                        parent_idx = root_id
                    else:
                        parent_idx = node_name_to_id_mapping[parent_mid]

                    edge_index.append([parent_idx, node_idx])

                # graph id counter
                graph_id_counter += 1

        # stack tensors  **edge index needs transposition
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        node_graph_id = torch.tensor(node_graph_id, dtype=torch.long)

        # add self-loops
        edge_index, _ = add_self_loops(edge_index)
        # remove repeated edges
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        # save data
        preprocessed_data = {
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'node_graph_id': node_graph_id
        }

        filename = os.path.join(self.preprocess_dir, self.preprocess_filename)
        os.makedirs(os.path.dirname(self.preprocess_dir), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(preprocessed_data, f)

        return preprocessed_data

    def load_preprocessed_data(self):
        filename = os.path.join(self.preprocess_dir, self.preprocess_filename)
        if not os.path.exists(filename):
            raise FileNotFoundError("Preprocessed data not found")

        with open(filename, 'rb') as f:
            preprocessed_data = pickle.load(f)
        return preprocessed_data

    def process(self):
        # called each time a new dataset class is defined
        if self.pre_process_flag:
            dt = self.preprocess()
        else:
            dt = self.load_preprocessed_data()

        # extract preprocessed data
        x = dt['x']
        y = dt['y']
        edge_index = dt['edge_index']
        node_graph_id = dt['node_graph_id']

        # PyG Data
        data_pyg = Data(x=x, edge_index=edge_index, y=y)

        # split to single graphs
        data, slices = self._split_data(data_pyg, node_graph_id)

        # convert to data list for transformation
        data_list = []
        for i in range(len(slices['x']) - 1):
            # index range for a single graph
            x_slice = slices['x'][i:i+2]
            edge_slice = slices['edge_index'][i:i+2]

            # data object for a single graph
            single_graph = Data(
                x=data.x[x_slice[0]:x_slice[1]],
                edge_index=data.edge_index[edge_slice[0]:edge_slice[1]],
                y=data.y[i]
            )

            # transform
            single_graph = self.transform(single_graph)
            data_list.append(single_graph)

        # convert back to inmemory dataset
        data, slices = self.collate(data_list)

        # save processed data
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def _split_data(data, node_graph_id):
        # node numbers for each graph
        node_counts = torch.bincount(node_graph_id)
        # node slices
        node_slice = torch.cat([torch.tensor([0]), torch.cumsum(node_counts, dim=0)])

        # edge numbers of each graph
        row, _ = data.edge_index
        edge_counts = torch.bincount(row)
        # edge slices
        edge_slice = torch.cat([torch.tensor([0]), torch.cumsum(edge_counts, dim=0)])

        # change edge indices -- each graph starts from 0
        data.edge_index -= node_slice[node_graph_id[row]].unsqueeze(0)

        # slice dictionary
        slices = {'edge_index': edge_slice, 'x': node_slice}
        slices['y'] = torch.arrange(0, node_counts.size(0), dtype=torch.long)

        return data, slices

    @staticmethod
    def _to_undirected(data):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # create reversed edges
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # merge original edges and reversed ones
        new_edge_index = torch.stack([edge_index, rev_edge_index], dim=1)

        # remove repeated edges
        new_edge_index, _ = coalesce(new_edge_index, None, data.num_nodes, data.num_nodes)

        # update element
        data.edge_index = new_edge_index

        return data


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_path, 'data')
    dataset = GCNDataset(root=data_path, pre_process=True)

if __name__ == '__main__':
    main()
