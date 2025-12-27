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
import argparse

from torch_geometric.utils import add_self_loops
from torch_sparse import coalesce

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.BertManager import BertManager


class GCNDataset(InMemoryDataset):
    def __init__(self, root, empty=False, transform=None, pre_process=False):
        self.pre_process_flag = pre_process
        if transform is None:
            self.transform = self._to_undirected
        self.processed = False
        super(GCNDataset, self).__init__(root, self.transform, None, None)

        if empty and not self.processed:
            # force to re-process
            self.process()
        else:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(root_path, "data", "models", "bert-base-chinese")
        manager = BertManager(model_dir)
        manager.load()
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        manager.model.to(device)
        manager.model.eval()

        from tqdm import tqdm

        for rumor in [True, False]:
            graph_filepath = os.path.join(self.raw_dir, f"{'rumor' if rumor else 'non-rumor'}-repost")
            blog_filepath = os.path.join(self.raw_dir, "original-microblog")
            filenames = glob.glob(os.path.join(graph_filepath, "*.json"))
            
            print(f"Processing {'rumor' if rumor else 'non-rumor'} data: {len(filenames)} graphs found.")

            for filename in tqdm(filenames, desc=f"Processing {'Rumor' if rumor else 'Non-Rumor'}"):
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
                        root_time = datetime.strptime(timestamp, "%a %b %d %H:%M:%S %z %Y")
                except ValueError as e:
                    print(f"Error at BLog {filename}")
                    raise e
                
                # prepare time features for root
                root_time_feat = [root_time.year, root_time.month, root_time.day, root_time.hour, root_time.minute, root_time.second]

                # collect all valid nodes and their texts
                valid_nodes = []
                all_texts = [root_info["text"]] # Start with root text
                all_time_feats = [root_time_feat] # Start with root time features
                
                # Filter valid nodes and collect data
                node_name_to_id_mapping = dict()
                
                # First pass: identify valid nodes
                # Root is implicitly valid and will get id global_node_id_counter
                root_id = global_node_id_counter
                
                # We will assign IDs after batch processing to ensure alignment
                # But we need to know which nodes are valid to build the batch
                
                temp_valid_nodes = [] # List of tuples (node_dict, time_feat)
                
                for node in dt:
                    timestamp = node['date']
                    try:
                        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    
                    time_feat = [ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second]
                    temp_valid_nodes.append((node, time_feat))
                    all_texts.append(node["text"])
                    all_time_feats.append(time_feat)

                # Batch process texts
                batch_size = 32 # Adjust based on GPU memory
                all_embeddings = []

                # pre-allocate embedding array
                num_texts = len(all_texts)
                embedding_dim = manager.model.config.hidden_size
                all_embeddings = np.zeros((num_texts, embedding_dim), dtype=np.float32)
                current_embedding_idx = 0
                
                for i in range(0, num_texts, batch_size):
                    batch_texts = all_texts[i:i+batch_size]
                    
                    input_text = manager.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128, # Consider reducing this if OOM or slow
                        padding="max_length"
                    )
                    
                    # Move inputs to device
                    input_text = {k: v.to(device) for k, v in input_text.items()}
                    
                    with torch.no_grad():
                        outputs = manager.model(**input_text)
                        # embeddings: [batch_size, hidden_size]
                        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        # all_embeddings.append(embeddings)

                        batch_len = len(batch_texts)
                        all_embeddings[current_embedding_idx:current_embedding_idx+batch_len] = embeddings
                        current_embedding_idx += batch_len
                
                # if all_embeddings:
                #     all_embeddings = np.concatenate(all_embeddings, axis=0)
                # else:
                #     # Should not happen as root is always there
                #     continue

                # Now construct the graph data

                graph_features = []
                graph_node_ids = []
                graph_edges = []
                
                # 1. Root Node
                feature_root = np.concatenate([np.array(all_time_feats[0]), all_embeddings[0]])
                # x.append(torch.from_numpy(feature_root).float())
                graph_features.append(feature_root)
                graph_node_ids.append(global_node_id_counter)
                
                global_node_id_counter += 1
                node_graph_id.append(graph_id_counter)
                
                # 2. Other Nodes
                # They correspond to all_embeddings[1:] and temp_valid_nodes
                for idx, (node, time_feat) in enumerate(temp_valid_nodes):
                    # idx 0 in temp_valid_nodes corresponds to embedding index 1
                    embedding = all_embeddings[idx + 1]
                    
                    feature_node = np.concatenate([np.array(time_feat), embedding])
                    # x.append(torch.from_numpy(feature_node).float())
                    graph_features.append(feature_node)
                    
                    # Mapping: current global counter
                    node_name_to_id_mapping[node['mid']] = global_node_id_counter
                    graph_node_ids.append(global_node_id_counter)
                    
                    global_node_id_counter += 1
                    node_graph_id.append(graph_id_counter)

                # label
                y.append(1 if rumor else 0)

                # edge index
                # We need to iterate over temp_valid_nodes again to build edges
                # because we need the full mapping to be ready
                
                for node, _ in temp_valid_nodes:
                    node_idx = node_name_to_id_mapping[node['mid']]
                    parent_mid = node['parent']

                    if parent_mid == '':
                        parent_idx = root_id
                    elif parent_mid in node_name_to_id_mapping:
                        parent_idx = node_name_to_id_mapping[parent_mid]
                    else:

                        continue
                        
                    # edge_index.append([parent_idx, node_idx])
                    graph_edges.append([parent_idx, node_idx])

                x.extend(graph_features)
                edge_index.extend(graph_edges)

                # graph id counter
                graph_id_counter += 1

        # stack tensors  **edge index needs transposition
        # x = np.array(x)
        # x = torch.from_numpy(x).float()
        x_tensor = torch.empty((len(x), len(x[0])), dtype=torch.float32)
        for i, feature in enumerate(x):
            x_tensor[i] = torch.tensor(feature, dtype=torch.float32)
        x = x_tensor

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

        filename = os.path.join(self.preprocess_dir, self.preprocessed_file_names)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(preprocessed_data, f)

        return preprocessed_data

    def load_preprocessed_data(self):
        filename = os.path.join(self.preprocess_dir, self.preprocessed_file_names)
        if not os.path.exists(filename):
            raise FileNotFoundError("Preprocessed data not found")

        with open(filename, 'rb') as f:
            preprocessed_data = pickle.load(f)
        return preprocessed_data

    def process(self):
        # called each time a new dataset class is defined
        self.processed = True
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
                edge_index=data.edge_index[:, edge_slice[0]:edge_slice[1]],
                y=data.y[i]
            )

            # transform
            single_graph = self.transform(single_graph)
            data_list.append(single_graph)

        # convert back to inmemory dataset
        self.data, self.slices = self.collate(data_list)

        # save processed data
        torch.save((self.data, self.slices), self.processed_paths[0])

    @staticmethod
    def _split_data(data, node_graph_id):
        # node numbers for each graph
        node_counts = torch.bincount(node_graph_id)
        # node slices
        node_slice = torch.cat([torch.tensor([0]), torch.cumsum(node_counts, dim=0)])

        # edge numbers of each graph
        row, _ = data.edge_index
        # edge_counts = torch.bincount(row)
        edge_counts = torch.bincount(node_graph_id[row])
        # edge slices
        edge_slice = torch.cat([torch.tensor([0]), torch.cumsum(edge_counts, dim=0)])

        # change edge indices -- each graph starts from 0
        # data.edge_index -= node_slice[node_graph_id[row]].unsqueeze(0)
        row_offset = node_graph_id[row].view(1, -1)
        node_slice_offset = node_slice[row_offset]
        data.edge_index -= node_slice_offset

        # slice dictionary
        slices = {'edge_index': edge_slice, 'x': node_slice}
        slices['y'] = torch.arange(0, node_counts.size(0), dtype=torch.long)

        return data, slices

    @staticmethod
    def _to_undirected(data):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # create reversed edges
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # merge original edges and reversed ones
        new_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)

        # remove repeated edges
        max_node_id = max(new_edge_index.size(0), new_edge_index.size(1))
        coalesce_num = max_node_id + 1
        new_edge_index, _ = coalesce(new_edge_index, None, coalesce_num, coalesce_num)

        # update element
        data.edge_index = new_edge_index

        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_process', action='store_true')
    parser.add_argument('--preprocess', action='store_true')

    args = parser.parse_args()
    process_flag = args.force_process
    preprocess_flag = args.preprocess

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_path, 'data')
    dataset = GCNDataset(root=data_path, empty=process_flag, pre_process=preprocess_flag)
    print("Preprocess Complete, files saved in {}".format(dataset.preprocess_dir))

if __name__ == '__main__':
    main()
