import numpy as np
import networkx as nx
from datetime import datetime

class FeatureEngineer:
    @staticmethod
    def extract_features(event: dict):
        # initialize
        reposts = event['reposts']
        features = list()

        # features from propagation tree
        features.append(len(reposts))   # feature -- size

        # feature -- max depth
        graph = event['graph']
        longest = nx.dag_longest_path_length(graph)
        features.append(longest)

        # features -- in & out degree
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]

        features.append(np.max(in_degrees))
        features.append(np.mean(out_degrees))
        features.append(np.std(out_degrees))

        # feature -- density
        features.append(nx.density(graph))

        # features from root node information
        root_info = event['root']
        user_info = root_info['user']
        user_notexist = type(user_info) is str and user_info == 'empty'
        features.append(1 if user_notexist else 0)

        features.append(root_info['pics'])
        features.append(root_info['comments'])
        features.append(root_info['reposts'])
        features.append(root_info['likes'])

        timestamp = datetime.fromtimestamp(root_info['time'])
        features.append(timestamp.month)
        features.append(timestamp.day)
        features.append(timestamp.hour)

        if user_notexist:
            features.extend([0] * 6)
        else:
            features.append(1 if user_info['verified'] else 0)
            features.append(1 if user_info['description'] else 0)
            features.append(1 if user_info['gender'] == 'm' else 0)
            features.append(user_info['messages'])
            features.append(user_info['followers'])
            features.append(user_info['friends'])

        # customized features

        return np.array(features, dtype=np.float32)
