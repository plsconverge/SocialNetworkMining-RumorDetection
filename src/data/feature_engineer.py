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

        # features -- out degree (树结构中max_in_degree基本都是1，没有区分度)
        out_degrees = [d for n, d in graph.out_degree()]

        # 处理空列表的情况
        features.append(np.max(out_degrees) if out_degrees else 0)  # max_out_degree
        features.append(np.mean(out_degrees) if out_degrees else 0)
        features.append(np.std(out_degrees) if out_degrees else 0)

        # 移除density特征：树结构的density基本固定，没有区分度
        # 移除user_notexist特征：全为0，没有区分度

        # features from root node information
        root_info = event['root']
        user_info = root_info['user']

        features.append(root_info.get('pics', 0))
        features.append(root_info.get('comments', 0))
        features.append(root_info.get('reposts', 0))
        features.append(root_info.get('likes', 0))

        timestamp = datetime.fromtimestamp(root_info['time'])
        features.append(timestamp.month)
        features.append(timestamp.day)
        features.append(timestamp.hour)

        user_notexist = type(user_info) is str and user_info == 'empty'
        if user_notexist:
            features.extend([0] * 6)
        else:
            features.append(1 if user_info.get('verified', False) else 0)
            features.append(1 if user_info.get('description', '') else 0)
            features.append(1 if user_info.get('gender') == 'm' else 0)
            features.append(user_info.get('messages', 0))
            features.append(user_info.get('followers', 0))
            features.append(user_info.get('friends', 0))

        # customized features

        return np.array(features, dtype=np.float32)
