import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime
from typing import Dict, List

class FeatureEngineer:
    @staticmethod
    def extract_features(event: Dict) -> np.ndarray:
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

    @staticmethod
    def extract_features_advanced(event: Dict) -> Dict:
        # initialize
        reposts = event['reposts']
        features = dict()

        # features from propagation tree
        features['size'] = len(reposts)

        # feature -- max depth
        graph = event['graph']
        longest = nx.dag_longest_path_length(graph)
        features['depth'] = longest

        # features -- out degree (树结构中max_in_degree基本都是1，没有区分度)
        out_degrees = [d for n, d in graph.out_degree()]

        # 处理空列表的情况
        features['max_out_degree'] = np.max(out_degrees)
        features['mean_out_degree'] = np.mean(out_degrees)
        features['std_out_degree'] = np.std(out_degrees)

        # 移除density特征：树结构的density基本固定，没有区分度
        # 移除user_notexist特征：全为0，没有区分度

        # features from root node information
        root_info = event['root']
        user_info = root_info['user']

        # pics: the number of pictures in the original blog
        features['pics'] = root_info['pics']
        features['comments'] = root_info['comments']
        features['reposts'] = root_info['reposts']
        features['likes'] = root_info['likes']

        timestamp = datetime.fromtimestamp(root_info['time'])
        features['month'] = timestamp.month
        features['day'] = timestamp.day
        features['hour'] = timestamp.hour

        user_notexist = type(user_info) is str and user_info == 'empty'
        if user_notexist:
            features['verified'] = 0
            features['description'] = 0
            features['gender'] = 0
            features['messages'] = 0
            features['followers'] = 0
            features['friends'] = 0
        else:
            features['verified'] = 1 if user_info.get('verified', False) else 0
            features['description'] = 1 if user_info.get('description', '') else 0
            features['gender'] = 1 if user_info.get('gender', '') else 0
            features['messages'] = user_info.get('messages', 0)
            features['followers'] = user_info.get('followers', 0)
            features['friends'] = user_info.get('friends', 0)

        # customized features

        # has-text feature & repost time period feature
        total_num = len(reposts)
        non_text_num = 0
        time_period_count = [0, 0]
        timestamp_list = []

        for repost in reposts:
            if repost['text'] == '':
                non_text_num += 1
            timestamp = repost['date']
            try:
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                timestamp_list.append(dt)
                if dt.hour == 0:
                    time_period_count[0] += 1
                elif dt.hour == 23:
                    time_period_count[1] += 1
            except ValueError:
                continue

        features['non_text_ratio'] = non_text_num / total_num
        features['00:00_repost'] = time_period_count[0] / total_num
        features['23:00_repost'] = time_period_count[1] / total_num

        # time interval feature
        freq_num = 0
        timestamp_list.sort()
        for i in range(1, len(timestamp_list)):
            time_diff = timestamp_list[i] - timestamp_list[i-1]
            if time_diff.total_seconds() < 15:
                freq_num += 1
        features['frequent_ratio'] = freq_num / len(timestamp_list)

        return features

    @staticmethod
    def convert_to_dataframe(feature_dicts: List, selected_keys: List = None):
        if selected_keys is None:
            selected_keys = list(feature_dicts[0].keys()) if feature_dicts else []

        df = pd.DataFrame(feature_dicts)[selected_keys]

        return df
