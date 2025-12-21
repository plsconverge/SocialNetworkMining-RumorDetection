import os
import json
import glob
from email.utils import parsedate_to_datetime

import networkx as nx
from datetime import datetime
from sklearn.model_selection import train_test_split

class CEDDataset:
    def __init__(self, filepath):
        self.rumor_dir = os.path.join(filepath, 'rumor-repost')
        self.non_rumor_dir = os.path.join(filepath, 'non-rumor-repost')
        self.origin_dir = os.path.join(filepath, 'original-microblog')

    def load_single(self, idx: int, rumor: bool):
        repost_dir = self.rumor_dir if rumor else self.non_rumor_dir
        json_files = glob.glob(os.path.join(repost_dir, '*.json'))
        with open(json_files[idx], 'r', encoding='utf-8') as json_file:
            nodelist = json.load(json_file)

        filename = os.path.basename(json_files[idx])
        rootpath = os.path.join(self.origin_dir, filename)
        with open(rootpath, 'r', encoding='utf-8') as json_file:
            root_info = json.load(json_file)
        root_id = f"{'Rumor' if rumor else 'NonRumor'}-{idx}"

        # build directed graph
        graph = nx.DiGraph()
        # root node
        graph.add_node(
            root_id,
            text=root_info['text'],
            date=datetime.fromtimestamp(root_info['time']).strftime('%Y-%m-%d %H:%M:%S'),
        )
        # repost nodes
        for repost in nodelist:
            graph.add_node(
                repost['mid'],
                text=repost['text'],
                date=repost['date']
            )
        # add edges
        for repost in nodelist:
            parent_id = repost['parent']
            if parent_id == '':
                parent_id = root_id
            graph.add_edge(parent_id, repost['mid'])

        dt = {
            'idx': idx,
            'label': 1 if rumor else 0,
            'root': root_info,
            'reposts': nodelist,
            'graph': graph
        }
        return dt

    def load_all(self):
        rumor_flg = [True, False]
        dt = []
        for rumor in rumor_flg:
            repost_dir = self.rumor_dir if rumor else self.non_rumor_dir
            json_files = glob.glob(os.path.join(repost_dir, '*.json'))

            for idx in range(len(json_files)):
                jsonfile = json_files[idx]
                with open(jsonfile, 'r', encoding='utf-8') as f:
                    nodelist = json.load(f)

                filename = os.path.basename(jsonfile)
                rootpath = os.path.join(self.origin_dir, filename)
                with open(rootpath, 'r', encoding='utf-8') as json_file:
                    root_info = json.load(json_file)
                root_id = f"{'Rumor' if rumor else 'NonRumor'}-{idx}"

                # build directed graph
                graph = nx.DiGraph()
                # root node
                try:
                    root_time = None
                    if type(root_info['time']) is int:
                        root_time = datetime.fromtimestamp(root_info['time']).strftime('%Y-%m-%d %H:%M:%S')
                    elif type(root_info['time']) is str:
                        root_time = parsedate_to_datetime(root_info['time']).strftime('%Y-%m-%d %H:%M:%S')
                except TypeError as e:
                    print(f"Error at BLog {filename}")
                    raise e
                graph.add_node(
                    root_id,
                    text=root_info['text'],
                    date=root_time,
                )
                # repost nodes
                for repost in nodelist:
                    graph.add_node(
                        repost['mid'],
                        text=repost['text'],
                        date=repost['date']
                    )
                # add edges
                for repost in nodelist:
                    parent_id = repost['parent']
                    if parent_id == '':
                        parent_id = root_id
                    graph.add_edge(parent_id, repost['mid'])

                dt_single = {
                    'idx': idx,
                    'label': 1 if rumor else 0,
                    'root': root_info,
                    'reposts': nodelist,
                    'graph': graph
                }
                dt.append(dt_single)

        return dt

    @staticmethod
    def split_dataset(dataset: list, test_size: float = 0.2, seed: int = 42):
        indices = [dt['idx'] for dt in dataset]
        labels = [dt['label'] for dt in dataset]

        train_ids, test_ids, y_train, y_test = train_test_split(
            indices, labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

        dt_dict = {dt['idx']: dt for dt in dataset}
        train_events = [dt_dict[idx] for idx in train_ids]
        test_events = [dt_dict[idx] for idx in test_ids]
        return train_events, test_events, y_train, y_test
