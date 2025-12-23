import os
import json
import glob
from email.utils import parsedate_to_datetime
from typing import List, TypeAlias, Dict

import networkx as nx
from datetime import datetime
from sklearn.model_selection import train_test_split

# Type aliases for better readability of code
Path: TypeAlias = str

class CEDDataset:
    def __init__(self, filepath):
        self.rumor_dir = os.path.join(filepath, 'rumor-repost')
        self.non_rumor_dir = os.path.join(filepath, 'non-rumor-repost')
        self.origin_dir = os.path.join(filepath, 'original-microblog')
        self._json_files_cache = None

    def _get_json_files(self, rumor: bool) -> List[Path]:
        """
        Get list of JSON files of repost (with caching)
        
        :param self: Description
        :param rumor: Determine whether rumor or nonrumor files are chosen
        :type rumor: bool
        :return: A list of the path of json files
        :rtype: List[Path]
        """
        if self._json_files_cache is None:
            self._json_files_cache = {}
        cache_key = 'rumor' if rumor else 'non_rumor'
        if cache_key not in self._json_files_cache:
            repost_dir = self.rumor_dir if rumor else self.non_rumor_dir
            self._json_files_cache[cache_key] = glob.glob(os.path.join(repost_dir, '*.json'))
        return self._json_files_cache[cache_key]

    def load_single(self, idx: int, rumor: bool) -> Dict:
        """
        Load data related to a single blog, including its original information 
        and the recordings of repost
        
        :param self: Description
        :param idx: The index of the json file to load (the blog)
        :type idx: int
        :param rumor: Determine whether rumor or nonrumor files are chosen
        :type rumor: bool
        :return: A dictionary containing the info of a single blog and its repost
        :rtype: Dict
        """
        json_files = self._get_json_files(rumor)
        jsonfile = json_files[idx]
        with open(jsonfile, 'r', encoding='utf-8') as json_file:
            nodelist = json.load(json_file) # list of dictionaries

        filename = os.path.basename(jsonfile)
        # get the information from the original blog json file
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
            if parent_id == '': # empty parent means direct repost
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

    def load_all(self) -> List[Dict]:
        """
        Load data from all blogs
        
        :param self: Description
        :return: A list of dictionaries of the info of blogs and reposts
        :rtype: List[Dict]
        """
        rumor_flg = [True, False]
        dt = []
        for rumor in rumor_flg:
            json_files = self._get_json_files(rumor)
            for idx in range(len(json_files)):
                dt_single = self.load_single(idx, rumor)
                dt.append(dt_single)
        return dt

    @staticmethod
    def split_dataset(dataset: List[Dict], test_size: float = 0.2, seed: int = 42):
        """
        split the dataset into training set and test set
        
        :param dataset: List of data for each blog
        :type dataset: List[Dict]
        :param test_size: Ratio of the size of test set
        :type test_size: float
        :param seed: The seed for splitting
        :type seed: int
        """
        # Use positional indices to avoid collisions caused by reusing `idx`
        # across rumor / non-rumor subsets.
        indices = list(range(len(dataset)))
        labels = [dt["label"] for dt in dataset]

        train_ids, test_ids, y_train, y_test = train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

        dt_dict = {i: dt for i, dt in enumerate(dataset)}
        train_events = [dt_dict[i] for i in train_ids]
        test_events = [dt_dict[i] for i in test_ids]
        return train_events, test_events, y_train, y_test



if __name__ == "__main__":
    import sys
    datapath = os.path.abspath("data/CED_Dataset")
    print(f"Testing CEDDataset creation from: {datapath}")

    dataset = None
    try:
        loader = CEDDataset(datapath)
        dataset = loader.load_all()
    except Exception as e:
        print(f"Create dataset failed: {e}")
        sys.exit(1)

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First data example: idx = {dataset[0]['idx']}, label = {dataset[0]['label']}")
        print(f"root keys: {list(dataset[0]['root'].keys())}")
        print(f"repost size: {len(dataset[0]['reposts'])}")
        print(f"graph nodes: {dataset[0]['graph'].number_of_nodes()}, edges: {dataset[0]['graph'].number_of_edges()}")

    train_set, test_set, y_train, y_test = CEDDataset.split_dataset(dataset)
    print(f"Train set size: {len(train_set)}, test set size: {len(test_set)}")
    print(f"Train set label distribution: {dict((x, y_train.count(x)) for x in set(y_train))}")
    print(f"Test set label distribution: {dict((x, y_test.count(x)) for x in set(y_test))}")


