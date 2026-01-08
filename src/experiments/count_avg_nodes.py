import os
import json
import glob

# paths
ROOTPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATAPATH = os.path.join(ROOTPATH, r'data//CED_Dataset')
RUMORPATH = os.path.join(DATAPATH, 'rumor-repost')
NONRUMORPATH = os.path.join(DATAPATH, 'non-rumor-repost')


def count_nodes():
    for rumor in [True, False]:
        filepath = RUMORPATH if rumor else NONRUMORPATH
        count_helper(filepath, rumor)

def count_helper(path, rumor):
    json_files = glob.glob(os.path.join(path, '*.json'))

    repost_count = 0
    tree_count = 0
    min_repost = float('inf')
    max_repost = 0

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tree_count += 1
        repost_len = len(data)
        repost_count += repost_len

        min_repost = min(min_repost, repost_len)
        max_repost = max(max_repost, repost_len)

    print("==================================")
    print(f"Summary for {'Rumor' if rumor else 'Non-Rumor'} Dataset")
    print("==================================")
    print("Blog Count: ", tree_count)
    print("Repost Count: ", repost_count)
    print("Min Repost: ", min_repost)
    print("Max Repost: ", max_repost)
    print("Mean Repost: ", repost_count / tree_count)
    print()


def main():
    count_nodes()
    return

if __name__ == '__main__':
    main()
