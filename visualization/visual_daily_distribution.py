import argparse
import os
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt

# paths
ROOTPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAPATH = os.path.join(ROOTPATH, r'data//CED_Dataset')
RUMORPATH = os.path.join(DATAPATH, 'rumor-repost')
NONRUMORPATH = os.path.join(DATAPATH, 'non-rumor-repost')


def plot_hourly_participation_heatmap(rumor=False):
    hourly_data = [0] * 24

    filepath = RUMORPATH if rumor else NONRUMORPATH
    json_files = glob.glob(os.path.join(filepath, '*.json'))
    if not json_files:
        print("Warning: No json files found, please check the path")
        return

    for idx, jsonfile in enumerate(json_files):
        try:
            with open(jsonfile, 'r', encoding='utf-8') as f:
                nodelist = json.load(f)

            for node in nodelist:
                try:
                    dt = datetime.strptime(node['date'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
                hourly_data[dt.hour] += 1
        except Exception as e:
            print(f"Error in json file: {jsonfile}")
            raise e

    total_num = sum(hourly_data)
    participation_percentage = [it / total_num * 100 for it in hourly_data]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        [f'{h:02d}:00' for h in range(24)],
        # hourly_data,
        participation_percentage,
        color='skyblue',
        edgecolor='k'
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(participation_percentage) * 0.01,
            f'{height:.2f}%',
            ha='center',
            va='bottom'
        )

    plt.title(f"User Participation 24-hour Heatmap ({'Rumor' if rumor else 'NonRumor'})")
    plt.xlabel("Time Period")
    plt.ylabel("Participation Percentage")
    plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rumor', action='store_true')

    args = parser.parse_args()
    plot_hourly_participation_heatmap(rumor=args.rumor)


if __name__ == '__main__':
    main()
