import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime

ROOTPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAPATH = os.path.join(ROOTPATH, r'data//CED_Dataset')
RUMORPATH = os.path.join(DATAPATH, 'rumor-repost')
NONRUMORPATH = os.path.join(DATAPATH, 'non-rumor-repost')


def visual_timeinterval(rumor=False):
    filepath = RUMORPATH if rumor else NONRUMORPATH
    json_files = glob.glob(os.path.join(filepath, '*.json'))
    if not json_files:
        print("Warning: No json files found, please check the path")
        return

    interval_grids = [0, 5, 10, 15, 30, 60, 120, 180, 240, 300, 600, 1200, 1800, 3600, 7200, 10800, 18000, 36000, 3600*24]
    interval_count = [0 for _ in range(len(interval_grids))]

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            nodes = json.load(f)

        timestamps = []
        for node in nodes:
            try:
                dt = datetime.strptime(node['date'], '%Y-%m-%d %H:%M:%S')
                timestamps.append(dt)
            except ValueError:
                continue
        timestamps.sort()

        intervals = []
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if time_diff >= 0:
                intervals.append(time_diff)

        for interval in intervals:
            for k in range(len(interval_grids) - 1):
                if interval_grids[k] <= interval < interval_grids[k+1]:
                    interval_count[k] += 1
                    break
            if interval >= interval_grids[-1]:
                    interval_count[-1] += 1

    # compute proportion
    total_num = sum(interval_count)
    interval_percentage = [it / total_num * 100 for it in interval_count]

    # visualization
    xticks = [f'{interval_grids[k]} -\n{interval_grids[k+1]}s' for k in range(len(interval_grids) - 1)]
    xticks.append(f'>={interval_grids[-1]}')

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        xticks,
        interval_percentage,
        color='skyblue',
        edgecolor='k'
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(interval_percentage) * 0.01,
            f'{height:.2f}%',
            ha='center',
            va='bottom'
        )

    plt.xlabel('Time Interval')
    plt.ylabel('Percentage')
    plt.title(f"Distribution according to Time Intervals -- {'Rumor' if rumor else 'Non-Rumor'}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rumor', action='store_true')

    args = parser.parse_args()
    visual_timeinterval(rumor=args.rumor)


if __name__ == '__main__':
    main()
