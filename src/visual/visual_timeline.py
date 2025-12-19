import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# paths
ROOTPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATAPATH = os.path.join(ROOTPATH, r'data//CED_Dataset')
RUMORPATH = os.path.join(DATAPATH, 'rumor-repost')
NONRUMORPATH = os.path.join(DATAPATH, 'non-rumor-repost')


def plot_timeline(rumor=False, idx=0, time_coverage=-1, show_suspicious=False):
    filepath = RUMORPATH if rumor else NONRUMORPATH
    json_files = glob.glob(os.path.join(filepath, '*.json'))
    if not json_files:
        print("Warning: No json files found, please check the path")
        return

    jsonfile = json_files[idx]
    with open(jsonfile, 'r', encoding='utf-8') as f:
        nodelist = json.load(f)

    # convert to dataframe
    df = pd.DataFrame(nodelist)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.sort_values('datetime')  # sort by time

    # cumulative participation element
    df['cumulative_participation'] = range(1, len(df) + 1)

    # print(df.head())
    print(f'Valid Nodes: {len(df)}')
    print(f"Timeline: {df['datetime'].iloc[-1]} - {df['datetime'].iloc[0]}")

    # save the core part
    if 0.0 < time_coverage < 1.0:
        cutoff_idx = int(np.ceil(len(df) * time_coverage)) - 1
        cutoff_idx = min(max(cutoff_idx, 0), len(df) - 1)
        cutoff_time = df.iloc[cutoff_idx]['datetime']

        df_core = df[df['datetime'] <= cutoff_time].copy()
    else:
        df_core = df.copy()

    # mark the nodes with many kids
    kids_threshold = 5
    df_core['is_suspicious'] = df_core['kids'].apply(len) >= kids_threshold
    suspicious_nodes = df_core[df_core['is_suspicious']]

    # visualization part

    time_span = df_core['datetime'].iloc[-1] - df_core['datetime'].iloc[0]
    total_seconds = time_span.total_seconds()
    total_days = time_span.days + total_seconds / (24.0 * 3600.0)

    # fig, ax = plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df_core['datetime'],
        df_core['cumulative_participation'],
        marker='.',
        linestyle='-',
        color='b',
        linewidth=2,
        markersize=5
    )

    # suspicious nodes
    if show_suspicious:
        ax.scatter(
            suspicious_nodes['datetime'],
            suspicious_nodes['cumulative_participation'],
            color='r',
            s=150,
            linewidth=1.5
        )

    plt.xlabel('Datetime')
    plt.ylabel('Cumulative Participation')
    plt.title(f'{"Rumor" if rumor else "Non-Rumor"} - {idx} Timeline')

    # modify time axis performance
    ax = plt.gca()

    if total_days <= 0.5:
        major_locator = mdates.HourLocator(interval=1)
        major_formatter = mdates.DateFormatter('%H:%M\n%m-%d')
        minor_locator = mdates.MinuteLocator(byminute=[0, 15, 30, 45])
    elif total_days <= 3:
        major_locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
        major_formatter = mdates.DateFormatter('%H\n%m-%d')
        minor_locator = mdates.HourLocator(byhour=2)
    elif total_days <= 7:
        major_locator = mdates.DayLocator(interval=1)
        major_formatter = mdates.DateFormatter('%m-%d\n%a')
        minor_locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
    elif total_days <= 30:
        major_locator = mdates.DayLocator(interval=2)
        major_formatter = mdates.DateFormatter('%m-%d')
        minor_locator = mdates.DayLocator(interval=1)
    else:
        major_locator = mdates.WeekdayLocator(byweekday=1, interval=2)
        major_formatter = mdates.DateFormatter('%m-%d\n%Y')
        minor_locator = mdates.DayLocator(interval=1)

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(minor_locator)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx', type=int, required=True, default=0)
    parser.add_argument('--rumor', action='store_true')
    parser.add_argument('-cover', type=float, default=-1)
    parser.add_argument('--show_suspicious', action='store_true')

    args = parser.parse_args()
    plot_timeline(
        rumor=args.rumor,
        idx=args.idx,
        time_coverage=args.cover,
        show_suspicious=args.show_suspicious
    )

if __name__ == '__main__':
    main()
