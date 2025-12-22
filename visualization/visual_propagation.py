import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
import glob
import random
import pandas as pd
import networkx as nx
import datetime
from collections import defaultdict, deque

ROOTPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAPATH = os.path.join(ROOTPATH, r'data//CED_Dataset')
RUMORPATH = os.path.join(DATAPATH, 'rumor-repost')
NONRUMORPATH = os.path.join(DATAPATH, 'non-rumor-repost')
ORIGINPATH = os.path.join(DATAPATH, 'original-microblog')


def compact_radial_layout(graph: nx.DiGraph, root_node_id: str):
    pos = dict()

    # depth
    depths = nx.single_source_shortest_path_length(graph, root_node_id)
    max_depth = max(depths.values())

    depth_groups = defaultdict(list)
    for node, depth in depths.items():
        depth_groups[depth].append(node)

    # set redius
    base_radii = dict()
    base_radius_step = 1.2
    for depth in range(max_depth + 1):
        if depth == 0:
            base_radii[depth] = 0
        else:
            base_radii[depth] = depth * base_radius_step

    # root node
    pos[root_node_id] = (0, 0)

    # depth-1 nodes -- first repost
    depth1_nodes = depth_groups[1]
    num_depth1 = len(depth1_nodes)
    radius1 = base_radii[1]

    # shuffle -- optional
    random.shuffle(depth1_nodes)

    for i, node in enumerate(depth1_nodes):
        angle = 2 * np.pi * i / num_depth1
        x = radius1 * np.cos(angle)
        y = radius1 * np.sin(angle)
        pos[node] = (x, y)

    # depth-2 nodes
    depth2_nodes = depth_groups[2]
    for node in depth2_nodes:
        predecessors = list(graph.predecessors(node))
        if predecessors:
            parent = predecessors[0]
            if parent in pos:
                parentx, parenty = pos[parent]
                depth2_siblings = [n for n in graph.successors(parent) if depths[n] == 2]
                sibling_index = depth2_siblings.index(node) if node in depth2_siblings else 0
                num_siblings = len(depth2_siblings)
                parent_angle = np.arctan2(parenty, parentx)

                if num_siblings == 1:
                    angle_offset = 0
                else:
                    max_angle = np.pi / 4
                    angle_offset = -max_angle / 2 + (max_angle / (num_siblings - 1)) * sibling_index
                angle = parent_angle + angle_offset
                extension = 0.6
                childx = parentx + extension * np.cos(angle)
                childy = parenty + extension * np.sin(angle)
                pos[node] = (childx, childy)

    # depth-3 nodes
    depth3_nodes = depth_groups[3]
    if depth3_nodes:
        for node in depth3_nodes:
            predecessors = list(graph.predecessors(node))
            if predecessors:
                parent = predecessors[0]
                if parent in pos:
                    parentx, parenty = pos[parent]

                    depth3_siblings = [n for n in graph.successors(parent) if depths[n] == 3]
                    sibling_index = depth3_siblings.index(node) if node in depth3_siblings else 0
                    num_siblings = len(depth3_siblings)
                    parent_angle = np.arctan2(parenty, parentx)

                    if num_siblings == 1:
                        radius3 = 0.3
                        childx = parentx + radius3 * np.cos(parent_angle)
                        childy = parenty + radius3 * np.sin(parent_angle)
                    else:
                        radius3 = 0.3
                        max_angle = np.pi / 6
                        angle_offset = -max_angle / 2 + (max_angle / (num_siblings - 1)) * sibling_index
                        angle = parent_angle + angle_offset
                        childx = parentx + radius3 * np.cos(angle)
                        childy = parenty + radius3 * np.sin(angle)
                    pos[node] = (childx, childy)

    # depth-4 nodes
    depth4_nodes = depth_groups[4]
    if depth4_nodes:
        for node in depth4_nodes:
            predecessors = list(graph.predecessors(node))
            if predecessors:
                parent = predecessors[0]
                if parent in pos:
                    parentx, parenty = pos[parent]

                    depth4_siblings = [n for n in graph.successors(parent) if depths[n] == 4]
                    sibling_index = depth4_siblings.index(node) if node in depth4_siblings else 0
                    num_siblings = len(depth4_siblings)
                    parent_angle = np.arctan2(parenty, parentx)

                    if num_siblings == 1:
                        radius4 = 0.1
                        childx = parentx + radius4 * np.cos(parent_angle)
                        childy = parenty + radius4 * np.sin(parent_angle)
                    else:
                        radius4 = 0.1
                        max_angle = np.pi / 8
                        angle_offset = -max_angle / 2 + (max_angle / (num_siblings - 1)) * sibling_index
                        angle = parent_angle + angle_offset
                        childx = parentx + radius4 * np.cos(angle)
                        childy = parenty + radius4 * np.sin(angle)
                    pos[node] = (childx, childy)

    # depth>4 nodes
    for depth in range(5, max_depth + 1):
        depth_nodes = depth_groups[depth]
        if depth_nodes:
            for node in depth_nodes:
                predecessors = list(graph.predecessors(node))
                if predecessors:
                    parent = predecessors[0]
                    if parent in pos:
                        parentx, parenty = pos[parent]
                        pos[node] = (parentx + 0.15, parenty + 0.05)

    return pos


def visualize_propagation_tree(idx=0, rumor=False, layout_type='hierarchy'):
    filepath = RUMORPATH if rumor else NONRUMORPATH
    json_files = glob.glob(os.path.join(filepath, '*.json'))
    if not json_files:
        print("Warning: No json files found, please check the path")
        return

    jsonfile = json_files[idx]
    filename = os.path.basename(jsonfile)
    rootfilepath = os.path.join(ORIGINPATH, filename)

    # root node
    if not os.path.exists(rootfilepath):
        print("Warning: No such file or directory: '{}'".format(rootfilepath))
    with open(rootfilepath, 'r', encoding='utf-8') as f:
        root_data = json.load(f)

    root_node_info = {
        'mid': f"ROOT_{root_data['time']}_{idx:04d}",
        'text': root_data['text'],
        'date': datetime.datetime.fromtimestamp(root_data['time']).strftime('%Y-%m-%d %H:%M:%S'),
        'isroot': True,
        'has_text': True,
        'kids': [],
        'parent': '',
        'is_virtual': False
    }

    # node list
    with open(jsonfile, 'r', encoding='utf-8') as f:
        nodelist = json.load(f)
    df = pd.DataFrame(nodelist)
    print(f"Node number: {len(df)}")

    df['has_text'] = df['text'].fillna('').str.strip().ne('')

    graph = nx.DiGraph()
    # root node
    graph.add_node(root_node_info['mid'], **root_node_info)

    node_attributes = {}
    for _, row in df.iterrows():
        mid = row['mid']
        attrs = {
            'uid': row['uid'],
            'date': row['date'],
            'parent': row['parent'],
            'text': row['text'],
            'has_text': row['has_text'],
            'is_virtual': False
        }

        graph.add_node(mid, **attrs)

    for _, row in df.iterrows():
        mid = row['mid']
        parent = row['parent']

        if parent == '':
            parent_id = root_node_info['mid']
        elif parent and parent in graph.nodes():
            parent_id = parent
        else:
            parent_id = root_node_info['mid']

        if parent_id in graph.nodes():
            graph.add_edge(parent_id, mid)

    # merge inner nodes
    merge_flag = True

    if merge_flag:
        merged_graph = graph.copy()

        depths = nx.single_source_shortest_path_length(graph, root_node_info['mid'])
        depth1_nodes = [node for node in graph.nodes() if depths[node] == 1]

        leaf_nodes_with_text = []
        leaf_nodes_without_text = []
        non_leaf_nodes = []

        for node in depth1_nodes:
            if len(list(graph.successors(node))) == 0:
                if graph.nodes[node]['has_text']:
                    leaf_nodes_with_text.append(node)
                else:
                    leaf_nodes_without_text.append(node)
            else:
                non_leaf_nodes.append(node)

        # merge
        group_size = 20

        # nodes with text
        text_virtual_nodes = []
        for i in range(0, len(leaf_nodes_with_text), group_size):
            group = leaf_nodes_with_text[i:i + group_size]
            if group:
                virtual_node_id = f"Virtual_Text_{i//group_size:04d}"
                text_virtual_nodes.append(virtual_node_id)

                merged_graph.add_node(
                    virtual_node_id,
                    is_virtual=True,
                    has_text=True,
                    is_root=False
                )
                merged_graph.add_edge(
                    root_node_info['mid'],
                    virtual_node_id
                )

                # delete initial leaf nodes
                for leaf in group:
                    if leaf in merged_graph:
                        merged_graph.remove_node(leaf)

        # nodes without text
        nontext_virtual_nodes = []
        for i in range(0, len(leaf_nodes_without_text), group_size):
            group = leaf_nodes_without_text[i:i + group_size]
            if group:
                virtual_node_id = f"Virtual_NonText_{i//group_size:04d}"
                nontext_virtual_nodes.append(virtual_node_id)

                merged_graph.add_node(
                    virtual_node_id,
                    is_virtual=True,
                    has_text=False,
                    is_root=False
                )
                merged_graph.add_edge(
                    root_node_info['mid'],
                    virtual_node_id
                )

                for leaf in group:
                    if leaf in merged_graph:
                        merged_graph.remove_node(leaf)

        # redirect pointer
        graph = merged_graph

    # visualization

    # layout_type
    if layout_type == 'hierarchy':
        pos = {}
        levels = defaultdict(list)
        levels[0].append(root_node_info['mid'])

        # BFS for level
        visited = {root_node_info['mid']}
        queue_mid = deque([(root_node_info['mid'], 0)])

        while queue_mid:
            current, level = queue_mid.popleft()
            for successor in graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    levels[level + 1].append(successor)
                    queue_mid.append((successor, level + 1))

        level_spacing = 2.0
        node_spacing = 1.5

        for level, nodes_in_level in levels.items():
            for i, node in enumerate(nodes_in_level):
                x = i * node_spacing - (len(nodes_in_level) - 1) * node_spacing / 2
                y = -level * level_spacing
                pos[node] = (x, y)

        for node in graph.nodes():
            if node not in pos:
                pos[node] = (0, 0)
                print(f"Warning: node {node['mid']} not in graph")

    elif layout_type == 'radial':
        # pos = {}
        #
        # distances = nx.single_source_shortest_path_length(graph, root_node_info['mid'])
        #
        # groups = defaultdict(list)
        # for node, dist in distances.items():
        #     groups[dist].append(node)
        #
        # max_dist = max(groups.keys()) if groups else 1
        # for dist, nodes in groups.items():
        #     radius = dist * 1.5
        #     num_nodes = len(nodes)
        #
        #     for i, node in enumerate(nodes):
        #         if dist == 0:
        #             pos[node] = (0, 0)
        #         else:
        #             angle = 2 * np.pi * i / num_nodes
        #             x = radius * np.cos(angle)
        #             y = radius * np.sin(angle)
        #             pos[node] = (x, y)
        pos = compact_radial_layout(graph, root_node_info['mid'])

    elif layout_type == 'circular':
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

    # classify nodes -- root node, normal repost node, plain repost node
    normal_nodes = []
    plain_nodes = []

    for node in graph.nodes():
        if node == root_node_info['mid']:
            continue

        node_data = graph.nodes[node]
        if node_data['has_text']:
            normal_nodes.append(node)
        else:
            plain_nodes.append(node)

    # plots
    fig, ax = plt.subplots(figsize=(8, 8))

    # edges
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color='#888888',
        arrows=True,
        arrowsize=5,
        arrowstyle='-|>',
        width=1.0,
        alpha=0.5,
        connectionstyle='arc3,rad=0.1'
    )

    # *** to implement ***
    # root node
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=[root_node_info['mid']],
        node_color='red' if rumor else 'green',
        node_size=120,
        edgecolors='black',
        linewidths=1
    )

    # normal node
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=normal_nodes,
        node_color='blue',
        node_size=35,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )

    # plain node
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=plain_nodes,
        node_color='#AAAAAA',
        node_size=25,
        alpha=0.5,
        edgecolors='black',
        linewidths=0.5
    )

    ax.set_title(f"Propagation diagram for {'Rumor' if rumor else 'NonRumor'} - {idx}")
    ax.axis('off')
    plt.tight_layout()

    legend_elements = [
        Patch(facecolor='red' if rumor else 'green', label='Original MicroBlog'),
        Patch(facecolor='blue', label='Reposts with Text'),
        Patch(facecolor='#AAAAAA', label='Reposts without Text')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()
    return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-idx', type=int, default=0)
    parser.add_argument('--rumor', action='store_true')
    parser.add_argument('-layout', type=str, default='radial', choices=['hierarchy', 'radial', 'circular', 'spring'])

    args = parser.parse_args()

    visualize_propagation_tree(
        idx=args.idx,
        rumor=args.rumor,
        layout_type=args.layout,
    )

if __name__ == '__main__':
    main()