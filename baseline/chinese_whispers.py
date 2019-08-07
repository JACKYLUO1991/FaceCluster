"""
Chinese Whispers Algorithm: Native version implementation
Cited paper: an Efficient Graph Clustering Algorithm \
    and its Application to Natural Language Processing Problems
http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
"""
from __future__ import print_function, division

import numpy as np
import shutil
import sys
import os
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt

import networkx as nx
import face_recognition
from imutils import paths

import seaborn as sns
sns.set()


def _face_distance(face_encodings, face_encoding_compare):
    """计算人脸编码嵌入向量之间的欧式距离"""
    if len(face_encodings) == 0:
        return np.empty(0)

    return 1. / np.linalg.norm(face_encodings - face_encoding_compare, axis=1)


def _chinese_whispers(encoding_list, *kwargs):
    """
    kwargs[0]: 人脸之间相似度阈值
    kwargs[1]: 算法迭代的次数
    """
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)
    # if len(encodings) <= 1:
    #     print("No enough encodings to cluster!")
    #     return []
    for idx, face_encoding_to_check in enumerate(encodings):
        node_id = idx + 1  # 编码从1开始
        # 初始化的时候将每张人脸作为无向图的一个节点
        node = (
            node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # 最后一个元素不需要边
        if (idx + 1) >= len(encodings):
            break

        compare_encodings = encodings[idx+1:]
        distances = _face_distance(compare_encodings, face_encoding_to_check)

        encoding_edges = []
        for i, distance in enumerate(distances):
            # 如果人脸之间的欧式距离大于阈值，两个人脸对应的节点添加边
            if distance > kwargs[0]:
                # 按相应的连接节点编码边
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))
        edges = edges + encoding_edges

    # 图搭建
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # 图的可视化
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    
    for _ in range(0, kwargs[1]):
        # 遍历程序, 给出图中所有的节点
        cluster_nodes = G.nodes()
        # Fixed big bug!!
        # shuffle(cluster_nodes)
        shuffle(list(cluster_nodes))

        for node in cluster_nodes:
            # 提取一个节点的邻居
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    # 判断该邻居的类别是否在其他邻居中存在，若存在则将相同类别的权重相加
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']
                                 ] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            edge_weight_sum = 0
            max_cluster = 0
            
            # TODO: 这块代码有点懵逼!!!
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster
            
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

    clusters = {}
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)
    # 从簇包含的样本大到小输出簇的值
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters


def cluster_face_encodings(face_encodings, threshold=0.5, iterations=20):
    if len(face_encodings) <= 1:
        print("Number of facial encodings must be greater than one, can't cluster")
        return []
    return _chinese_whispers(face_encodings.items(), threshold, iterations)


def main(output_dir):
    face_encodings = {}

    for image_path in tqdm(list(paths.list_images("../dataset"))):
        # print(image_path)
        rgb = face_recognition.load_image_file(image_path)
        boxes = face_recognition.face_locations(
            rgb, model='hog')  # 'cnn' or 'hog'
        encodings = face_recognition.face_encodings(rgb, boxes)
        # 判断人脸是否成功被检测, 当前只支持一张人脸的识别
        if len(encodings) == 1:
            face_encodings[image_path] = encodings[0]
        else:
            continue

    sorted_clusters = cluster_face_encodings(
        face_encodings, threshold=2.0, iterations=20)
    # print(len(sorted_clusters))

    print("\n Start clustering...")
    for idx, cluster in tqdm(enumerate(sorted_clusters)):
        cluster_dir = os.path.join(output_dir, str(idx))
        
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
            
        for path in cluster:
            # os.path.basename：返回最后的文件名
            shutil.copy(path, os.path.join(
                cluster_dir, os.path.basename(path)))


if __name__ == "__main__":
    main(output_dir='../results')
