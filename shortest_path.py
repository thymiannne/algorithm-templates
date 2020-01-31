#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import heapq
from collections import deque


class Graph:
    def __init__(self):
        self.vertex = set()
        self.edge = {}

    def add_edge(self, tail, head, weight):
        if tail != head:  # self loop排除
            self.vertex.add(tail)
            self.vertex.add(head)
            self.edge[tail, head] = weight

    def add_weighted_edges_from(self, weights):
        for weight in weights:
            self.add_edge(*weight)


def bfs(G: Graph, source: int) -> dict:
    """重み無しなら幅優先探索でOK
    """
    adjacent = {v: [] for v in G.vertex}
    for tail, head in G.edge.keys():
        adjacent[tail].append(head)
    distance = {v: math.inf for v in G.vertex}
    distance[source] = 0
    prev = {}
    reached = {k: False for k in adjacent}
    reached[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacent[u]:
            if not reached[v]:
                reached[v] = True
                distance[v] = distance[u] + 1
                prev[v] = u
    return distance


def bellman_ford(G: Graph, source: int) -> dict:
    """ベルマンフォード法
    """
    distance = {v: math.inf for v in G.vertex}
    distance[source] = 0
    prev = {}
    vertices = list(G.vertex.copy())

    for i in range(len(vertices) - 1):
        for u, v in G.edge.keys():
            if distance[v] > distance[u] + G.edge[u, v]:
                distance[v] = distance[u] + G.edge[u, v]
                prev[v] = u

    for u, v in G.edge.keys():
        if distance[u] + G.edge[u, v] < distance[v]:
            raise AssertionError('Graph contains a negative-weight cycle!')
    return distance


def dijkstra(G: Graph, source: int) -> dict:
    """ヒープ無しダイクストラ法
    """
    adjacent = {v: [] for v in G.vertex}
    for tail, head in G.edge.keys():
        adjacent[tail].append(head)
    distance = {v: math.inf for v in G.vertex}
    distance[source] = 0
    prev = {}
    vertices = list(G.vertex.copy())
    while vertices:
        u = min(vertices, key=lambda x: distance[x])
        vertices.remove(u)
        for v in adjacent[u]:
            if distance[v] > distance[u] + G.edge[u, v]:
                distance[v] = distance[u] + G.edge[u, v]
                prev[v] = u
    return distance


def dijkstra_heap(G: Graph, source: int) -> dict:
    """優先度付きキュー（二分ヒープ）を利用したダイクストラ法
    """
    adjacent = {v: [] for v in G.vertex}
    for tail, head in G.edge.keys():
        adjacent[tail].append(head)
    distance = {v: math.inf for v in G.vertex}
    distance[source] = 0
    prev = {}
    H = []
    for v in G.vertex:
        heapq.heappush(H, (distance[v], v))  # (始点から頂点vまでの距離, 頂点v)の順のタプルをキューに入れる
    while H:  # キューが空でない限り
        distance_u, u = heapq.heappop(H)  # キューから頂点uをポップ(自動的に距離が最小のやつを選んでる)
        for v in adjacent[u]:
            alt = distance_u + G.edge[u, v]
            if distance[v] > alt:
                distance[v] = alt
                prev[v] = u
                heapq.heappush(H, (alt, v))  # ヒープに新たな組(vの距離,頂点v)を加える。
                # vの要素が元々あったのと重複するが、古い方はaltの分岐でelseに行く為ヒープはいずれ空になる。
    return distance


if __name__ == '__main__':
    G = Graph()
    weights = [[0, 1, 2], [0, 2, 6], [1, 2, 1], [1, 3, 5], [2, 3, 3]]
    G.add_weighted_edges_from(weights)
    print(bellman_ford(G, 0))
    print(dijkstra(G, 0))
    print(dijkstra_heap(G, 0))
    # 格子グラフでタイム計測テスト
    import time
    import random

    n, m = 100, 100
    G = Graph()
    weights = []
    for i in range(n):
        for j in range(m):
            if j < m - 1:
                weights.append([i * m + j, i * m + j + 1, random.randint(0, 10)])
            if i < n - 1:
                weights.append([i * m + j, (i + 1) * m + j, random.randint(0, 10)])
    G.add_weighted_edges_from(weights)
    before = time.time()
    bfs(G, 0)
    start = time.time()
    length_b = dijkstra(G, 0)[n * m - 1]
    foo = time.time()
    length_d = dijkstra(G, 0)[n * m - 1]
    middle = time.time()
    length_h = dijkstra_heap(G, 0)[n * m - 1]
    end = time.time()
    assert length_b == length_d and length_d == length_h
    print('bfs:', start - before)  # 大体0.009181022644042969s
    print('bellman ford:', foo - start)  # 大体6.942919015884399s
    print('dijkstra:', middle - foo)  # 大体6.883855104446411s
    print('heap dijkstra:', end - middle)  # 大体0.046556949615478516s
