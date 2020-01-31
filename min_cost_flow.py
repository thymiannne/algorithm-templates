#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import heapq
from collections import deque


# import networkx as nx


class DiGraph:
    """
    有向グラフのクラス

    :attribute node: set
    :attribute edge: dict tail -> head -> capacity, cost -> value, value
    """

    def __init__(self):
        self.node = set()
        self.edge = {}

    def __repr__(self):
        return str(self.edge)

    def add_node(self, node_name):
        self.node.add(node_name)
        if node_name not in self.edge:
            self.edge[node_name] = {}

    def add_edge(self, tail_name, head_name, capacity, cost=0):
        if tail_name not in self.edge:
            self.edge[tail_name] = {}
        if head_name not in self.edge:
            self.edge[head_name] = {}
        self.edge[tail_name][head_name] = {'capacity': capacity, 'cost': cost}
        if tail_name not in self.node:
            self.add_node(tail_name)
        if head_name not in self.node:
            self.add_node(head_name)

    def remove_edge(self, tail_name, head_name):
        del self.edge[tail_name][head_name]

    def nodes(self):
        return self.node

    def edges(self):
        edge_list = []
        for tail in self.edge:
            for head in self.edge[tail]:
                edge_list.append((tail, head))
        return edge_list

    def successors(self, tail_name):
        return self.edge[tail_name].keys()

    def total_cost(self):
        return sum(self.edge[tail][head]['cost'] * self.edge[tail][head]['capacity']
                   for tail, head in self.edges())

    @classmethod
    def residual_network(cls, G, flow):
        """
        与えられた問題例Gとflowからresidual networkを作る関数．
        :param G: graph
        :param flow
        :return: residual_network
        """
        residual_net = cls()
        for tail, head in G.edges():
            amount = flow.edge[tail][head]['capacity']
            capacity = G.edge[tail][head]['capacity']
            cost = G.edge[tail][head]['cost']
            if amount < capacity:
                residual_net.add_edge(tail, head, capacity - amount, cost)
            if amount > 0:
                residual_net.add_edge(head, tail, amount, -cost)
        return residual_net

    @classmethod
    def bellman_ford(cls, residual, source, target):
        """
        ベルマンフォード法．
        コストが負の枝を含むならダイクストラ法ではなく
        こちらを使うこと．
        :param residual: residual network
        :param source
        :param target
        :return: minimum cost path
        """
        prev = {}
        dist = {v: math.inf for v in residual.nodes()}
        dist[source] = 0
        for i in range(len(residual.nodes()) - 1):
            for u, v in residual.edges():
                if dist[v] > dist[u] + residual.edge[u][v]['cost']:
                    dist[v] = dist[u] + residual.edge[u][v]['cost']
                    prev[v] = u
        for u, v in residual.edges():
            if dist[u] + residual.edge[u][v]['cost'] < dist[v]:
                raise AssertionError('Graph contains a negative-weight cycle!')
        if dist[target] == math.inf:
            return None
        else:
            path = cls()
            v = target
            while v != source:
                u = prev[v]
                path.add_edge(u, v, 1)
                v = u
            return path

    @classmethod
    def dijkstra_heap(cls, residual, source, target):
        """
        優先度付きキュー（二分ヒープ）を利用したダイクストラ法．
        コストが必ず正ならダイクストラでいいが，もし負のコストを含むなら
        Bellman Ford法を使わないといけない．
        :param residual: residual network
        :param source
        :param target
        :return: minimum cost path
        """
        prev = {}
        dist = {v: math.inf for v in residual.nodes()}
        dist[source] = 0
        H = []
        for v in residual.nodes():
            heapq.heappush(H, (dist[v], v))  # (始点から頂点vまでの距離, 頂点v)の順のタプルをキューに入れる
        while H:  # キューが空でない限り
            distance_u, u = heapq.heappop(H)  # キューから頂点uをポップ(自動的に距離が最小のやつを選んでる)
            for v in residual.successors(u):
                alt = distance_u + residual.edge[u][v]['cost']
                if dist[v] > alt:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(H, (alt, v))  # ヒープに新たな組(vの距離,頂点v)を加える．
                    # vの要素が元々あったのと重複するが，古い方はaltの分岐でelseに行く為ヒープはいずれ空になる．

        if dist[target] == math.inf:
            return None
        else:
            path = cls()
            v = target
            while v != source:
                u = prev[v]
                path.add_edge(u, v, 1)
                v = u
            return path

    @classmethod
    def shortest_path_repeat(cls, G, source, target, k):
        """
        最短路反復で最小費用流を解く．
        流量kの時に最も総費用の小さいフローを求める．
        :param G: Graph
        :param source
        :param target
        :param k: flow value
        :return: minimum cost flow and those value
        """
        flow = cls()
        for tail, head in G.edges():
            flow.add_edge(tail, head, 0, G.edge[tail][head]['cost'])
        i = 0
        while i < k:
            residual = cls.residual_network(G, flow)
            # path = cls.bellman_ford(residual, source, target)
            path = cls.dijkstra_heap(residual, source, target)
            if path is None:
                print('there is no feasible solution...')
                break
            for tail, head in path.edges():
                flow.edge[tail][head]['capacity'] += 1
            i += 1
        cost = flow.total_cost()
        return flow, cost


if __name__ == '__main__':
    G = DiGraph()
    G.add_edge(0, 1, 2, 2)
    G.add_edge(0, 2, 2, 5)
    G.add_edge(1, 2, 2, -1)
    G.add_edge(1, 3, 1, 8)
    G.add_edge(2, 3, 3, 3)
    print(G)
    print(DiGraph.shortest_path_repeat(G, 0, 3, 3))
