#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from collections import deque


# import networkx as nx


class DiGraph:
    """
    有向グラフのクラス

    :attribute node: set
    :attribute edge: dict tail -> head -> value
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

    def add_edge(self, tail_name, head_name, capacity):
        if tail_name not in self.edge:
            self.edge[tail_name] = {}
        if head_name not in self.edge:
            self.edge[head_name] = {}
        self.edge[tail_name][head_name] = capacity
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
            amount = flow.edge[tail][head]
            capacity = G.edge[tail][head]
            if amount < capacity:
                residual_net.add_edge(tail, head, capacity - amount)
            if amount > 0:
                residual_net.add_edge(head, tail, amount)
        return residual_net

    @classmethod
    def augmenting_path(cls, residual_net, source, target):
        """
        与えられた残余ネットワークとsourceとtargetからaugmenting pathを見つける関数．
        Edmonds-Karpアルゴリズム用に，枝数最小のaugmenting pathをBFSで見つける．
        Augmenting pathがある場合には，（augmenting path上の最小枝容量，DiGraphとしてのaugmenting path）を返す．
        Augmenting pathがない場合には，（0，空グラフ）を返す．
        :param residual_net
        :param source
        :param target
        :return: minimum capacity and augmenting path
        """

        dist = {v: math.inf for v in residual_net.nodes()}  # sourceからの距離
        prev = {}
        dist[source] = 0
        visited = {v: False for v in residual_net.nodes()}  # 探索済みかどうか
        visited[source] = True
        queue = deque([source])
        while queue:  # BFSで最短経路を探る
            v = queue.popleft()
            for w in residual_net.successors(v):
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)
                    dist[w] = dist[v] + 1
                    prev[w] = v
        aug_path = cls()
        if dist[target] != math.inf:
            v = target
            while v != source:
                u = prev[v]
                aug_path.add_edge(u, v, residual_net.edge[u][v])
                v = u

            minimum_capacity = min(aug_path.edge[tail][head]
                                   for tail, head in aug_path.edges())
        else:
            minimum_capacity = 0
        return minimum_capacity, aug_path

    @classmethod
    def edmonds_karp(cls, G, source, target):
        """
        Edmonds-Karpアルゴリズムを実行する関数．
        残余ネットワークに増加パスがある限り，流れに足していく．
        最大流を自前のDiGraph形式で返す．
        計算量はO(VE^2)
        :param G: graph
        :param source
        :param target
        :return: maximum flow and those value
        """
        flow = cls()
        for tail, head in G.edges():
            flow.add_edge(tail, head, 0)
        while True:
            residual_net = cls.residual_network(G, flow)
            min_cap, aug_path = cls.augmenting_path(residual_net, source, target)
            if min_cap == 0:
                break
            for tail, head in aug_path.edges():
                flow.edge[tail][head] += min_cap
        amount = sum(flow.edge[source].values())
        return flow, amount

    @classmethod
    def level_graph(cls, residual_net, source, target):
        """
        残余ネットからレベルグラフを構築する．
        レベルグラフとは，残余ネットのうち始点からの距離が正方向に向かっている
        枝のみを抽出したもの．
        BFSを用いる．
        :param residual_net
        :param source
        :param target
        :return: level graph
        """

        dist = {v: math.inf for v in residual_net.nodes()}
        dist[source] = 0
        visited = {v: False for v in residual_net.nodes()}
        visited[source] = True
        queue = deque([source])
        while queue:  # BFSで各頂点のレベル（始点からの距離）を割り当てる
            v = queue.popleft()
            for w in residual_net.successors(v):
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)
                    dist[w] = dist[v] + 1

        if dist[target] != math.inf:
            level = cls()
            for tail, head in residual_net.edges():
                if dist[head] == dist[tail] + 1:
                    level.add_edge(tail, head, residual_net.edge[tail][head])
            return level
        else:
            return None

    @classmethod
    def block_flow(cls, level, source, target):
        """
        レベルグラフからブロックフローを算出する．
        DFSを用いる．
        ブロックフローとは，各枝についてレベルグラフから
        ブロックフローの容量を引けば，s-tパスが無くなるようなフローのこと．
        要はレベルグラフの容量を埋めるようなフロ＝だと思えばいい．
        :param level graph
        :param source
        :param target
        :return: block flow
        """
        block = cls()
        for tail, head in level.edges():
            block.add_edge(tail, head, 0)
        while True:  # レベルグラフにs-tパスがある限り，DFSで経路を見つける
            prev = {}
            visited = {v: False for v in level.nodes()}
            visited[source] = True
            stack = [source]
            goal = False
            while stack:
                v = stack.pop()
                for w in level.successors(v):
                    if not visited[w]:
                        visited[w] = True
                        stack.append(w)
                        prev[w] = v
                        if w == target:
                            goal = True
                            break
            if goal:
                path = cls()
                v = target
                while v != source:
                    u = prev[v]
                    path.add_edge(u, v, level.edge[u][v])
                    v = u

                f = min(level.edge[tail][head]
                        for tail, head in path.edges())
                for tail, head in path.edges():
                    block.edge[tail][head] += f
                    if path.edge[tail][head] == f:  # 枝容量が埋まってれば
                        level.remove_edge(tail, head)  # レベルグラフから枝を除く
            else:
                break
        return block

    @classmethod
    def dinic(cls, G, source, target):
        """
        Dinic (Dinitzともいう) アルゴリズムを実行する関数．
        最大流を自前のDiGraph形式で返す．
        計算量はO(V^2E)  edmonds karpより速い
        edmonds karpとの違いについて述べると，
        edmonds karpでは各イテレーション毎に
        BFSを使って残余ネットの最短経路を一つだけ求めているのに対し，
        dinicでは各イテレーション毎に残余ネットからレベルグラフを取り，
        そこから増加パスの総和とも言えるブロックフローを足している．
        BFSとDFSどっちも使ってるってのがなんていうか，よき
        :param G: Graph
        :param source
        :param target
        :return: maximum flow and those value
        """
        flow = cls()
        for tail, head in G.edges():
            flow.add_edge(tail, head, 0)  # ここだけNetworkX利用時と違う．
        while True:
            residual_net = cls.residual_network(G, flow)
            level = cls.level_graph(residual_net, source, target)
            if level is None:
                break
            block = cls.block_flow(level, source, target)
            for tail, head in block.edges():
                flow.edge[tail][head] += block.edge[tail][head]
        amount = sum(flow.edge[source].values())
        return flow, amount


if __name__ == '__main__':
    G = DiGraph()
    G.add_edge(1, 2, 5)
    G.add_edge(1, 3, 20)
    G.add_edge(3, 2, 7)
    G.add_edge(2, 4, 15)
    G.add_edge(2, 5, 3)
    G.add_edge(3, 5, 13)
    G.add_edge(4, 5, 10)
    G.add_edge(4, 6, 15)
    G.add_edge(5, 6, 10)
    print(G)
    print(DiGraph.edmonds_karp(G, 1, 6))
    print(DiGraph.dinic(G, 1, 6))

    # 二部グラフでタイム計測実験
    import random
    import time

    H = DiGraph()
    males = ['m{}'.format(i) for i in range(100)]
    females = ['f{}'.format(i) for i in range(100)]
    for male in males:
        H.add_edge('s', male, 10000)
    for female in females:
        H.add_edge(female, 't', 10000)
    for _ in range(1000):
        H.add_edge(random.choice(males), random.choice(females), random.randint(1, 10))
    start = time.time()
    flow_e, f_e = DiGraph.edmonds_karp(H, 's', 't')
    middle = time.time()
    flow_d, f_d = DiGraph.dinic(H, 's', 't')
    end = time.time()
    assert f_e == f_d
    print('edmonds karp: {}'.format(middle - start))  # 大体3.367427110671997s
    print('dinic: {}'.format(end - middle))  # 大体0.5525598526000977s
