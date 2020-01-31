#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import networkx as nx
from collections import deque


class UnionFindTree:
    def __init__(self, iterable):
        self.parent = {v: v for v in iterable}
        self.rank = {v: 0 for v in iterable}

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

    def find(self, x):
        if self.parent[x] == x:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def same_check(self, x, y):
        return self.find(x) == self.find(y)


class Graph:
    def __init__(self):
        self.vertex = set()
        self.edge = {}
        self.adjacent = {}

    def __repr__(self):
        return str(self.edge)

    def add_edge(self, tail, head, weight):
        if tail != head:  # self loop排除
            self.vertex.add(tail)
            self.vertex.add(head)
            self.edge[tail, head] = weight
            if tail not in self.adjacent:
                self.adjacent[tail] = [head]
            else:
                self.adjacent[tail].append(head)
            if head not in self.adjacent:
                self.adjacent[head] = [tail]
            else:
                self.adjacent[head].append(tail)

    def add_weighted_edges_from(self, weights):
        for weight in weights:
            self.add_edge(*weight)

    def remove_edge(self, tail, head):
        del self.edge[tail, head]
        self.adjacent[tail].remove(head)
        self.adjacent[head].remove(tail)

    def total_weight(self):
        return sum(self.edge.values())


def is_tree(G, start):
    """幅優先探索で巡回路判定やろうとしたけど、
    うまくいかないので使わないこと
    """
    connected = {v: False for v in G.vertex}
    connected[start] = True
    queue = deque([start])
    u = -1
    while queue:
        v = queue.popleft()
        for w in G.adjacent[v]:
            if not connected[w]:
                connected[w] = True
                queue.append(w)
            elif w != u:
                return False
        u = v
    return True


def kruskal(G):
    """
    クラスカル法。最小全域木を返す。
    UnionFind木を用いて巡回路判定を行う。
    :param G: Graph
    :return: minimum spanning tree
    """
    H = Graph()
    UFT = UnionFindTree(G.vertex)
    n = len(G.vertex) - 1
    edges = sorted(G.edge.items(), key=lambda x: x[1])
    print('edges: ', edges)
    i = 0
    for (u, v), weight in edges:
        if UFT.same_check(u, v):
            print('not tree!')
        else:
            H.add_edge(u, v, weight)
            UFT.union(u, v)
            print('tree!')
            i += 1
        if i >= n:
            break
    print('parents: ', UFT.parent)
    print('rank: ', UFT.rank)
    return H


if __name__ == '__main__':
    G = Graph()
    weights = [[1, 3, 5], [1, 2, 3], [2, 3, 1], [3, 4, 5], [4, 5, 9], [3, 5, 2]]
    G.add_weighted_edges_from(weights)
    print(kruskal(G))

    G = Graph()
    weights = [[1, 2, 2], [1, 3, 1], [2, 3, 1], [2, 4, 3], [2, 5, 5], [3, 5, 3], [4, 5, 4]]
    G.add_weighted_edges_from(weights)
    print(kruskal(G))

    G = Graph()
    weights = [[1, 2, 33],
               [2, 3, 16],
               [1, 4, 98],
               [2, 5, 9],
               [3, 6, 84],
               [4, 5, 73],
               [5, 6, 49],
               [4, 7, 18],
               [5, 8, 61],
               [6, 9, 58],
               [7, 8, 64],
               [8, 9, 98]]
    G.add_weighted_edges_from(weights)
    print(kruskal(G))
