#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque


def dfs(adjacent_vertex, start_vertex):
    """深さ優先探索
    """
    reached = {k: False for k in adjacent_vertex}
    reached[start_vertex] = True
    stack = [start_vertex]
    while stack:
        v = stack.pop()
        for w in adjacent_vertex[v]:
            if not reached[w]:
                reached[w] = True
                stack.append(w)


def bfs(adjacent_vertex, start_vertex):
    """幅優先探索
    """
    reached = {k: False for k in adjacent_vertex}
    reached[start_vertex] = True
    queue = deque([start_vertex])
    while queue:
        v = queue.popleft()
        for w in adjacent_vertex[v]:
            if not reached[w]:
                reached[w] = True
                queue.append(w)


if __name__ == '__main__':
    pass
