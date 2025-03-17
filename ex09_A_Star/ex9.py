#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

thr_free = 0.9

def plot_path(path, x_start, x_goal, M):
    plt.matshow(M, cmap="gray")
    if path.shape[0] > 2:
        plt.plot(path[:, 1], path[:, 0], 'b')
    plt.plot(x_start[1], x_start[0], 'or')
    plt.plot(x_goal[1], x_goal[0], 'xg')
    plt.show()

def plot_path_and_visited(path, visited, x_start, x_goal, M, visit_step = 50):
    plt.matshow(M, cmap="gray")
    if path.shape[0] > 2:
        plt.plot(path[:, 1], path[:, 0], 'b')
    plt.plot(x_start[1], x_start[0], 'or')
    plt.plot(x_goal[1], x_goal[0], 'xg')
    for i in range(0, len(visited), visit_step):
        plt.scatter(visited[i][1], visited[i][0], color='orange', s=1, alpha=1)
    plt.show()


def is_valid(y, x, M):
    if (0 <= y < M.shape[0]) and (0 <= x < M.shape[1]) and (M[y, x] > thr_free):
        return True
    return False

def Euclidean(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def plan_path_uninformed(x_start, x_goal, M):
    
    # position to explore     
    queue = deque([tuple(x_start)])
    # visited nodes in M
    visited = set([tuple(x_start)])
    # parent cell of each expanded cell, for backtracking
    parent = {}

    # possible movements
    movements = [
        (-1, 0),    # up
        (-1, 1),    # right-up
        (0, 1),     # right
        (1, 1),     # right-down
        (1, 0),     # down
        (1, -1),    # left-down
        (0, -1),    # left
        (-1, -1),   # left-up
    ]
    
    while queue:
        
        # 1. Pick one point to explore and remove it from queue
        current_point = queue.popleft()
        # 2. Check if the point is x_goal
        if current_point == tuple(x_goal):
            path = [list(x_goal)]
            while current_point != tuple(x_start):
                current_point = parent[current_point]
                path.insert(0, list(current_point))
            return path, [list(v) for v in visited]
        # 3. Get neighbours and check their validity
        for move in movements:
            next_point = (current_point[0] + move[0], current_point[1] + move[1])
            if is_valid(*next_point, M) and next_point not in visited:
                queue.append(next_point)
                visited.add(next_point)
                parent[next_point] = current_point
        # 4. Add neighbours to queue if not already visited
    
    return [], [list(v) for v in visited] 
    
def plan_path_astar(x_start, x_goal, M):
    
    x_start, x_goal = tuple(x_start), tuple(x_goal)
    # position to explore     
    open_list = [[Euclidean(*x_start, *x_goal), 0, *x_start]]
    
    # parent cell of each expanded cell, for backtracking
    parent = {}

    # possible movements
    movements = [
        (-1, 0),    # up
        (-1, 1),    # right-up
        (0, 1),     # right
        (1, 1),     # right-down
        (1, 0),     # down
        (1, -1),    # left-down
        (0, -1),    # left
        (-1, -1),   # left-up
    ]

    # visited nodes in M
    visited = set([x_start])
    
    while open_list:
        open_list.sort(key=lambda element: element[0])
        # 1. Pick the point to explore which minimizes f and remove it from queue
        current_f, current_g, current_y, current_x = open_list.pop(0)
        current_point = (current_y, current_x)
        # 2. Check if the point is x_goal
        if current_point == x_goal:
            path = [list(x_goal)]
            while current_point != x_start:
                current_point = parent[current_point]
                path.insert(0, list(current_point))
            return path, [list(v) for v in visited]
        # 3. Get neighbours, check their validity and expected cost
        for move in movements:
            next_point = (current_point[0] + move[0], current_point[1] + move[1])
            if is_valid(*next_point, M) and next_point not in visited:
                next_g = current_g + 1
                next_f = next_g + Euclidean(*next_point, *x_goal)
                open_list.append([next_f, next_g, *next_point])
                visited.add(next_point)
                parent[next_point] = current_point
    
    return [], [list(v) for v in visited]

class Node:
    def __init__(self, parent = None, cell = None):
        # parent is the node from which you reach the current one
        self.parent = parent
        # cell is the position of the node 
        self.cell = cell 
        # values for the cost functions
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __str__(self):
        return str(self.cell)
    
    def __repr__(self):
        return str(self)
