#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    # works only if poses are given row by row
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0]/map_res) + origin[0];
    new_pose[1] = np.round(pose[1]/map_res) + origin[1];
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    # in order for this function to work, you have to transpose the ranges, and give it in column by column.
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # works only if poses are given row by row
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    if p == 0:
        Log_Odds = -float("inf")
    elif p == 1:
        Log_Odds = float("inf")
    else:
        Log_Odds = np.log(p/(1-p))
    return Log_Odds
    
def logodds2prob(l):
    p = 1 - 1/(1 + np.exp(l))
    return p
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    if cell == endpoint:
        l = prob2logodds(prob_occ)
    else:
        l = prob2logodds(prob_free)
    return l

def grid_mapping_with_known_poses(poses_raw, ranges_raw, map_res, occ_gridmap, prior, prob_free, prob_occ):
    ranges_raw_transposed = ranges_raw.T
    vectorized_prob2logodds = np.vectorize(prob2logodds)
    vectorized_logodds2prob = np.vectorize(logodds2prob)

    log_odds_grid_map = vectorized_prob2logodds(occ_gridmap) # going to the logodds space

    for i in range(poses_raw.shape[0]):
        m_veh_loc = poses2cells(poses_raw[i,:], log_odds_grid_map, map_res)
        m_end_points = ranges2cells(ranges_raw_transposed[:,i], poses_raw[i,:], log_odds_grid_map, map_res)
        m_free_points = []
        for j in range(m_end_points.shape[1]):
            m_free_points.append(bresenham(m_veh_loc[0], m_veh_loc[1], m_end_points[0,j], m_end_points[1,j])[:-1])
        m_free_points = np.concatenate([beam for beam in m_free_points], axis=0)
        # m_end_points is a 2xn array, and m_free_points is an nx2 array. beware while indexing!
        # update the occupied cells:
        for k in range(m_end_points.shape[1]):
            log_odds_grid_map[m_end_points[0, k], m_end_points[1, k]] += prob2logodds(prob_occ)
        # update the free cells:
        for l in range(m_free_points.shape[0]):
            log_odds_grid_map[m_free_points[l, 0], m_free_points[l, 1]] += prob2logodds(prob_free)

    prob_grid_map = vectorized_logodds2prob(log_odds_grid_map)
    return prob_grid_map
