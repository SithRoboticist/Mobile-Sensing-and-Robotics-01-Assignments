# -*- coding: utf-8 -*-
import pickle

import matplotlib.pyplot as plt
import numpy as np


def world2map(pose, gridmap, map_res):
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    # rays within range
    max_range = 80
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    points = np.array([
        np.multiply(ranges[idx], np.cos(angles[idx])),
        np.multiply(ranges[idx], np.sin(angles[idx]))
    ])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    """
    The same function used in the 'occupancy grid map' HW, and thus the same non-stated problem:
    r_ranges and r_angles need to be fed to this function as column vectors. >> must transpose the
    forms we have before feeding them to function.

    for further use in below functions, note that output is 2*37 (or generally 2*n)
    """
    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map,
                                                              1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map,
                                                              0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi
    particles[:, 3] = 1.0
    return particles


def plot_particles(particles, img_map, map_res):
    plt.matshow(img_map, cmap="gray")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    plt.plot(xs, ys, '.b')
    plt.xlim(0, np.size(img_map, 1))
    plt.ylim(0, np.size(img_map, 0))
    plt.show()


################### solution ###################
def wrapToPi(theta):
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta <= -np.pi:
        theta += 2 * np.pi
    return theta
    
    
def sample_normal_distribution(b):
    
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-b, b)
    
    return 0.5*tot


def forward_motion_model(x_robo, del_rot_1, del_trans, del_rot_2):
    
    x_prior, y_prior, theta_prior, trivial = x_robo

    x_post = x_prior + del_trans * np.cos(theta_prior + del_rot_1)
    y_post = y_prior + del_trans * np.sin(theta_prior + del_rot_1)
    theta_post = wrapToPi(theta_prior + del_rot_1 + del_rot_2)

    return np.array([x_post, y_post, theta_post, 1])


def sample_motion_model_odometry(x_robo_prev, u, noise_parameters):
    
    del_rot_1, del_trans, del_rot_2 = u
    a1, a2, a3, a4 = noise_parameters

    del_rot_1_hat = del_rot_1 + sample_normal_distribution(a1 * abs(del_rot_1) + a2 * del_trans)
    del_trans_hat = del_trans + sample_normal_distribution(a3 * del_trans + a4 * (abs(del_rot_1) + abs(del_rot_2)))
    del_rot_2_hat = del_rot_2 + sample_normal_distribution(a1 * abs(del_rot_2) + a2 * del_trans)

    x_t = forward_motion_model(x_robo_prev, del_rot_1_hat, del_trans_hat, del_rot_2_hat)
    
    return x_t


def compute_weights(x_poses, z_obs, gridmap, likelihood_map, map_res):
    """
    note that x_pose is an nx4 array. It includes all the sample points within a timestep.
    """
    weights = np.zeros((x_poses.shape[0], 1))
    for i, x_pose in enumerate(x_poses):
        weight = 1.0
        map_end_points = ranges2cells(z_obs[1,:].reshape(37, 1), z_obs[0,:].reshape(37, 1), x_pose, gridmap, map_res)
        for j in range(map_end_points.shape[1]):
            if (0 < map_end_points[0, j] < likelihood_map.shape[1]) and (0 < map_end_points[1, j] < likelihood_map.shape[0]):
                weight *= likelihood_map[map_end_points[1, j], map_end_points[0, j]]
                # the transpose relationship is due to x and j being along one another; same with y and i
            else:
                weight *= 1e-6
        
        weights[i, 0] = weight
    
    weights = weights/sum(weights)

    return weights


def resample(particles, weights, gridmap):
    """
    Inputs:
        - particles (nx4 ndarray): weighted particles that we want to resample to frequency-based particles
        - weights (nx1 ndarray): weights corresponding to the input particles
        - gridmap (2d ndarray): not sure what this guy is doing here! :) 
          just give it whatever img_map you have. I'd rather abstain from changing the code as mush as possible
    Output:
        - resampled_particles (nx4 ndarray): frequency-based particles. ready for motion update.
    """
    resampled_particles = []

    J = particles.shape[0] # number of particles >> number of pointers
    spacing = sum(weights)/J # Although we expect sum(weights) to be 1 since we're using normalized weights
    r = np.random.uniform(0, spacing) # starting point, i.e. position of the starting pointer on the cumulative weight axis
    c = weights[0]
    
    i = 0
    for j in range(1, J + 1):
        U = r + (j - 1) * spacing
        while U > c:
            i += 1
            c += weights[i]
        resampled_particles.append(particles[i])

    return np.array(resampled_particles)


def mc_localization(odom, z, num_particles, particles, noise, gridmap, likelihood_map, map_res, img_map):

    for i in range(len(odom)):
        
        z_t = z[i]
        u = odom[i]

        # updating the position of each particle through sample-based motion model
        # It was remiss of me to write this function in a way so it can deal with
        # a single value at a time! should've wrote it like the compute_weights function
        for j in range(num_particles):
            particles[j] = sample_motion_model_odometry(particles[j], u, noise)
        
        weights = compute_weights(particles, z_t, gridmap, likelihood_map, map_res)
        particles = resample(particles, weights, gridmap)

    return particles