# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def plot_state(mu, S, M):

    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_xlim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    # visualize result
    plt.plot(mu[0], mu[1], '.b')
    plot_2dcov(mu, S)
    plt.draw()
    plt.pause(0.01)


def plot_2dcov(mu, cov):

    # covariance only in x,y
    d, v = np.linalg.eig(cov[:-1, :-1])

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if (v[0, 0] == 0):
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])

    # create an ellipse
    ellipse = Ellipse((mu[0], mu[1]),
                      width=a * 2,
                      height=b * 2,
                      angle=np.deg2rad(theta),
                      edgecolor='blue',
                      alpha=0.3)

    ax = plt.gca()

    return ax.add_patch(ellipse)


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


def inverse_motion_model(pose, pose_prev):

    rot1 = wrapToPi(np.arctan2(pose[1] - pose_prev[1], pose[0] - pose_prev[0]) - pose_prev[2])
    trans = np.sqrt((pose[0] - pose_prev[0])**2 + (pose[1] - pose_prev[1])**2)
    rot2 = wrapToPi(pose[2] - pose_prev[2] - rot1)
    
    return rot1, trans, rot2

def ekf_predict(mu, S, u, R):
    """
    Performs the prediction step of the EKF based on the differential drive motion model

    Args:
        mu(ndarray): a 3x1 numpy array containing the expected values of x, y, and theta from the previous step. This is our linearization point
        S(ndarray): a 3x3 matrix containing the variances(noise) of x, y, and theta of the prior belief
        u(ndarray): a 2x3 numpy array containing the odometry readings 'pose' and 'prev_pose' >>> used in inverse_motion_model to give rot1, trans, and rot2
        R(ndarray): M would be consistent with the slides! a 3x3 diagonal matrix containing the variances(noise) of the process, in the CONTROL SPACE.

    Returns:
        mu(ndarray): a 3x1 numpy array containing the predicted expected values of x, y, and theta
        S(ndarray): a 3x3 matrix containing the variances(noise) of x, y, and theta of the predicted belief
    """
    theta = mu[2,0]
    rot1, trans, rot2 = inverse_motion_model(u[1,:], u[0,:])
    
    G_t = np.array([[1, 0, -trans*np.sin(theta + rot1)],
                    [0, 1,  trans*np.cos(theta + rot1)],
                    [0, 0,              1             ]])
    
    V_t = np.array([[-trans*np.sin(theta + rot1), np.cos(theta + rot1), 0],
                    [ trans*np.cos(theta + rot1), np.sin(theta + rot1), 0],
                    [            1              ,           0         , 1]])
    
    # M (here, R) is assumed given, so we don't have to calculate it using the alpha coefficients

    mu_bar = mu + np.array([[trans*np.cos(theta + rot1)],
                            [trans*np.sin(theta + rot1)],
                            [        rot1 + rot2       ]])
    
    S_bar = G_t@S@(G_t.T) + V_t@R@(V_t.T)

    return mu_bar, S_bar

def ekf_correct(mu_bar, S_bar, z, Q, M):
    """
    Performs the prediction step of the EKF based on the range-bearing observation model

    Args:
        mu_bar(ndarray): a 3x1 numpy array containing x_bar, y_bar, and theta_bar, the results of the current prediction step
        S_bar(ndarray): a 3x3 matrix containing the variances(noise) of x, y, and theta of the predicted belief
        z(ndarray): a 3xk matrix. k is the number of observed landmarks in this step. each column contains range, bearing, and landmark ID.
        Q(ndarray): 2x2 diagonal matrix. containts variances of range and bearing measurements.
        M(ndarray): 30x2 matrix. each row contains x and y of the landmark in the map.

    Returns:
        mu(ndarray): 3x1 matrix containing x, y, and theta. predicted and corrected through the EKF.
        S(ndarray): 3x3 covariance matrix. represents the final uncertainty of the entire current prediction-correction.
    """
    k = z.shape[1] # z being of shape 3xk, k number of observed landmarks in the current step
    for i in range(k):
        x_bar, y_bar, theta_bar = mu_bar[:, 0]
        lmx, lmy = M[int(z[2, i]), :] # map x and y of the observed landmark
        q = (lmx - x_bar)**2 + (lmy - y_bar)**2

        z_hat = np.array([[                        np.sqrt(q)                          ],
                          [ wrapToPi(np.arctan2(lmy - y_bar, lmx - x_bar) - theta_bar) ]], dtype=np.float64)
        
        H = np.array([[-(lmx - x_bar)/np.sqrt(q), -(lmy - y_bar)/np.sqrt(q),  0],
                      [     (lmy - y_bar)/q     ,     -(lmx - x_bar)/q     , -1]], dtype=np.float64)
        
        S_t = H@S_bar@(H.T) + Q
        K_t = S_bar@(H.T)@np.linalg.inv(S_t)
        
        mu_bar += K_t@(z[:2, i].reshape(2, 1) - z_hat)
        mu_bar[2, 0] = wrapToPi(mu_bar[2, 0])
        S_bar = (np.eye(3) - K_t@H)@S_bar

    mu, S = mu_bar, S_bar
    return mu, S

def run_ekf_localization(dataset, R, Q, verbose=False):
    # TODO

    # Initialize state variable
    mu = dataset['gt'][0]
    S = np.zeros([3, 3])

    # Read map
    M = dataset['M']

    # Gathered observations
    z = dataset['z']

    # Odometry readings
    odom = dataset['odom']

    # initialize figure
    plt.figure(10)
    # axes = plt.gca()
    # axes.set_xlim([0, 25])
    # axes.set_ylim([0, 25])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    num_steps = len(dataset['gt'])
    for i in range(num_steps):
        # iterate over prediction/correction and plot
        mu_bar, S_bar = ekf_predict(mu.reshape(3,1), S, np.array([odom[i],odom[i+1]]), R)
        mu, S = ekf_correct(mu_bar, S_bar, z[i], Q, M)
        if i%5 == 0:
            plt.plot(mu[0], mu[1], '.b')
        if i== num_steps-2:
            break

    plt.show()         
    return mu, S