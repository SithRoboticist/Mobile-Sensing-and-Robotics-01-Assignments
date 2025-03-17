# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(42)
# For random error reproductibility
# And of course, because the answer to the great question of “life, the universe and everything” is 42!

# define constants
d = 1                           # distance from rear wheel to front wheel
delta_max = 2 * math.pi / 3           # maximum steering angle
noisy_controls = False          # turn on/off noisy controls
Kp_velocity = 0.9                          # proportional gain
Kd_velocity = 0.15                          # derivative gain
Ki_velocity = 0                          # integral gain
Ld = 2                          # Lookahead distance
v_max = 30
integral_error_velocity = 0
prev_error_velocity = 0
integral_error_delta = 0
prev_error_delta = 0


# Robot Kinematics

class Robot:
    def __init__(self):
        self.x = 0          # x-position in the world coordinate system
        self.y = 0          # y-position in the world coordinate system
        self.theta = 0      # orientation in the world coordinate system
        self.v = 0          # velocity of the vehicle
        self.delta = 0      # steering angle of the front wheel

    def apply_control(self, a, delta_dot, dt):

        # update robot velocity and steering angle
        if noisy_controls == True:
            self.v = self.v + a*dt + 0.1*np.random.randn(1)[0]
            self.delta = self.delta + delta_dot*dt + \
                np.deg2rad(2)*np.random.randn(1)[0]
        else:
            self.v = self.v + a*dt
            self.delta = self.delta + delta_dot*dt

        # steering angle constraint
        if math.fabs(self.delta) > delta_max:
            self.delta = np.sign(self.delta)*delta_max

        # update robot pose
        self.x = self.x + self.v*math.cos(self.theta)*dt
        self.y = self.y + self.v*math.sin(self.theta)*dt
        self.theta = wrapToPi(
            self.theta + ((self.v*math.tan(self.delta))/d)*dt)


# Helper functions

def calculate_lookahead_index(robot, xs_des, ys_des):
    # search nearest point index
    dx = [robot.x - icx for icx in xs_des]
    dy = [robot.y - icy for icy in ys_des]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    distance_this_index = calc_distance(robot, xs_des[ind], ys_des[ind])

    while True:
        ind = ind + 1 if (ind + 1) < len(xs_des) else ind
        distance_next_index = calc_distance(robot, xs_des[ind], ys_des[ind])
        if distance_this_index < distance_next_index:
            break
        distance_this_index = distance_next_index

    # search look ahead target point index
    L = 0
    while Ld > L and (ind + 1) < len(xs_des):
        dx = xs_des[ind] - robot.x
        dy = ys_des[ind] - robot.y
        L = math.hypot(dx, dy)
        ind += 1

    return ind


def calc_distance(robot, point_x, point_y):
    dx = robot.x - point_x
    dy = robot.y - point_y
    return math.hypot(dx, dy)

# Manually added function
def calc_alpha(robot, lookahead_x, lookahead_y):
    robot_to_lookahead_vec_angle = np.arctan2(lookahead_y - robot.y, lookahead_x - robot.x)
    alpha = wrapToPi(robot_to_lookahead_vec_angle - robot.theta)
    return alpha


def calc_delta(alpha):
    global d, Ld
    delta = wrapToPi(np.arctan2(2*d*np.sin(alpha), Ld))
    return delta

# normalize angles between -pi and pi


def wrapToPi(theta):
    while theta < -math.pi:
        theta = theta + 2 * math.pi
    while theta > math.pi:
        theta = theta - 2 * math.pi
    return theta

# plots arrow given robot's trajectory or robot pose


def plot_arrow(x, y, theta, length=1.0, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, itheta) in zip(x, y, theta):
            plot_arrow(ix, iy, itheta)
    else:
        plt.arrow(x, y, length * math.cos(theta), length * math.sin(theta),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

# Control functions

# computes linear control
def longitudinal_controller(v, v_des, dt):
    global Kp_velocity, Kd_velocity, Ki_velocity, v_max
    global integral_error_velocity
    global prev_error_velocity

    error = v_des - v

    P = Kp_velocity * error

    integral_error_velocity += error*dt
    I = Ki_velocity * integral_error_velocity

    D = Kd_velocity * (error - prev_error_velocity) / dt
    prev_error_velocity = error

    a = P + I + D

    if v + a*dt > v_max:
        a = (v_max - v) / dt

    return a

# computes steering angle delta using pure-pursuit strategy
def lateral_controller(delta, delta_des, dt, K_p, K_d, K_i):
    global integral_error_delta, prev_error_delta, delta_max

    error = delta_des - delta

    P = K_p * error

    integral_error_delta =+ error*dt
    I = K_i * integral_error_delta

    D = K_d * (error - prev_error_delta) / dt
    prev_error_delta = error

    delta_dot = P + I + D

    if delta + delta_dot*dt > delta_max:
        delta_dot = (delta_max - delta) / dt
    
    return delta_dot