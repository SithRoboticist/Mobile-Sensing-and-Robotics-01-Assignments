#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")




def motion_model_simple(action, prior):
    posterior = np.zeros(len(prior))
    backward_passive_forward = {'F':[0.1, 0.15, 0.75], 'B':[0.75, 0.15, 0.1]}
    for i in range(1,(len(prior)-1),1):
        posterior[i] = prior[i-1]*backward_passive_forward[action][2] + prior[i]*backward_passive_forward[action][1] + prior[i+1]*backward_passive_forward[action][0]
    posterior[0] = prior[0]*sum(backward_passive_forward[action][:2]) + prior[1]*backward_passive_forward[action][0]
    posterior[-1] = prior[-1]*sum(backward_passive_forward[action][1:]) + prior[-2]*backward_passive_forward[action][2]
    return posterior





def motion_model_matrix(action, belief):
    length = len(belief)
    mmm = np.zeros((length, length)) # motion model matrix
    B_S_F = {'F':[0.1, 0.15, 0.75], 'B':[0.75, 0.15, 0.1]}
    P_B = B_S_F[action][0]
    P_S = B_S_F[action][1]
    P_F = B_S_F[action][2]
    for i in range(length):
        if(i==0): # left boundry
            mmm[i, i:i+2] = P_S + P_B, P_B
        elif(i==(length-1)): # right boundry
            mmm[i, i-1:i+1] = P_F, P_S + P_F
        else:
            mmm[i, i-1:i+2] = P_F, P_S, P_B
    return mmm

    
    
    
    
def sensor_model(observation, belief, world):
    observation_belief_pair_pobabilities = {(1, 1):0.75, (0, 1):0.25, (0, 0):0.85, (1, 0):0.15}
    p_observation_given_belief = np.array([observation_belief_pair_pobabilities[(observation, world[i])] for i in range(len(world))])
    normalization_factor = 1 / np.dot(p_observation_given_belief, belief)
    normalized_observation_probabilities = normalization_factor*p_observation_given_belief
    # print('observation correction term: ', normalized_observation_probabilities) ###
    return normalized_observation_probabilities





def recursive_bayes_filter(actions, observations, belief, world):
    
    corrected_beliefs = []
    non_corrected_belief = belief
    
    for i in range(len(observations)):
        corrected_belief = sensor_model(observations[i], non_corrected_belief, world)*non_corrected_belief
        corrected_beliefs.append(corrected_belief)
        # print('corrected belief', i,':', corrected_beliefs[i]) ###
        # print(sum(corrected_beliefs[i]))
        if i == len(actions):
            break
        non_corrected_belief = motion_model_simple(actions[i], corrected_belief)
        
    return corrected_beliefs    
