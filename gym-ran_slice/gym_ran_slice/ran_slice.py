#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a slice RAN environment for the OpenAI gym environment

Created on December 13, 2021

@author: juanjosealcaraz
"""

import numpy as np
import gym
from gym import spaces

class RanSlice(gym.Env):
    ''' 
    class for a ran slicing environment 
    '''
    def __init__(self, node_b = None, penalty = 100):
        self.node_b = node_b
        self.penalty = penalty
        self.n_prbs = node_b.n_prbs
        self.n_slices = node_b.n_slices_l1
        self.n_variables = node_b.get_n_variables()
        self.action_space = spaces.Box(low=0, high = self.n_prbs,
                                        shape=(self.n_slices,), dtype=np.int) #得保证总的小于总prbs
        self.observation_space = spaces.Box(low=-float('inf'), high=+float('inf'), 
                                            shape=(self.n_variables,), dtype=np.float) #只是定义空间会头具体取值还要我去限制
    #动作空间一个列表包括每个函数可以
    def reset(self):
        """
        Reset the environment 
        """
        state = self.node_b.reset()

        return state # reward, done, info can't be included

    def step(self, action):
        """
        :action: [int, int, ...] Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional information
        """
        # apply the action
        state, info = self.node_b.step(action) #强化学习四元组 1.状态 2.奖励 3.动作 4.转移概率
        total_violations = info['violations'].sum() #step 就是读取状态给出奖励
        info['total_violations'] = total_violations
        if total_violations > 0:
            # if SLA not fulfilled the reward is negative
            reward = -1 * self.penalty * total_violations #惩罚
        else:
            # if SLA fulfilled the reward is the amount of free resources
            reward = max(0, self.node_b.n_prbs - action.sum()) #奖励是剩的资源越多越好 ，这是一个值得考量的问题

        return state, float(reward), False, info

    def render(self):
        pass