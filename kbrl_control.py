#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

"""

import numpy as np

DEBUG = True

class Learner:
    '''
    Auxiliary class with elements and variables for the KBRL agent
    '''
    def __init__(self, algorithm, indexes, initial_action, security_factor):
        self.algorithm = algorithm
        self.indexes = indexes
        self.initial_action = initial_action
        self.security_factor = security_factor
        self.step = 1

class KBRL_Control:
    '''
    KBRL_Control: Kernel Model-Based RL for online learning and control.
    Its objective is to assign spectrum resources (RBs) among several ran slices
    '''
    def __init__(self, learners, n_prbs, alfa = 0.05, accuracy_range = [0.99, 0.999]):
        self.learners = learners # there must be one learner instance per slice
        self.accuracy_range = accuracy_range
        self.n_slices = len(learners)
        self.n_prbs = n_prbs
        self.alfa = alfa
        self.adjusted = 0
        self.action = np.array([h.initial_action for h in learners], dtype = np.int16)
        self.security_factors = np.array([h.security_factor for h in learners], dtype = np.int16)        
        self.margins = np.array([0]*self.n_slices, dtype = np.int16) #这个裕量怎么处理
        intial_value = (self.accuracy_range[0] + self.accuracy_range[1])/2 #0.995
        self.accuracies = np.full((self.n_slices, self.n_prbs), intial_value, dtype = float) # 5 ， 200

    def select_action(self, state):
        action = np.zeros((self.n_slices), dtype = np.int16)
        adjusted = 0
        for i, h in enumerate(self.learners):
            algorithm = h.algorithm
            _i_ = h.indexes
            l1_state = state[_i_]
            min_prbs = 0
            max_prbs = self.n_prbs

            # we check the prediction for assignment "offset" prbs below the action 检查预测情况为了分配偏移的prbs
            offset = self.security_factors[i]
            margin = 0
            for l1_prbs in range(max(min_prbs - offset,0), max_prbs+1):# 0到200 就是求最小能满足SLA的
                x = np.append(l1_state, l1_prbs/self.n_prbs) #这个切片的状态和prb占总的的比例
#关键在于用来预测的变量，里面实际上没有强化学习的四元组
                prediction = algorithm.predict(x) #这里面有精确度吗
                if prediction == 1:
                    a = min(self.n_prbs, l1_prbs + offset) #offset多加的 在
                    margin = a - l1_prbs #这不就是offset吗
                    l1_prbs = a
                    break
            action[i] = l1_prbs
            self.margins[i] = margin

        assigned_prbs = action.sum() #这是求总数的了
        if assigned_prbs > self.n_prbs: # not enough resources
            adjusted = 1
            action, diff = self.adjust_action(action, assigned_prbs, self.n_prbs)
            self.margins = self.margins - diff #如果这个是负数，就说明这个达不到预期的精度了
        
        self.action = action

        return action, adjusted #如果调整了就说明超了
    #等比例减小的意思
    def adjust_action(self, action, assigned_prbs, n_prbs):
        relative_p = action / assigned_prbs #等比例减小的意思
        new_action = np.array([np.floor(n_prbs * p) for p in relative_p], dtype=np.int16)
        return new_action, action - new_action
    #更新控制是根据四元组 主要是更新预测结果吧
    def update_control(self, state, action, reward):
        hits = np.zeros((self.n_slices), dtype = np.int16)

        for i, h in enumerate(self.learners):
            algorithm = h.algorithm
            _i_ = h.indexes #这个智能体负责的十个状态 你像难道这样也要给一个lstm预测吗 ，他是一个智能体一个预测算法
            l1_state = state[_i_]
            l1_action = action[i]
            x = np.append(l1_state, l1_action/self.n_prbs) #状态以及分配到的prb占总的比例
            y_pred = algorithm.predict(x) #预测不是是不是1吗 直接吧这个算法变成lstm可以吗
            y = reward[i] #是验证预测的准不准的吗
            hit = y == y_pred #就是指是不是预测准了
            margin = max(0, self.margins[i])
            if y_pred == 1: #预测是成功 实际上失败了是
                if hit == 0: # with the same or less margin we would have made the same mistake
                    self.accuracies[i,0:margin+1] = (1 - self.alfa) * self.accuracies[i,0:margin+1] #就是比这个分配的少准确率是降低的
                else: # with the same or more margin we would have succeeded as well 预测成功了实际上也成功了
                    self.accuracies[i,margin:] = (1 - self.alfa) * self.accuracies[i,margin:] + self.alfa #就是比这个多的话准确率会很高
            if not self.adjusted: # if the action was not adjusted then update security_factor
                self.security_factors[i] = np.argmax(self.accuracies[i,:] > self.accuracy_range[0]) #这是关键，就是找到最小的prb满足精度要求

            hits[i] = hit
            # sample augmentation
            if y == 1: # fulfilled
                for a in range(l1_action, self.n_prbs + 1): # same or more prbs would obtain the same y
                    new_x = np.append(l1_state, a/self.n_prbs)
                    y_pred = algorithm.predict(new_x)
                    algorithm.update(new_x, y)
            else: # not fulfilled (y = -1)
                for a in range(0, l1_action + 1): #  same or fewer prbs would obtain the same y
                    new_x = np.append(l1_state, a/self.n_prbs)
                    y_pred = algorithm.predict(new_x)
                    algorithm.update(new_x, y)

        return hits
    #system是环境node_env steps是总步数
    def run(self, system, steps, learning_time = -1):
        action = self.action

        SLA_history = np.zeros((steps), dtype = np.int16) #状态之一
        reward_history = np.zeros((steps), dtype = np.float)# 奖励
        violation_history = np.zeros((steps), dtype = np.int16)#违反的SLA，也是状态之一吧，专用于kbrl的
        adjusted_actions = np.zeros((steps), dtype = np.int16)# 相当于动作
        resources_history = np.zeros((steps), dtype = np.int16)
        hits_history = np.zeros((len(action),steps), dtype = np.int16)

        state = system.reset() #5个切片的状态就是50个变量

        for i in range(steps):
            new_state, reward, _, info = system.step(action)
            SLA_labels = info['SLA_labels'] #这个才像真实的reward
            if learning_time < steps:
                #上一次的状态，这一次的动作，这一次的奖励
                hits = self.update_control(state, action, SLA_labels) #这应该是更新预测器的 用的是上一次的状态
            action, self.adjusted = self.select_action(new_state)
            state = new_state

            SLA_history[i] = SLA_labels.sum()
            reward_history[i] = reward
            violation_history[i] = info['total_violations']
            resources_history[i] = action.sum()
            adjusted_actions[i] = self.adjusted
            hits_history[:,i] = hits

            if i % 100 == 0:
                print('run kbrl steps {}'.format(i))

        print('mean resources = {}'.format(resources_history.mean()))
        print('total violations = {}'.format(violation_history.sum()))
        print('mean adjusted = {}'.format(adjusted_actions.mean()))
        print('mean accuracy = {}'.format(hits_history.mean(axis=1)))

        output = {
            'reward': reward_history, 
            'resources': resources_history, 
            'hits': hits_history,
            'adjusted': adjusted_actions,
            'SLA': SLA_history,
            'violation': violation_history
        }

        return output