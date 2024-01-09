#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3, 2021

@author: juanjosealcaraz
"""

import numpy as np

'''proportional fair scheduler for average cqi reports'''
class ProportionalFair:
    def __init__(self, mcs_codeset, granularity = 2, slot_length = 1e-3, window = 50, sym_per_prb = 158):
        self.granularity = granularity
        self.mcs_codeset = mcs_codeset
        self.b = 1/window # 1/50
        self.a = 1 - self.b # 49/50
        self.sym_per_prb = sym_per_prb # 158 每个资源块有158个符号 
        self.slot_length = slot_length # 1ms

    def allocate(self, ues, n_prb, error_bound = 0.1):
        '''
        Updates the following variables of the ues:
        - ue.bits : assigned bits in this subframe
        - ue.prbs : assigned prbs in this subframe
        - ue.p : reception probability
        '''
        # create auxiliary data structures
        n_ues = len(ues) #用户数
        ue_rbs = np.zeros(n_ues, dtype = np.int) #每个用户分配的资源块数
        ue_mcs = np.zeros(n_ues, dtype = np.int) #每个用户分配的调制方式
        ue_queue = np.zeros(n_ues, dtype = np.int) #每个用户的队列长度
        ue_rate = np.zeros(n_ues, dtype = np.int) #每个用户的速率 rbs * bits_per_sym*sym_per_prb
        ue_bits = np.zeros(n_ues, dtype = np.int) #每个用户的比特数
        ue_th = np.zeros(n_ues) #每个用户的平均速率 

        # extract ue information
        for i, ue in enumerate(ues):
            ue_th[i] = max(ue.th, 1) # to avoid division by zero
            ue_queue[i] = ue.queue
            # determine the mcs given the objective and the estimated snr
            ue_mcs[i], bits_per_sym = self.mcs_codeset.mcs_rate_vs_error(ue.e_snr, error_bound) #指定错误下界情况下的最大调制方式mcs调制编号和每个符号的比特数=调制速率和调制阶数
            # achievable rate for the ue
            ue_rate[i] = self.sym_per_prb * bits_per_sym #每个资源块的搭载比特数
        
        # loop over the resources
        for r in range(0, n_prb, self.granularity): #这个分配是占时间的，总的158/2=79次，就是79个时间片，
            # prbs to be allocated in this iteration
            prbs = min(n_prb - r, self.granularity) #每次分配的资源块数基本就是两个

            # selected user for this resource (remove users without data)
            index = np.argmax(ue_rate * (ue_queue > 0)/ ue_th) #选择一个资源块除以吞吐量最大的用户ue_th是总量吗，
            #按说不能提前知道总量啊，还是说已经发送的，获得和发送质量成比例的吞吐量，要不为了效率最大会把资源块给质量最好的用户，这样就不会有用户一直占用资源块了
            
            # assign the resource to this ue
            ue_rbs[index] += prbs

            # update queue and throughput of this user
            tx_bits = min(prbs * ue_rate[index], ue_queue[index])
            ue_queue[index] -= tx_bits
            ue_bits[index] += tx_bits #每个用户发送的比特数

            # update the estimated throughput with current allocation
            ue_th[index] = self.a * ue_th[index] + self.b * ue_bits[index] / self.slot_length
    
        # update ues
        prb_i = 0
        for i, ue in enumerate(ues):
            prbs = ue_rbs[i]
            ue.prbs = prbs
            ue.bits = ue_bits[i]
            if prbs:
                snr_values = ue.snr[prb_i: prb_i + prbs]
                ue.p = self.mcs_codeset.response(ue_mcs[i], snr_values) #根据调制方式和信噪比计算接收概率
            else:
                ue.p = 0
            prb_i += prbs
