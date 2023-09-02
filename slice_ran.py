#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

Classes:

UE
SliceRANmMTC
SliceRANeMBB

"""
DEBUG = True
CBR = 0
VBR = 1

import numpy as np
from traffic_generators import VbrSource, CbrSource

class MTCdevice:
    def __init__(self, id, repetitions, slice_ran_id):
        self.id = id
        self.repetitions = repetitions
        self.slice_ran_id = slice_ran_id
    def __repr__(self):
        return 'MTC {}'.format(self.id)
    
class SliceRANmMTC:
    '''
    Generates message arrivals at the mMTC devices
    according to the characteristics defined in MTC_description:
    - n_devices: total number of devices
    - repetition_set: possible repetitions
    - period_set: possible times between message arrivals
    '''
    #这一个切片已经确定1000个设备了，也不知道怎么分配prbs
    def __init__(self, rng, id, SLA, MTCdescription, state_variables, norm_const, slots_per_step):
        self.type = 'mMTC'
        self.rng = rng
        self.id = id
        self.SLA = SLA #delay = 300
        self.state_variables = state_variables # ['devices', 'avg_rep', 'delay']
        self.norm_const = norm_const # 100 all
        self.slots_per_step = slots_per_step

        self.n_devices = MTCdescription['n_devices'] 
        self.repetition_set = MTCdescription['repetition_set']
        self.period_set = MTCdescription['period_set']

        self.reset()

    def reset(self):
        self.reset_state()
        self.reset_info()
        self.period = np.ones((self.n_devices), dtype=np.int64) #每个设备的周期初始化为1，1000个设备
        self.t_to_arrival = np.zeros((self.n_devices), dtype=np.int64) #每个设备的到达时间初始化为0   一千的数组
        self.devices = [] #设备列表 1000个设备不一定什么时候来
        for i in range(self.n_devices): #i就是第几个设备
            repetitions = self.rng.choice(self.repetition_set) #随机选择重复次数 2到128
            self.period[i] = self.rng.choice(self.period_set) #随机选择周期 1000到100000
            self.t_to_arrival[i] = 1 + self.rng.choice(np.arange(self.period[i])) #到达的周期，每隔多少次到达
            self.devices.append(MTCdevice(i, repetitions, self.id)) #重复次数是不是会归0
    #时隙slot是用来干什么的就是一个时隙内环境的变化 但是一次到达之后，这样看的话实际上repetition没用到
    def slot(self):
        self.slot_counter += 1

        # advance time
        self.t_to_arrival -= 1

        # arrivals
        arrival_list = []
        arrivals = self.t_to_arrival == 0 #每个时隙都会有一些设备流量到达
        indices = np.where(arrivals)

        # print('indices = {}'.format(indices))
        for i in indices[0]:
            arrival_list.append(self.devices[i])

        # prepare for next arrival (deterministic inter arrival time)
        self.t_to_arrival[arrivals] = self.period[arrivals]

        return arrival_list, []

    def reset_info(self):
        self.info = {'delay': 0, 'avg_rep': 0, 'devices': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float32)

    def get_n_variables(self):
        return len(self.state_variables)

    def get_state(self):
        '''convert the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state
    #主要看这个怎么用的，这个决定了时延是怎么算的 而且也没有持续时间
    def update_info(self, delay, avg_rep, devices):
        self.info['delay'] += delay
        self.info['avg_rep'] += avg_rep
        self.info['devices'] += devices
        

    def compute_reward(self):
        '''assesses SLA violations'''
        SLA_fulfilled = self.info['delay']/self.slots_per_step < self.SLA['delay'] #300
        return not(SLA_fulfilled)
    
class UE:
    '''
    eMBB UE contains a traffic source that can be CRB (GBR) or VBR (non-GBR)
    '''
    def __init__(self, id, slice_ran_id, traffic_source, type, window = 50, slot_length = 1e-3):
        self.id = id
        self.slice_ran_id = slice_ran_id
        self.traffic_source = traffic_source
        self.type = type
        self.th = 0
        self.b = 1/window
        self.a = 1 - self.b
        self.queue = 0
        self.slot_length = slot_length

        # per subframe variables
        self.snr = 0 # real error values per prb
        self.e_snr = 0 # estimated error
        self.new_bits = 0 # incoming bits 传入数据
        self.bits = 0 # assigned bits 处理的数据
        self.prbs = 0 # assigned prbs
        self.p = 0 # reception probability
    
    def estimate_snr(self, snr):
        self.snr = snr
        self.e_snr = round(np.mean(snr)) #平均信噪比  

    #流量就是用来生成数据的 两个参数每步新到流量。以及更新这个用户的队列
    def traffic_step(self):
        self.new_bits = self.traffic_source.step() #流量源每个时隙的到达数
        self.queue += self.new_bits #队列长度
    #slef.bits需要在别的地方更新，处理了多少数据
    def transmission_step(self, received):
        if not received:
            self.bits = 0
        self.queue = max(self.queue - self.bits, 0)
        self.th = self.a * self.th + self.b * self.bits / self.slot_length

    def __repr__(self):
        return 'UE {}'.format(self.id)

class SliceRANeMBB:
    '''
    Generates arrivals and departures of eMBB ues.
    There are two traffic types: CRB (GBR) and VBR (non-GBR)
    CBR traffic parameters are given in CBR_description
    VBR traffic parameters are given in VBR_description
    '''
    #slots_per_step = 50
    #slot_length = 1e-3
    #norm_const 标准常数
    def __init__(self, rng, user_counter, id, SLA, CBR_description, VBR_description, state_variables, norm_const, slots_per_step, slot_length = 1e-3):
        self.type = 'eMBB'
        self.rng = rng
        self.user_counter = user_counter
        self.id = id
        self.slot_length = slot_length
        self.slots_per_step = slots_per_step
        self.observation_time = slots_per_step * slot_length #一个步长内的观测时间50ms
        self.SLA = SLA # service level agreement description
        self.state_variables = state_variables
        self.norm_const = norm_const

        self.cbr_arrival_rate = CBR_description['lambda'] #2/60
        self.cbr_mean_time = CBR_description['t_mean'] #30
        self.cbr_bit_rate = CBR_description['bit_rate'] #500000bits/s

        self.vbr_arrival_rate = VBR_description['lambda'] #5.0/60.0,
        self.vbr_mean_time = VBR_description['t_mean'] #30.0,
        # 1000 ，500 ，1
        self.vbr_source_data = {
            'packet_size': VBR_description['p_size'],
            'burst_size': VBR_description['b_size'],
            'burst_rate':VBR_description['b_rate']
        }
        self.reset()

    def reset(self):
        self.slot_counter = 0
        self.remaining_time = {} #持续时间
        self.cbr_steps_next_arrival = 0 #cbr下一个到达的时间
        self.vbr_steps_next_arrival = 0
        self.vbr_ues = {} #这是字典啊
        self.cbr_ues = {}
        self.reset_state()
        self.reset_info()

    def get_n_variables(self):
        return len(self.state_variables)

    def cbr_cac(self):
        '''Admission control for CBR users'''#准入控制
        slots = max(self.slot_counter,1) #时隙数量
        time = slots * self.slot_length
        cbr_prb = self.info['cbr_prb'] / slots #总共的时隙。总共的prb 平均一个时隙内的
        cbr_th = self.info['cbr_th'] / time #平均速率
        if cbr_prb >= self.SLA['cbr_prb'] or cbr_th >= self.SLA['cbr_th']: #如果prb或者速率超过了阈值这是满足SLA的
            return False
        return True

    def cbr_arrivals(self):
        if self.cbr_steps_next_arrival == 0: #下一次到达距离现在这步的时间
            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.cbr_arrival_rate)#指数分布到达时间 参数是1/lambda 就是指发生一次的平均时间。期望是1/lambda
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length) #四舍五入取整 300000个时隙到达一次 一个时隙就是1ms
            self.cbr_steps_next_arrival = inter_arrival_time #到达一个才会初始化一个用户吗

            if self.cbr_cac(): # check admission control 是指没有满足SLA 实际上意味着同一个切片同一时间大概率会完全服务于一个用户 
                # generate new user
                ue_id = next(self.user_counter) 
                cbr_source = CbrSource(bit_rate = self.cbr_bit_rate) 
                ue = UE(ue_id, self.id, cbr_source, CBR)
                self.cbr_ues[ue_id] = ue

                # generate holding time
                holding_time = self.rng.exponential(self.cbr_mean_time) #30 平均就是这么多期望是30
                holding_time = np.rint(holding_time / self.slot_length) #300000 也就是说平均处理完一个到一个
                self.remaining_time[ue_id] = holding_time #第几个用户的持续时间

                return [ue] # return user
        else:
            self.cbr_steps_next_arrival -= 1    
        return []
    #也就是说这种流量全部都要，接收，不会让用户等待
    def vbr_arrivals(self):
        if self.vbr_steps_next_arrival == 0:
            # create new vbr user
            ue_id = next(self.user_counter)
            vbr_source = VbrSource(**self.vbr_source_data)
            ue = UE(ue_id, self.id, vbr_source, VBR)
            self.vbr_ues[ue_id] = ue

            # generate holding time
            holding_time = self.rng.exponential(self.vbr_mean_time) #30
            holding_time = np.rint(holding_time / self.slot_length)
            self.remaining_time[ue_id] = holding_time #这个剩余时间应该得是两个加起来啊 vbr和cbr 不同的ue_id只有一个流量源，每个用户有一个

            # generate next arrival
            inter_arrival_time = self.rng.exponential(1.0 / self.vbr_arrival_rate)
            inter_arrival_time = np.rint(inter_arrival_time / self.slot_length)
            self.vbr_steps_next_arrival = inter_arrival_time
            return [ue]
        else:
            self.vbr_steps_next_arrival -= 1 #这个是下一个到达初始化的
            return []
    #slot离去吗
    def departures(self):
        departures = []
        current_ids = list(self.remaining_time.keys()) #用户id
        for id in current_ids:
            self.remaining_time[id] -= 1 #持续时间减一，这应该是slot里的
            if self.remaining_time[id] == 0:
                departures.append(id) #每个时间服务完成的用户数
                del self.remaining_time[id] # delete timer 服务完会被删除，remaining_time里面都是正在服务的用户
                self.vbr_ues.pop(id, None) # delete ue if here
                self.cbr_ues.pop(id, None) # or here    
        return departures   

    def slot(self): #一个时隙无非就是到达和离开
        self.slot_counter += 1
        arrivals = self.cbr_arrivals()
        arrivals.extend(self.vbr_arrivals())
        departures = self.departures()
        return arrivals, departures

    def reset_info(self):
        self.info = {'cbr_traffic': 0, 'cbr_th': 0, 'cbr_prb': 0, 'cbr_queue':0, 'cbr_snr': 0,\
                    'vbr_traffic': 0, 'vbr_th': 0, 'vbr_prb': 0, 'vbr_queue': 0, 'vbr_snr': 0}
        self.slot_counter = 0

    def reset_state(self):
        self.state = np.full((len(self.state_variables)), 0, dtype = np.float32)
    
    def update_info(self): #每步一更新的话，这是50ms的
        queue = 0
        snr = 0
        n = 0
        for ue in self.cbr_ues.values():
            self.info['cbr_traffic'] += ue.new_bits #总的到达数据
            self.info['cbr_th'] += ue.bits  #总的处理数据
            self.info['cbr_prb'] += ue.prbs #总的prbs
            queue += ue.queue
            snr += ue.e_snr #信噪比还用加的吗，总信噪比回头得平均吧
            n += 1
        n = max(n,1) #最少是1 n是用户数量，下面俩是平均值
        self.info['cbr_queue'] += queue/n #为什么队列要除以时隙数
        self.info['cbr_snr'] += snr/n

        queue = 0
        snr = 0
        n = 0
        for ue in self.vbr_ues.values():
            self.info['vbr_traffic'] += ue.new_bits
            self.info['vbr_th'] += ue.bits
            self.info['vbr_prb'] += ue.prbs
            queue += ue.queue
            snr += ue.e_snr
            n += 1
        n = max(n,1)
        self.info['vbr_queue'] += queue/n
        self.info['vbr_snr'] += snr/n

    def compute_reward(self):
        '''assesses SLA violations'''
        cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th'] #多个用户计算 50毫秒内的
        #SLA也就是说SLA是切片的属性，一个切片可以对应多个用户
        cbr_prb = self.info['cbr_prb']/self.slots_per_step > self.SLA['cbr_prb']
        cbr_queue = self.info['cbr_queue']/self.slots_per_step < self.SLA['cbr_queue']
        vbr_th = self.info['vbr_th']/self.observation_time > self.SLA['vbr_th']
        vbr_prb = self.info['vbr_prb']/self.slots_per_step > self.SLA['vbr_prb']
        vbr_queue = self.info['vbr_queue']/self.slots_per_step < self.SLA['vbr_queue']
        # the slice has to guarantee the objective delay for cbr and vbr if their traffics do not surpass the maximum
        cbr_fulfilled = cbr_th or cbr_prb or cbr_queue #如果有一个满足就是满足了
        vbr_fulfilled = vbr_th or vbr_prb or vbr_queue
        SLA_fulfilled = cbr_fulfilled and vbr_fulfilled
        return not(SLA_fulfilled) #not(SLA_fulfilled)是1就是不满足SLA SLA_fulfilled是0就是满足SLA

    def get_state(self):
        '''converts the info into a normalized vector'''
        for i, var in enumerate(self.state_variables):
            self.state[i] = self.info[var] / self.norm_const[var]        
        return self.state

