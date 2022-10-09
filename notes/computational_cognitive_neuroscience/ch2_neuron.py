from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Recorder:
    def __init__(s):
        s.records: Dict[str, List] = {}

    def record(s, k, v):
        if k not in s.records: s.records[k] = []
        s.records[k].append(v)
    
    def show(s):
        scale = 1.8
        plt.figure(figsize=(2*scale, len(s.records)*scale))
        for i, (k, v) in enumerate(s.records.items()):
            plt.subplot(len(s.records), 1, i+1)
            plt.title(k)  
            plt.plot(v)
        plt.tight_layout() 
        plt.show()

#============================================

def pos(x): return np.where(x < 0, 0, x)

def make_act_func(gamma, sigma):
    x = np.linspace(-2, 2, 1000)
    x2 = gamma * pos(x)
    y = x2 / (x2+1)
    gaussian = np.exp(-x**2/(2*sigma**2))
    gaussian /= np.sum(gaussian)
    y2 = np.convolve(y, gaussian, mode='same')
    mask = np.gradient(y2) >= 0
    x = x[mask]; y = y[mask]; y2 = y2[mask]
    act_func = interp1d(x, y2)
    if 0:
        plt.plot(x, y)
        plt.plot(x, act_func(x))
        plt.show()
    return act_func

#======================================

class Neuron:
    def __init__(s):
        s.VMR = 0.3
        s.GBE = 0.3; s.GBI = 1.0; s.GL = 0.3
        s.EE = 1.0; s.EI = 0.25; s.EL = 0.3
        s.DT_VM = 0.3
        s.THETA = 0.5 

        s.vm = s.VMR
        s.y = 0
    
    def update(s, ge, gi=0):
        ge *= s.GBE; gi *= s.GBI
        
        inet = ge*(s.EE-s.vm) + gi*(s.EI-s.vm) + s.GL*(s.EL-s.vm)
        s.vm += s.DT_VM * inet

        if mode=='spike':
            if s.vm > s.THETA: s.y = 1; s.vm = s.VMR
            else: s.y = 0
        elif mode=='rate':
            # Vm_eq = (ge*Ee + gi*Ei + gl*El) / (ge + gi + gl)
            ge_thr = (gi*(s.EI-s.THETA) + s.GL*(s.EL-s.THETA)) / (s.THETA-s.EE)
            ys = act_func(ge - ge_thr)
            s.y += s.DT_VM * (ys - s.y) 

        recorder.record('ge', ge)
        recorder.record('Inet', inet)
        recorder.record('Vm', s.vm)
        recorder.record('act', s.y)

#======================================

class Neuron2:
    def __init__(s):
        s.DT, s.G_TAU = 1, 1.4
        s.MAX_AVG = 0
        s.FF, s.FF0 = 1, 0.1
        s.FB_TAU, s.FB = 1.4, 1
        s.GI = 1.8
        s.EI, s.EL, s.EE, s.THR = 0.25, 0.3, 1, 0.5
        s.GBL, s.GBE = 0.3, 0.3
        s.VM_ACT_THR = 0.01
        s.VM_TAU = 3.3
        s.VMR = 0.3

        s.ge = 0
        s.fbi = 0
        s.act = 0
        s.vm = s.VMR

    def update(s, ge_raw):
        s.ge += s.DT * (1/s.G_TAU) * (ge_raw - s.ge)
        s.ge *= s.GBE

        avg_ge = np.mean(s.ge)
        ff_in = avg_ge + s.MAX_AVG * (np.max(s.ge) - avg_ge)
        ffi = s.FF * np.maximum(ff_in - s.FF0, 0)
        s.fbi += (1/s.FB_TAU) * (s.FB * np.mean(s.act) - s.fbi)
        gi = s.GI * (ffi + s.fbi)

        inet = s.ge*(s.EE-s.vm) + s.GBL*(s.EL-s.vm) + gi*(s.EI-s.vm)
        s.vm += (1/s.VM_TAU) * inet

        if mode=='spike':
            if s.vm > s.THR: s.act = 1; s.vm = s.VMR
            else: s.act = 0
        elif mode=='rate':
            ge_thr = (gi*(s.EI-s.THR) + s.GBL*(s.EL-s.THR)) / (s.THR-s.EE)
            nw_act = np.where(
                (s.act < s.VM_ACT_THR) & (s.vm <= s.THR),
                act_func(s.vm - s.THR),
                act_func(s.ge - ge_thr)
            )
            s.act += (1/s.VM_TAU) * (nw_act - s.act)

        recorder.record('ge', s.ge)
        recorder.record('Inet', inet)
        recorder.record('Vm', s.vm)
        recorder.record('act', s.act)


if __name__=='__main__':
    recorder = Recorder()
    act_func = make_act_func(gamma=30, sigma=0.01)
    mode = 'rate'
    neuron = Neuron()
    for t in range(10): neuron.update(0)
    for t in range(10, 160): neuron.update(1)
    for t in range(160, 200): neuron.update(0)
    #T = 200
    #for t in range(T): neuron.update(np.sin(t/T*4*np.pi)**2)
    recorder.show()
