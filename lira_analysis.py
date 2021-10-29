#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:05:36 2021

@author: halidziya
"""
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

filename = 'data.csv'

data = pd.read_csv(filename)
n = len(data)
degisim_noktasi = 290 # 16 Nisan 2017
usd_price = data['Price'].tolist()[::-1]

def fit_line(x, y):
    lin = kl.Input((1,))
    out = kl.Dense(1)(lin)
    model = Model(lin, out)
    model.set_weights([np.array([[0.02]]),np.array([0])])
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
    model.compile(Adam(learning_rate=lr_schedule), 'mse')
    model.fit(x, y, epochs=1000, batch_size=128)
    wl, bl = model.get_weights()
    return wl[0][0], bl[0]


x_cumhuriyet = np.arange(0, degisim_noktasi, dtype=float)
cumhuriyet = np.array(usd_price[:degisim_noktasi])
x_baskanlik =  np.arange(degisim_noktasi, n, dtype=float)
baskanlik = np.array(usd_price[degisim_noktasi:])

w_cumhuriyet, b_cumhuriyet = fit_line(x_cumhuriyet, cumhuriyet)
y_pred = w_cumhuriyet*x_cumhuriyet + b_cumhuriyet

constraint = [degisim_noktasi]*1000 # Degisim noktalari yakinligi
contraint_pred = [y_pred[-1]]*1000
w_baskanlik, b_baskanlik = fit_line(np.hstack([x_baskanlik, constraint]),
                                    np.hstack([baskanlik, contraint_pred]))


estimate = (w_cumhuriyet*x_baskanlik + b_cumhuriyet)[-1]
actual =  (w_baskanlik*x_baskanlik + b_baskanlik)[-1]
loss_ratio = 1 - (estimate/actual)


plt.figure(figsize=(8,8))
plt.plot(usd_price)
plt.plot(x_cumhuriyet, w_cumhuriyet*x_cumhuriyet + b_cumhuriyet)
plt.plot(x_baskanlik, w_baskanlik*x_baskanlik + b_baskanlik)
plt.plot(x_baskanlik, w_cumhuriyet*x_baskanlik + b_cumhuriyet, '--')
plt.xlabel('Zaman')
plt.ylabel('USD')
plt.title(f'Turk Tipi Baskanlik Sisteminin Dolar uzerinde Etkisi (-%{int(loss_ratio*100)})')
plt.legend(['Gerceklesen', 'Oncesi', 'Sonrasi', 'Degisim olmasaydi'])
plt.savefig('analysis.svg')
plt.savefig('analysis.png', dpi=600)