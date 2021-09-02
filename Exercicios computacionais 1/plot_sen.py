# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:48:10 2021

@author: ttlba
"""

import matplotlib.pyplot as plt #For plotting
from math import sin, pi #For generating input signals
import numpy as np

### Frequencia da funcao seno gerada
f_c = 1000 #1KHz

### Periodo de amostragem
fs = 48000 # Freq de amostragem = 48KHz
T = 1/fs

### Numero de amostras em 1s
ns = fs*1

### Inicializacao de arrays para coletar 1s de dados
input  = [0]*ns
t_axis = np.arange(0., ns)*T

### Funcao seno amostrada ate 1s 
for i in range(ns):
    input[i] = sin(2 * pi * f_c * i * T) 
    

### Seleciona amostras do seno: #1/100 de 1s
n_plot=200
t_plot = t_axis[0:n_plot] 
input_section = input[0:n_plot] 

### Plot da funcao seno continua (funcao plot "simula" uma funcao continua)
plt.figure(1)                
plt.ylabel('sen($2\pi f_c t$)')
plt.xlabel('t [s]') 
plt.title('Senóide continua')      
plt.plot(t_plot,input_section)

### Plot da funcao seno amostrada com fs = 48KHz
plt.figure(2)                
plt.ylabel('sen($2\pi f_c n T$)')
plt.xlabel('n') 
plt.title('Senóide discreta')      
plt.stem(input_section)