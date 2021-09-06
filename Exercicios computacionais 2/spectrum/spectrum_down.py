# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import welch, lfilter
#from scipy import fftpack

#Carrega o arquivo
samplerate, data = wavfile.read('581010__xcreenplay__smoking-in-the-angel-section2.wav')

#Carrega o arquivo em dois canais (audio estereo)
print(f"numero de canais = {data.shape[1]}")

#Tempo total = numero de amostras / fs
length = data.shape[0] / samplerate
print(f"duracao = {length}s")

#Carrega os coeficientes do filtro
b = np.genfromtxt('coeffs.csv', delimiter=',')

#Interpola para determinar eixo do tempo
time = np.linspace(0., length, data.shape[0])

#Downsampling dos canais esquerdo e direito
data_l = data[:, 0]
data_r = data[:, 1]

#Filtra os dados dos canais esquerdo e direito
data_l = lfilter(b, 1, data[:, 0])
data_r = lfilter(b, 1, data[:, 1])

M=1

xl = data_l[0:-1:M]
xr = data_r[0:-1:M]

#Estima o espectro do sinal utilizando a funcao welch
x  = xl # canal esquerdo
fs = 2
#fs = samplerate
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')

#Plota o espectro do sinal para frequencias normalizadas entre 0 1 pi 
#(frequencias positivas)

plt.figure(1)
plt.semilogy(f, Pxx_spec)
plt.xlabel('frequencia [rad/($\pi$)]')
plt.ylabel('Esoectro')
plt.show()

#Plota coeficientes do filtro FIR
# plt.figure(3)
# plt.stem(b)
# plt.show()

# Escrita de arquivo dizimado
# audio = np.array([xl, xr]).T
# scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
# filename = 'signal_decimated_' + str(M) + '.wav'
# write(filename, samplerate//M, scaled)
    
