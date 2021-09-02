# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import welch
from scipy import fftpack

#Carrega o arquivo
samplerate, data = wavfile.read('581010__xcreenplay__smoking-in-the-angel-section2.wav')

#Carrega o arquivo em dois canais (audio estereo)
print(f"numero de canais = {data.shape[1]}")

#Tempo total = numero de amostras / fs
length = data.shape[0] / samplerate
print(f"duracao = {length}s")

#Plota as figuras ao longo do tempo

#Interpola para determinar eixo do tempo
time = np.linspace(0., length, data.shape[0])

#Plota os canais esquerdo e direito
plt.figure(1)
plt.plot(time, data[:, 0], label="Canal esquerdo")
plt.legend()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
#plt.show()

plt.figure(2)
plt.plot(time, data[:, 1], label="Canal direito")
plt.legend()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
#plt.show()

#Estima o espectro do sinal utilizando a funcao welch
x  = data[:, 0] # canal esquerdo
fs = 2*np.pi
#fs = samplerate
f, Pxx_spec = welch(x, fs, 'flattop', 512, scaling='spectrum')

#Plota o espectro do sinal para frequencias normalizadas entre 0 1 pi 
#(frequencias positivas)

plt.figure(3)
plt.semilogy(f, Pxx_spec)
plt.xlabel('frequencia [rad]')
plt.ylabel('Esoectro')
#plt.show()


#Plota espectro usando funcao FFT
nfft=4096
freq = np.linspace(0., samplerate, nfft) #Interpola para determinar eixo da frequencia
sig_fft = fftpack.fft(x,nfft)
plt.figure(4)
plt.plot(freq, np.abs(sig_fft))
plt.xlabel('frequencia [Hz]')
plt.ylabel('Esoectro de amplitudes')
plt.plot(freq, np.abs(fftpack.fftshift(sig_fft)))
plt.show()

