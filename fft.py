import matplotlib.pyplot as plt
import numpy as np


#numero de pontos
n =1000

#tamanho do eixo x
tx = 200

#frequencia angular
w = 2.0 * np.pi/tx

#base tempo
t = np.linspace(0, tx, n)

s1 = 2.0 * np.cos(2.0*w*t)
s2 = 1.0 * np.cos(30.0*w*t)

s = s1 + s2

#base de tempo para frequencia
freq = np.fft.fftfreq(n)

mascara = freq > 0

fft_calculo = np.fft.fft(s)
fft_abs = 2.0*np.abs(fft_calculo/n)
plt.figure(1)
plt.title("Sinal Original")
plt.plot(t, s)

plt.figure(2)
plt.title("Sinal Oda fft")
plt.plot(freq[mascara], fft_abs[mascara])

plt.show()