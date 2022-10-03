
from pylab import *
import matplotlib
import numpy as np

fe=8000
Te=1/fe



# Question 2.3

# Preliminray Test
N= 63 #21 #5 #4   #20 number of coefficients are N+1 = 21
n=arange(-(N-1)/2,(N-1)/2+1)
figure(1)
plt.plot(n,'o')
plt.show()
g=0.5*sinc(n/2) # Half-band low-pass filter of 21 coefficients (symetrical)

figure(2)
plt.stem(n, g)
plt.show()
# Frequency domain representation
Nfft=  4096 #1024 # Zero-padding condition because the size of the filter is small (20 samples)
G=fft(g,Nfft)
figure(3)
freq_reel=arange(Nfft)/Nfft*fe
plt.plot(freq_reel, abs(G),'b')
plt.title('Nfft=4096')

plt.show()

Nfft=  1024 #1024 # Zero-padding condition because the size of the filter is small (20 samples)
G=fft(g,Nfft)
figure(3)
freq_reel=arange(Nfft)/Nfft*fe
plt.plot(freq_reel, abs(G),'b')
plt.title('Nfft=1024')
plt.show()

Nfft=  63 #1024 # Zero-padding condition because the size of the filter is small (20 samples)
G=fft(g,Nfft)
figure(3)
freq_reel=arange(Nfft)/Nfft*fe
plt.plot(freq_reel, abs(G),'b')
plt.title('Nfft=64')

plt.show()


Nfft=  8192 #1024 # Zero-padding condition because the size of the filter is small (20 samples)
G=fft(g,Nfft)
figure(3)
freq_reel=arange(Nfft)/Nfft*fe
plt.plot(freq_reel, abs(G),'b')
plt.title('Nfft=8192')

plt.show()