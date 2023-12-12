import numpy as np 
import random
from scipy.optimize import minimize
from sphere_decoding.sphereDecodingUseC import sphere_decoding_BER
import matplotlib.pyplot as plt
from timeit import default_timer as time



# generate transmit signal
def generate_random_bit_sequence(length):
    return ''.join(random.choice('01') for _ in range(length))

def qam16_modulation(binary_input):
    mapping = {
        '0000': (1+1j),
        '0001': (1+3j),
        '0010': (3+1j),
        '0011': (3+3j),
        '0100': (1-1j),
        '0101': (1-3j),
        '0110': (3-1j),
        '0111': (3-3j),
        '1000': (-1+1j),
        '1001': (-1+3j),
        '1010': (-3+1j),
        '1011': (-3+3j),
        '1100': (-1-1j),
        '1101': (-1-3j),
        '1110': (-3-1j),
        '1111': (-3-3j)
    }
    return mapping.get(binary_input, "Invalid binary input")/np.sqrt(10)

def generate_x_sequence(length, Nt):
    total_bits_sequence = generate_random_bit_sequence(length*Nt*4)
    bits_sequence = [total_bits_sequence[i:i+4] for i in range(0, len(total_bits_sequence), 4)]
    x_sequence = [np.array([qam16_modulation(bits_sequence[i+j]) for j in range(Nt)]) for i in range(0, len(bits_sequence), Nt)]
    return bits_sequence, x_sequence

def generate_noise(SNR, Nr):
    return np.sqrt(1/(2*SNR))*(np.random.randn(Nr,1)+1j*np.random.randn(Nr,1))

# generate training and tesing data
def generate_data(Nr,Nt,SNR_dB,length,H_channel):
    bits_sequence, x_sequence = generate_x_sequence(length, Nt)
    SNR= 10**(SNR_dB/10)
    n_sequence = [generate_noise(SNR, Nr) for i in range(length)]
    y_sequence = [np.dot(H_channel, x_sequence[i].reshape(Nt,1)) + n_sequence[i] for i in range(length)]
    return bits_sequence, x_sequence, y_sequence


# generate training and tesing data
Nt = 5
Nr = 5
# generate channel

iter_num = 1
SNR_list = np.array([0,5,10,15,20,25])

SD_mean_performance_20 = np.zeros(len(SNR_list))
SD_mean_performance_5 = np.zeros(len(SNR_list))

H_list = [np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt)) for ii in range(iter_num)]

for ii in range(len(SNR_list)):
    SNR_dB = SNR_list[ii]
    print("SNR_dB: "+str(SNR_dB))
    
    SD_performance_20 = np.zeros(iter_num)
    SD_performance_5 = np.zeros(iter_num)

    for jj in range(iter_num):
        # print("current iter num: " +str(jj))
        H = H_list[jj]
        # print(H)
        bits_sequence_testing, x_sequence_testing, y_sequence_testing = generate_data(Nr,Nt,SNR_dB,1024,H)
        SD_performance_20[jj] = sphere_decoding_BER(H, y_sequence_testing, bits_sequence_testing, 20)
        SD_performance_5[jj] = sphere_decoding_BER(H, y_sequence_testing, bits_sequence_testing, 0.1)

    SD_mean_performance_20[ii] = np.mean(SD_performance_20)
    SD_mean_performance_5[ii] = np.mean(SD_performance_5)


fig = plt.figure()

ax1 = fig.add_subplot(111)

lns1 = ax1.plot(SNR_list, SD_mean_performance_20, '-ro', linewidth=2.0, label="Sphere Decoding, d=20")
lns2 = ax1.plot(SNR_list, SD_mean_performance_5, '-bo', linewidth=2.0, label="Sphere Decoding, d=5")

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="lower left")
ax1.grid()

ax1.set_xticks(SNR_list)
ax1.set_yscale("log")
ax1.set_adjustable("datalim")
ax1.set_ylim(1e-6, 0.5)
ax1.set_ylabel("BER")
ax1.set_xlabel("SNR(dB)")


# plt.savefig('convergence.pdf',dpi=600, bbox_inches='tight')
plt.show()
