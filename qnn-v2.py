'''
modify qnn-v1. 
In layer 1, we consider the repeated calculation motivating simplification
'''

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

def bits2signals(bits):
    # bits: input binary string with length of (4*Nt) 
    return np.array([qam16_modulation(bits[i:i+4]) for i in range(0, len(bits), 4)]).reshape(Nt,1)

def calculate_layer1(H_hat):
    output = np.empty((Nt,16,Nr), dtype=np.complex128)
    for ii in range(Nt):
        h = H_hat[:,ii]
        for jj in range(16):
            s_j = qam16_modulation(str(bin(jj)[2:].zfill(4)))*h
            output[ii][jj] = s_j
    return output

def calculate_layer2(output_layer1, y):
    dimension = 2**(4*Nt)
    output = np.empty(dimension)
    for index in range(dimension):
        s = np.zeros(Nr , dtype=np.complex128)
        binary_index = str(bin(index)[2:].zfill(4*Nt))
        for ii in range(Nt):
            s += output_layer1[ii][int(binary_index[4*ii:4*ii+4], 2)]
        error = y - s.reshape(Nr,1)
        output[index] = np.exp(-np.square(np.linalg.norm(error)))
    return output

def calculate_layer3(layer2_output):
    sum_exp = np.zeros((4*Nt, 2))
    for index, prob in enumerate(layer2_output):
        bits = str(bin(index)[2:].zfill(4*Nt))
        for ii in range(4*Nt):
            sum_exp[ii][eval(bits[ii])] += prob
    output = np.empty((4*Nt))
    for ii in range(4*Nt):
        output[ii] = sum_exp[ii][1]/(sum_exp[ii][1]+sum_exp[ii][0])
    return output

def calculate_square_error(layer3_output, true_sequence):
    dimension = len(true_sequence)
    loss = 0
    for index in range(dimension):
        if true_sequence[index] == '1':
            loss += np.square(1-layer3_output[index])
        else:
            loss += np.square(layer3_output[index])
    return loss

def calculate_cost_function(H_hat_vec):
    # a = time()
    H_hat = H_hat_vec[0:Nr*Nt].reshape(Nr,Nt)+1j*H_hat_vec[Nr*Nt:2*Nr*Nt].reshape(Nr,Nt)
    total_loss = 0
    training_length = len(y_sequence)
    for ii in range(training_length):
        output1 = calculate_layer1(H_hat)

        output2 = calculate_layer2(output1, y_sequence[ii])

        output3 = calculate_layer3(output2)

        true_sequence = ''.join(bits_sequence[ii*Nt+jj] for jj in range(Nt))

        total_loss += calculate_square_error(output3, true_sequence)

    mean_loss = total_loss/training_length
    # print(time()-a)
    return mean_loss

def detection(y, H_trained):
    output1 = calculate_layer1(H_trained)

    output2 = calculate_layer2(output1, y)

    output3 = calculate_layer3(output2) 

    detect_result = ''
    for ii in range(len(output3)):
        if(output3[ii]>0.5):
            detect_result += '1'
        else:
            detect_result += '0'
    return(detect_result)

def count_differences(str1, str2):
    return sum(a != b for a, b in zip(str1, str2))


def training():
    H_hat_vec = np.sqrt(1/2)*(np.random.randn(Nr*Nt*2))

    out = minimize(calculate_cost_function, x0=H_hat_vec, method="COBYLA", options={'maxiter':10})

    H_hat_vec = out.x

    H_trained = H_hat_vec[0:Nr*Nt].reshape(Nr,Nt)+1j*H_hat_vec[Nr*Nt:2*Nr*Nt].reshape(Nr,Nt)
    
    return H_trained

def calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing):
    error = 0
    for ii in range(len(y_sequence_testing)):
        detect_result = detection(y_sequence_testing[ii], H_trained)
        true_sequence = ''.join(bits_sequence_testing[ii*Nt+jj] for jj in range(Nt))
        error += count_differences(detect_result, true_sequence)
    BER = error/(len(y_sequence_testing)*len(detect_result))
    return BER


# generate training and tesing data
Nt = 2
Nr = 4
# generate channel

iter_num = 1
SNR_list = np.array([15])

SD_mean_performance = np.zeros(len(SNR_list))
QNN_mean_performance = np.zeros(len(SNR_list))

H_list = [np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt)) for ii in range(iter_num)]

for ii in range(len(SNR_list)):
    SNR_dB = SNR_list[ii]
    print("SNR_dB: "+str(SNR_dB))
    
    SD_performance = np.zeros(iter_num)
    QNN_performance = np.zeros(iter_num)

    for jj in range(iter_num):
        # print("current iter num: " +str(jj))
        H = H_list[jj]
        # print(H)
        bits_sequence_testing, x_sequence_testing, y_sequence_testing = generate_data(Nr,Nt,SNR_dB,1024,H)
        SD_performance[jj] = sphere_decoding_BER(H, y_sequence_testing, bits_sequence_testing, 1)
        print("SD: "+str(SD_performance[jj]))

        bits_sequence, x_sequence, y_sequence = generate_data(Nr,Nt,SNR_dB,128,H)
        H_trained = training()
        QNN_performance[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
        print("QNN: "+str(QNN_performance[jj]))

    SD_mean_performance[ii] = np.mean(SD_performance)
    QNN_mean_performance[ii] = np.mean(QNN_performance)

print(SD_mean_performance)
print(QNN_mean_performance)