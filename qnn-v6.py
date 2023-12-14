'''
optimization by gradient descent and momentum
'''

import numpy as np 
import random
from scipy.optimize import minimize
from sphere_decoding.sphereDecodingUseC import sphere_decoding_BER
import matplotlib.pyplot as plt
from timeit import default_timer as time



# generate signals for simulation
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

def generate_data(Nr,Nt,SNR_dB,length,H_channel):
    bits_sequence, x_sequence = generate_x_sequence(length, Nt)
    SNR= 10**(SNR_dB/10)
    n_sequence = [generate_noise(SNR, Nr) for i in range(length)]
    y_sequence = [np.dot(H_channel, x_sequence[i].reshape(Nt,1)) + n_sequence[i] for i in range(length)]
    return bits_sequence, x_sequence, y_sequence



# training H_hat



def bits2signals(bits):
    # bits: input binary string with length of (4*Nt) 
    return np.array([qam16_modulation(bits[i:i+4]) for i in range(0, len(bits), 4)]).reshape(Nt,1)

def calculate_layer1_training(H_hat, y):
    dimension_layer1 = 2**(4*Nt)
    # layer1 output
    output = np.empty(dimension_layer1)
    # calculate gradient components in layer1
    gradients = np.zeros((dimension_layer1, Nr, Nt), dtype=np.complex128)
    for index in range(dimension_layer1):
        bits = str(bin(index)[2:].zfill(4*Nt))
        s = bits2signals(bits)
        # s_conjugate_transpose = s.conj().T
        error = y - np.dot(H_hat,s)
        value =  np.exp(-np.square(np.linalg.norm(error)))
        output[index] = value
        gradient_component = np.dot(error, s.conj().T)
        gradients[index] = -value*(-gradient_component)
    return output, gradients

def layer2_matrix(n):
    if n == 1:
        return np.array([0,1])
    else:
        last_ = layer2_matrix(n-1)
        half_cols_num = 2**(n-1)
        first_row = np.concatenate((np.zeros(half_cols_num), np.ones(half_cols_num)))
        remain_rows = np.hstack((last_, last_))
        # print(remain_rows)
        return np.vstack((first_row, remain_rows))


def calculate_layer2_training(layer1_output, true_output):
    total_prob = np.sum(layer1_output)
    # print(total_prob)
    A = layer2_matrix(4*Nt)
    sum_prob_1 = np.dot(A, layer1_output)
    # layer2 output
    output = np.array([sum_prob_1[ii]/total_prob for ii in range(4*Nt)])
    # calculate gradient components in layer2
    gradients = np.zeros(2**(4*Nt))
    for ii in range(len(gradients)):
        for jj in range(4*Nt):
            gradients[ii] += 2*(true_output[jj]-output[jj])*(-(A[jj][ii]/total_prob)+(sum_prob_1[jj]/np.square(total_prob)))
    return output, gradients


def calculate_square_error(layer2_output, true_sequence):
    return np.linalg.norm(layer2_output-true_sequence)**2



def calculate_cost_function(H_hat):
    total_loss = 0
    total_gradients = np.zeros((Nr,Nt), dtype=np.complex128)
    training_length = len(y_sequence)
    for ii in range(training_length):
        # print(ii)
        true_sequence = ''.join(bits_sequence[ii*Nt+jj] for jj in range(Nt))
        true_sequence = np.array([eval(ii) for ii in true_sequence])
        layer1_output, layer1_gradients = calculate_layer1_training(H_hat, y_sequence[ii])
        layer2_output, layer2_gradients = calculate_layer2_training(layer1_output, true_sequence)
        total_loss += calculate_square_error(layer2_output,true_sequence)
        # SGD
        # if np.random.rand() < 0.6:
        for jj in range(2**(4*Nt)):
            total_gradients += (layer2_gradients[jj]*layer1_gradients[jj])
    mean_loss = total_loss/training_length
    return mean_loss, total_gradients


def training(max_iter):
    H_hat = np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt))
    # H_hat = H
    momentum = np.zeros((Nr,Nt),dtype=np.complex128)
    for iter_num in range(max_iter):
        # solve the gradient
        mean_loss, total_gradients = calculate_cost_function(H_hat)
        print("loss: "+str(mean_loss))
        # update H_hat
        momentum = (1-beta1)*total_gradients + beta1*momentum
        # print(alpha * momentum)
        # print("momentum norm: "+str(np.log10(np.sum(np.square(np.abs(momentum))))))
        H_hat -= alpha * momentum
        # print(H_hat)

    return H_hat


# testing QNN for detection
def calculate_layer1_testing(H_hat, y):
    dimension_layer1 = 2**(4*Nt)
    # layer1 output
    output = np.zeros(dimension_layer1)
    for index in range(dimension_layer1):
        bits = str(bin(index)[2:].zfill(4*Nt))
        s = bits2signals(bits)
        s_conjugate_transpose = s.conj().T
        error = y - np.dot(H_hat,s)
        value =  np.exp(-np.square(np.linalg.norm(error)))
        output[index] = value
    return output

def calculate_layer2_testing(layer1_output):
    total_prob = np.sum(layer1_output)
    A = layer2_matrix(4*Nt)
    sum_prob_1 = np.dot(A, layer1_output)
    # layer2 output
    output = np.array([sum_prob_1[ii]/total_prob for ii in range(4*Nt)])
    return output

def detection(y, H_trained):
    layer1_output = calculate_layer1_testing(H_trained, y)
    layer2_output = calculate_layer2_testing(layer1_output)
    detect_result = ''
    for ii in range(len(layer2_output)):
        if(layer2_output[ii]>0.5):
            detect_result += '1'
        else:
            detect_result += '0'
    return(detect_result)

def count_differences(str1, str2):
    return sum(a != b for a, b in zip(str1, str2))


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
SNR_list = np.array([0])

alpha = 0.01
beta1 = 0 #momentum rate
training_length = 50
pilot_length = 128


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

        bits_sequence_testing, x_sequence_testing, y_sequence_testing = generate_data(Nr,Nt,SNR_dB,1024,H)
        SD_performance[jj] = sphere_decoding_BER(H, y_sequence_testing, bits_sequence_testing, 1)
        print("SD: "+str(SD_performance[jj]))

        bits_sequence, x_sequence, y_sequence = generate_data(Nr,Nt,SNR_dB,pilot_length,H)
        H_trained = training(training_length)

        QNN_performance[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
        print("QNN: "+str(QNN_performance[jj]))

    SD_mean_performance[ii] = np.mean(SD_performance)
    QNN_mean_performance[ii] = np.mean(QNN_performance)

print(SD_mean_performance)
print(QNN_mean_performance)


# fig = plt.figure()

# ax1 = fig.add_subplot(111)

# lns1 = ax1.plot(SNR_list, SD_mean_performance, '-ro', linewidth=2.0, label="Sphere Decoding")
# lns2 = ax1.plot(SNR_list, QNN_mean_performance, '-bo', linewidth=2.0, label="QNN Decoding")

# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc="lower left")
# ax1.grid()

# ax1.set_xticks(SNR_list)
# ax1.set_yscale("log")
# ax1.set_adjustable("datalim")
# ax1.set_ylim(1e-6, 0.5)
# ax1.set_ylabel("BER")
# ax1.set_xlabel("SNR(dB)")


# # plt.savefig('convergence.pdf',dpi=600, bbox_inches='tight')
# plt.show()
