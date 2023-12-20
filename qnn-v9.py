'''
correct the way of generating data
'''

import numpy as np 
import random
from scipy.linalg import orth
from sphere_decoding.sphereDecodingUseC import sphere_decoding_BER
import matplotlib.pyplot as plt
# from timeit import default_timer as time

# 交叉熵正常训练


Nt = 2
Nr = 4

iter_num = 30
channel_list = np.load("channel_list_4_2.npy")
H_list = channel_list[0:iter_num]
cov_list = np.load("covmatrix_list_4.npy")

SNR_list = np.array([30])


alpha = 1e-4

max_iter = 100

pilot_length = 128

beta1 = 0

SD_mean_performance = np.zeros(len(SNR_list))
QNN_mean_performance_128 = np.zeros(len(SNR_list))

save_loss = np.empty((len(SNR_list), iter_num))
save_BER = np.empty((len(SNR_list), iter_num))
save_channel = np.empty((len(SNR_list), iter_num, Nr, Nt), dtype=np.complex128)

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

def generate_channel(SNR):
    return np.sqrt(SNR)*H_list

def generate_covmatrix(dimension):
    U = orth(np.random.randn(dimension,dimension))
    x = np.random.rand(dimension)
    x = (x/np.sum(x))*dimension
    V = np.diag(x)
    whitening_matrix = np.dot(U, np.sqrt(V))
    cov_matrix = np.dot(whitening_matrix, whitening_matrix.conj().T)
    return cov_matrix

def generate_noise(cov_matrix,Nr):
    real_part = np.random.multivariate_normal(np.zeros(Nr), cov_matrix/2)
    imag_part = np.random.multivariate_normal(np.zeros(Nr), cov_matrix/2)
    return (real_part+1j*imag_part).reshape(Nr,1)

def generate_data(Nr,Nt,length,H_channel,cov_matrix):
    bits_sequence, x_sequence = generate_x_sequence(length, Nt)
    n_sequence = np.empty((length,Nr,1), dtype=np.complex128)
    for ii in range(length):
        n_sequence[ii] = generate_noise(cov_matrix,Nr)
    y_sequence = np.empty((length,Nr,1), dtype=np.complex128)
    for ii in range(length):
        s = np.dot(H_channel, x_sequence[ii].reshape(Nt,1))
        y_sequence[ii] = s + n_sequence[ii]
    return bits_sequence, x_sequence, y_sequence

# training H_hat

def bits2signals(bits):
    # bits: input binary string with length of (4*Nt) 
    return np.array([qam16_modulation(bits[i:i+4]) for i in range(0, len(bits), 4)]).reshape(Nt,1)

def calculate_layer1_training(H_hat, y):
    dimension_layer1 = 2**(4*Nt)
    # layer1 output
    error_norm = np.empty(dimension_layer1)
    output = np.empty(dimension_layer1)
    # calculate gradient components in layer1
    gradients = np.zeros((dimension_layer1, Nr, Nt), dtype=np.complex128)
    gradient_component = np.zeros((dimension_layer1, Nr, Nt), dtype=np.complex128)
    for index in range(dimension_layer1):
        bits = str(bin(index)[2:].zfill(4*Nt))
        s = bits2signals(bits)
        # s_conjugate_transpose = s.conj().T
        error = y - np.dot(H_hat,s)
        error_norm[index] = np.square(np.linalg.norm(error))
        gradient_component[index] = np.dot(error, s.conj().T)

    min_error_norm = np.min(error_norm)

    for index in range(dimension_layer1):
        value =  np.exp(-error_norm[index]+min_error_norm)
        output[index] = value
        gradients[index] = -value*(-gradient_component[index])
    # print(output)
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
    output = sum_prob_1/total_prob
    # calculate gradient components in layer2
    gradients = np.zeros(2**(4*Nt))
    # print(output)
    epsilon = 1e-10  # 为了防止log(0)的情况，添加一个小的常数
    output = np.clip(output, epsilon, 1. - epsilon)
    for ii in range(len(gradients)):
        # gradient1 = true_output/output
        # gradient2 = (np.ones(len(true_output))-true_output)/(np.ones(len(output))-output)
        for jj in range(4*Nt):
            gradient1 = true_output[jj]/output[jj]
            gradient2 = (1-true_output[jj])/(1-output[jj])
            gradient3 = A[jj][ii]/total_prob
            gradient4 = sum_prob_1[jj]/np.square(total_prob)
            # gradients for cross entropy
            gradients[ii] += (-1/(4*Nt))*(gradient1-gradient2)*(gradient3-gradient4)
            # gradients for MSE
            # gradients[ii] += (1/(4*Nt))*2*(output[jj]-true_output[jj])*(gradient3-gradient4)

    return output, gradients


def calculate_square_error(layer2_output, true_sequence):
    return np.linalg.norm(layer2_output-true_sequence)**2

def calculate_cross_entropy(layer2_output, true_sequence):
    epsilon = 1e-10  # 为了防止log(0)的情况，添加一个小的常数
    layer2_output = np.clip(layer2_output, epsilon, 1. - epsilon)
    cross_entropy = -np.mean(true_sequence * np.log(layer2_output) + (1 - true_sequence) * np.log(1 - layer2_output))
    return cross_entropy


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
        total_loss += calculate_cross_entropy(layer2_output,true_sequence)
        # SGD
        if np.random.rand() < 0.9:
            for jj in range(2**(4*Nt)):
                total_gradients += (layer2_gradients[jj]*layer1_gradients[jj])
    mean_loss = total_loss/training_length
    return mean_loss, total_gradients


def training(max_iter):
    # H_hat = np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt))
    H_hat = np.zeros((Nr,Nt), dtype=np.complex128)
    # H_hat = np.copy(H_w)
    momentum = np.zeros((Nr,Nt),dtype=np.complex128)
    last_loss = -100
    mean_loss = -200
    for iter_num in range(max_iter):
        # solve the gradient
        mean_loss, total_gradients = calculate_cost_function(H_hat)
        print("loss: "+str(mean_loss))
        if np.abs(last_loss-mean_loss) < 0.001:
            return H_hat, mean_loss
        else:
            last_loss = mean_loss
        # update H_hat
        momentum = (1-beta1)*total_gradients + beta1*momentum
        H_hat -= alpha * momentum
        # print(H_hat)
    return H_hat, mean_loss

# testing QNN for detection
def calculate_layer1_testing(H_hat, y):
    dimension_layer1 = 2**(4*Nt)
    # layer1 output
    output = np.zeros(dimension_layer1)
    error_norm = np.empty(dimension_layer1)
    for index in range(dimension_layer1):
        bits = str(bin(index)[2:].zfill(4*Nt))
        s = bits2signals(bits)
        error = y - np.dot(H_hat,s)
        error_norm[index] = np.square(np.linalg.norm(error))

    min_error_norm = np.min(error_norm)

    for index in range(dimension_layer1):
        value =  np.exp(-error_norm[index]+min_error_norm)
        output[index] = value
    return output


def calculate_layer2_testing(layer1_output):
    total_prob = np.sum(layer1_output)
    # print(total_prob)
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

for ii in range(len(SNR_list)):
    SNR_dB = SNR_list[ii]
    SNR = 10**(SNR_dB / 10)
    
    SD_performance = np.zeros(iter_num)
    QNN_performance_128 = np.zeros(iter_num)

    for jj in range(iter_num):
        print("----------------------------current SNR_dB: " +str(SNR_dB))
        print("----------------------------current iter num: " +str(jj))

        H = H_list[jj] * np.sqrt(SNR)
        cov = cov_list[0]

        bits_sequence_testing, x_sequence_testing, y_sequence_testing = generate_data(Nr,Nt,1024,H,cov)
        SD_performance[jj] = sphere_decoding_BER(H, y_sequence_testing, bits_sequence_testing, 1)
        print("SD: "+str(SD_performance[jj]))

        H_w = np.copy(H)

        bits_sequence, x_sequence, y_sequence = generate_data(Nr,Nt,pilot_length,H,cov)
        H_trained, loss = training(max_iter)
        
        
        BER = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)

        # save_BER[ii][jj] = BER

        QNN_performance_128[jj] = BER
        print("QNN: "+str(BER))

    SD_mean_performance[ii] = np.mean(SD_performance)
    QNN_mean_performance_128[ii] = np.mean(QNN_performance_128)