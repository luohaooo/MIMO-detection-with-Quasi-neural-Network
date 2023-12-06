'''
Random sampling of branches 
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


def sensing_field_for_one_dimension(random_num):
    num = int(random_num/2)
    result = []
    ls = random.sample(range(8),num)
    for ii in range(num):
        result.append(bin(ls[ii])[2:].zfill(4))
        result.append(bin(15-ls[ii])[2:].zfill(4))
    return result


def sensing_field_for_high_dimension(Nt, random_num):
    if Nt == 1:
        return sensing_field_for_one_dimension(random_num)
    else:
        previous_sensing_field = sensing_field_for_high_dimension(Nt-1, random_num)
        this_dimension_sensing_field =  sensing_field_for_one_dimension(random_num)
        new_sensing_field = []
        for ii in range(len(previous_sensing_field)):
            for jj in range(random_num):
                new_sensing_field.append(previous_sensing_field[ii]+this_dimension_sensing_field[jj])
        return new_sensing_field


def calculate_layer1(H_hat, y):
    output = {}
    for bits in sensing_field:
        s = bits2signals(bits)
        error = y - np.dot(H_hat,s)
        value =  np.exp(-np.square(np.linalg.norm(error)))
        output[bits] = value
    return output

def calculate_layer2(layer1_output):
    sum_exp = [[0 for i in range(2)] for j in range(4*Nt)]
    for bits in layer1_output:
        value = layer1_output[bits]
        for index in range(4*Nt):
            sum_exp[index][eval(bits[index])] += value
    output = {}
    for index in range(4*Nt):
        # llr = np.log(sum_exp[index][1]/sum_exp[index][0])
        output[index] = (sum_exp[index][1])/(sum_exp[index][1]+sum_exp[index][0])
    return output

def calculate_cross_entropy(layer2_output, true_sequence):
    dimension = len(true_sequence)
    entropy = 0
    for index in range(dimension):
        if true_sequence[index] == '1':
            entropy += (-np.log(layer2_output[index]))
    return entropy

def calculate_square_error(layer2_output, true_sequence):
    dimension = len(true_sequence)
    loss = 0
    for index in range(dimension):
        if true_sequence[index] == '1':
            loss += np.square(1-layer2_output[index])
        else:
            loss += np.square(layer2_output[index])
    return loss

def calculate_cost_function(H_hat_vec):
    H_hat = H_hat_vec[0:Nr*Nt].reshape(Nr,Nt)+1j*H_hat_vec[Nr*Nt:2*Nr*Nt].reshape(Nr,Nt)
    # H_hat = H_hat_vec
    total_loss = 0
    training_length = len(y_sequence)
    for ii in range(training_length):
        layer1_output = calculate_layer1(H_hat, y_sequence[ii])
        layer2_output = calculate_layer2(layer1_output)
        true_sequence = ''.join(bits_sequence[ii*Nt+jj] for jj in range(Nt))
        total_loss += calculate_square_error(layer2_output,true_sequence)
    mean_loss = total_loss/(training_length)

    # print(mean_loss)
    return mean_loss
        
def detection(y, H_trained):
    layer1_output = calculate_layer1(H_trained, y)
    layer2_output = calculate_layer2(layer1_output)
    detect_result = ''
    for ii in range(len(layer2_output)):
        if(layer2_output[ii]>0.5):
            detect_result += '1'
        else:
            detect_result += '0'
    return(detect_result)

def count_differences(str1, str2):
    return sum(a != b for a, b in zip(str1, str2))


def training():
    H_hat_vec = np.sqrt(1/2)*(np.random.randn(Nr*Nt*2))

    out = minimize(calculate_cost_function, x0=H_hat_vec, method="COBYLA", options={'maxiter':300,'catol':1e-3})

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

iter_num = 5
SNR_list = np.array([15])

SD_mean_performance = np.zeros(len(SNR_list))
QNN_mean_performance_2 = np.zeros(len(SNR_list))
QNN_mean_performance_6 = np.zeros(len(SNR_list))
QNN_mean_performance_10 = np.zeros(len(SNR_list))
QNN_mean_performance_14 = np.zeros(len(SNR_list))
QNN_mean_performance_16 = np.zeros(len(SNR_list))

H_list = [np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt)) for ii in range(iter_num)]

for ii in range(len(SNR_list)):
    SNR_dB = SNR_list[ii]
    print("SNR_dB: "+str(SNR_dB))
    
    SD_performance = np.zeros(iter_num)
    QNN_performance_2 = np.zeros(iter_num)
    QNN_performance_6 = np.zeros(iter_num)
    QNN_performance_10 = np.zeros(iter_num)
    QNN_performance_14 = np.zeros(iter_num)
    QNN_performance_16 = np.zeros(iter_num)


    for jj in range(iter_num):
        print("current iter num: " +str(jj))
        H = H_list[jj]
        # print(H)
        bits_sequence_testing, x_sequence_testing, y_sequence_testing = generate_data(Nr,Nt,SNR_dB,1024,H)
        SD_performance[jj] = sphere_decoding_BER(H, y_sequence_testing, bits_sequence_testing, 10)
        print("SD: "+str(SD_performance[jj]))

        for sampling_num in [2,6,10,14,16]:
            sensing_field = sensing_field_for_high_dimension(Nt, sampling_num)

            bits_sequence, x_sequence, y_sequence = generate_data(Nr,Nt,SNR_dB,128,H)
            H_trained = training()
            print("sampling_num: "+str(sampling_num))
            if sampling_num == 2:
                QNN_performance_2[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
                print("QNN: "+str(QNN_performance_2[jj]))
            if sampling_num == 6:
                QNN_performance_6[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
                print("QNN: "+str(QNN_performance_6[jj]))
            if sampling_num == 10:
                QNN_performance_10[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
                print("QNN: "+str(QNN_performance_10[jj]))
            if sampling_num == 14:
                QNN_performance_14[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
                print("QNN: "+str(QNN_performance_14[jj]))
            if sampling_num == 16:
                QNN_performance_16[jj] = calculate_BER(H_trained, bits_sequence_testing, y_sequence_testing)
                print("QNN: "+str(QNN_performance_16[jj]))
        

    SD_mean_performance[ii] = np.mean(SD_performance)
    QNN_mean_performance_2[ii] = np.mean(QNN_performance_2)
    QNN_mean_performance_6[ii] = np.mean(QNN_performance_6)
    QNN_mean_performance_10[ii] = np.mean(QNN_performance_10)
    QNN_mean_performance_14[ii] = np.mean(QNN_performance_14)
    QNN_mean_performance_16[ii] = np.mean(QNN_performance_16)

print("sd: "+str(SD_mean_performance))
print("qnn_2: "+str(QNN_mean_performance_2))
print("qnn_6: "+str(QNN_mean_performance_6))
print("qnn_10: "+str(QNN_mean_performance_10))
print("qnn_14: "+str(QNN_mean_performance_14))
print("qnn_16: "+str(QNN_mean_performance_16))





