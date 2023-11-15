import numpy as np 
import random
from scipy.optimize import minimize

def generate_random_bit_sequence(length):
    return ''.join(random.choice('01') for _ in range(length))

def qam16_modulation(binary_input):
    mapping = {
        '0000': (-3-3j),
        '0001': (-3-1j),
        '0010': (-3+3j),
        '0011': (-3+1j),
        '0100': (-1-3j),
        '0101': (-1-1j),
        '0110': (-1+3j),
        '0111': (-1+1j),
        '1000': (3-3j),
        '1001': (3-1j),
        '1010': (3+3j),
        '1011': (3+1j),
        '1100': (1-3j),
        '1101': (1-1j),
        '1110': (1+3j),
        '1111': (1+1j)
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
    SNR = 10.0**(SNR_dB/10.0)
    n_sequence = [generate_noise(SNR, Nr) for i in range(length)]
    y_sequence = [np.dot(H_channel, x_sequence[i].reshape(Nt,1)) + n_sequence[i] for i in range(length)]
    return bits_sequence, x_sequence, y_sequence

def bits2signals(bits):
    # bits: input binary string with length of (4*Nt) 
    return np.array([qam16_modulation(bits[i:i+4]) for i in range(0, len(bits), 4)]).reshape(Nt,1)
def calculate_layer1(H_hat, y):
    dimension_layer1 = 2**(4*Nt)
    output = {}
    for index in range(dimension_layer1):
        bits = str(bin(index)[2:].zfill(4*Nt))
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
    for ii in range(training_length):
        layer1_output = calculate_layer1(H_hat, y_sequence[ii])
        layer2_output = calculate_layer2(layer1_output)
        true_sequence = ''.join(bits_sequence[ii*Nt+jj] for jj in range(Nt))
        total_loss += calculate_square_error(layer2_output,true_sequence)
    mean_loss = total_loss/training_length
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

def calculate_BER(H_trained):
    # tesing set
    testing_length = 1000
    bits_sequence_testing, x_sequence_testing, y_sequence_testing = generate_data(Nr,Nt,SNR,testing_length,H)
    error = 0
    for ii in range(len(y_sequence_testing)):
        detect_result = detection(y_sequence_testing[ii], H_trained)
        true_sequence = ''.join(bits_sequence_testing[ii*Nt+jj] for jj in range(Nt))
        error += count_differences(detect_result, true_sequence)
    BER = error/(len(y_sequence_testing)*len(detect_result))
    return BER

def training_testing():
    H_hat_vec = np.sqrt(1/2)*(np.random.randn(Nr*Nt*2))
    BER_performance = 1

    out = minimize(calculate_cost_function, x0=H_hat_vec, method="COBYLA", options={'maxiter':100})

    H_hat_vec = out.x

    H_trained = H_hat_vec[0:Nr*Nt].reshape(Nr,Nt)+1j*H_hat_vec[Nr*Nt:2*Nr*Nt].reshape(Nr,Nt)
    BER_performance = calculate_BER(H_trained)

    return H_trained, BER_performance

# generate training and tesing data
Nt = 2
Nr = 4
# generate channel
H = np.sqrt(1/2)*(np.random.randn(Nr,Nt)+1j*np.random.randn(Nr,Nt))


for training_length in [4, 8, 16, 32, 64, 128, 256]:
    iter_num = 5
    BER_list = np.zeros(iter_num)
    for ii in range(iter_num):
        bits_sequence, x_sequence, y_sequence = generate_data(Nr,Nt,10,training_length,H)
        H_trained, BER_list[ii] = training_testing()
    mean_BER = np.mean(BER_list)
    print([training_length, mean_BER])
