import ctypes
import numpy as np

def sphere_decoding_BER(H_hat, y_sequence, bits_sequence, maxL):
    
    lib = ctypes.CDLL("./sphere_decoding/SD")

    true_sequence = ''.join(bits_sequence[i] for i in range(len(bits_sequence)))
    bits_length = len(true_sequence)
    error_num = 0
    index = 0

    Qm = 4
    numTxPort = len(H_hat[0])
    numSymData = len(y_sequence)
    numRxAnt = len(y_sequence[0])

    for ii in range(numSymData):

        recVec = y_sequence[ii]

        H0 = np.transpose(H_hat)

        H1 = [0 for i in range(2*numRxAnt*numTxPort)]
        flag = 0
        for col in range(numRxAnt):
            for row in range(numTxPort):
                H1[2*flag] = np.real(H0[row][col])
                H1[2*flag+1] = np.imag(H0[row][col])
                flag += 1

        y1 = [0 for i in range(2*numRxAnt)]
        flag = 0
        for item in range(numRxAnt):
            y1[flag*2] = np.real(recVec[item])[0]
            y1[flag*2+1] = np.imag(recVec[item])[0]
            flag += 1

        p = [0 for i in range(numTxPort*Qm)]

        pLLR = (ctypes.c_float * (numTxPort*Qm))(*p)
        py = (ctypes.c_float * len(y1))(*y1)
        pH = (ctypes.c_float * len(H1))(*H1)

        LyrN = ctypes.c_int32(numTxPort)
        SCnum = ctypes.c_int32(1)
        ModuType = ctypes.c_int32(2**Qm)
        maxL0 = ctypes.c_float(maxL)

        lib.MIMO_mldDec(pLLR, py, pH, LyrN, SCnum, ModuType, maxL0)

        for jj in range(numTxPort*Qm):
            if (pLLR[jj] > 0 and true_sequence[index] == '0') or (pLLR[jj] < 0 and true_sequence[index] == '1'):
                error_num += 1
            index += 1
            # print(pLLR[jj])
    return error_num/bits_length