import numpy as np
import pickle
import os

def load_NSTDB(n_type=1):
    with open('./Data/QT/noise.pkl', 'rb') as input:
        nstdb = pickle.load(input)
    
    #####################################
    # NSTDB - Extract three types of noise
    #####################################

    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)
    
    # BW noise extraction
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]
    
    # EM noise extraction
    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]
    
    # MA noise extraction
    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]

    #####################################
    # Data split
    #####################################
    if n_type == 1:
        bw_test = bw_noise_channel2_b
        bw_train = bw_noise_channel1_a
        em_test = em_noise_channel2_b
        em_train = em_noise_channel1_a
        ma_test = ma_noise_channel2_b
        ma_train = ma_noise_channel1_a
    elif n_type == 2:
        bw_test = bw_noise_channel1_b
        bw_train = bw_noise_channel2_a
        em_test = em_noise_channel1_b
        em_train = em_noise_channel2_a
        ma_test = ma_noise_channel1_b
        ma_train = ma_noise_channel2_a
    else:
        raise Exception("Sorry, n_type should be 1 or 2")
    
    return bw_train, em_train, ma_train, bw_test, em_test, ma_test


def load_QT():
    with open('./Data/QT/data.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)
    
    #####################################
    # QTDatabase
    #####################################
    beats_train = []
    beats_test = []

    test_set = [
                'sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database
                
                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database
                
                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                
                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database
                
                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database
                
                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH
                
                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]

    L = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(L)
            b_sq = np.array(b)
            
            init_padding = 0
            if b_sq.shape[0] > (L - init_padding):
                continue
            
            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)
                
    return beats_train, beats_test


def load_SimEMG():
    with open('./Data/SimEMG/data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_clean = data['train']['clean'] / 200
    train_noisy = data['train']['noisy'] / 200
    
    val_clean = data['val']['clean'] / 200 
    val_noisy = data['val']['noisy'] / 200
    
    X_train = np.expand_dims(train_noisy, axis=2)
    y_train = np.expand_dims(train_clean, axis=2)
    X_val = np.expand_dims(val_noisy, axis=2)
    y_val = np.expand_dims(val_clean, axis=2)
    
    return X_train, y_train, X_val, y_val


def load_Icentiak11():
    with open('./Data/Icentiak11/data.pkl', 'rb') as f:
        data = pickle.load(f)

        X_test = []
        ids = []

        for ind, (key, value) in enumerate(data.items()):
            ids.append(key)
            segment = np.reshape(value, (-1, 1, 512))
            X_test.append(segment)
        
        
        X_test = np.concatenate(X_test, axis=0)        
        X_test = np.array(X_test)
        X_test = np.transpose(X_test, (0, 2, 1))

        return X_test, ids


def load_Arrhythmia(data_path):
    data_path = './Data/Arrhythmia'
    file_paths = []
    ids = []
    cnts = []
    
    # delete records which has been included in QT Dataset
    delete_ids = ['100', '102', '103', '104', '114', '116', '117', '123', '213', '221', '223', '230', '231', '232', '233']
    
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                file_id = file.split('_')[0]
                if file_id not in delete_ids:
                    ids.append(file_id)
                    file_paths.append(os.path.join(root, file))
                
    sorted_pairs = sorted(zip(ids, file_paths), key=lambda x: x[0])
    ids, file_paths = zip(*sorted_pairs)
    ids = list(ids)

    X_test = []
    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True)
        data = np.concatenate((data[:, :, 0:1], data[:, :, 1:2]), axis=0)
        
        cnts.append(data.shape[0])
        X_test.append(data)
        
    X_test = np.concatenate(X_test, axis=0)
    
    return X_test, ids, cnts


def load_CPSC():
    data = np.load('./Data/CPSC/data.npy', allow_pickle=True).item()
    
    ids = []
    infos = []
    X_test = []
    
    for key, value in data.items():
        ids.append(key)
        infos.append({'label': value['label'], 'length': value['length'], 'shape': value['data'].shape[0]})
        X_test.append(value['data'])
        
    X_test = np.concatenate(X_test, axis=0).transpose(0, 2, 1)
    
    return X_test, ids, infos