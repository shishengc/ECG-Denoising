import numpy as np
import yaml
from pathlib import Path
import _pickle as pickle

import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
from datasets.load_data import load_NSTDB, load_QT, load_SimEMG, load_Arrhythmia

def Data_Preparation(n_type=1):

    print('Getting the Data ready ... ')
    
    
    config_path = Path("./config") / "data_prepare.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Load QT Database
    with open(config['qt_path'], 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open(config['nstdb_path'], 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals,_,_] = nstdb
    #[_, em_signals, _ ] = nstdb
    #[_, _, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]


    #####################################
    # Data split
    #####################################
    if n_type == 1:
        noise_test = bw_noise_channel2_b
        noise_train = bw_noise_channel1_a
    elif n_type == 2:
        noise_test = bw_noise_channel1_b
        noise_train = bw_noise_channel2_a
    else:
        raise Exception("Sorry, n_type should be 1 or 2")

    #####################################
    # QTDatabase
    #####################################

    beats_train = []
    beats_test = []

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
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
    
    skip_beats = 0
    L = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(L)
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (L - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)
    
    sn_train = []
    sn_test = []

    noise_index = 0
    
    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + L]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += L

        if noise_index > (len(noise_train) - L):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100

    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save(config['rnd_test_path'], rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + L]
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        
        sn_test.append(signal_noise)
        noise_index += L

        if noise_index > (len(noise_test) - L):
            noise_index = 0


    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    
    X_test = np.array(sn_test)
    y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    print('Dataset ready to use.')

    return X_train, y_train, X_test, y_test
    

def Add_RMN(beats, bw_noise, em_noise, ma_noise, repeat=1, seed=1234, is_train=False):
    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    np.random.seed(seed=seed)
    
    L = 512
    beat_list = []
    sn_list = []
    noise_types = []
    snr_values = []
    
    if is_train:
        bw_ind, em_ind, ma_ind = (np.random.randint(0, len(bw_noise) - L), 
                                 np.random.randint(0, len(em_noise) - L), 
                                 np.random.randint(0, len(ma_noise) - L))
    else:
        bw_ind, em_ind, ma_ind = 0, 0, 0
    
    for i in range(len(beats)):
        for j in range(repeat):
            noise_type = np.random.randint(1, 8)
            noise_types.append(noise_type)

            snr_db = np.random.randint(-6, 19)   
            snr_values.append(snr_db)
            scale = 10**(snr_db / 10.0)

            noise = np.zeros(L)

            if noise_type & 1:
                bw_segment = bw_noise[bw_ind:bw_ind + L]
                noise += bw_segment
                bw_ind += L
                if bw_ind > (len(bw_noise) - L):
                    bw_ind = 0

            if noise_type & 2:
                em_segment = em_noise[em_ind:em_ind + L]
                noise += em_segment
                em_ind += L
                if em_ind > (len(em_noise) - L):
                    em_ind = 0

            if noise_type & 4:
                ma_segment = ma_noise[ma_ind:ma_ind + L]
                noise += ma_segment
                ma_ind += L
                if ma_ind > (len(ma_noise) - L):
                    ma_ind = 0

            signal_power = np.mean(beats[i]**2)
            noise_power = np.mean(noise**2)
        
            if noise_power > 0:
                alpha = np.sqrt(signal_power / (scale * noise_power))
                noise = noise * alpha

            noise = beats[i] + noise
            sn_list.append(noise)
            beat_list.append(beats[i])
    
    return sn_list, beat_list, noise_types, snr_values


def Data_Preparation_RMN(n_type=1):

    print('Getting the Data ready ... ')

    # Load QT Database
    beats_train, beats_test = load_QT()

    # Load NSTDB
    (bw_train, em_train, ma_train,
    bw_test, em_test, ma_test) = load_NSTDB(n_type)

    sn_train, _, train_types, train_snr = Add_RMN(
        beats_train, bw_train, em_train, ma_train)

    sn_test, _, test_types, test_snr = Add_RMN(
        beats_test, bw_test, em_test, ma_test)

    np.save('./Data/prepared/rnd_test.npy', test_snr)

    X_train, y_train = np.array(sn_train), np.array(beats_train)
    X_test, y_test = np.array(sn_test), np.array(beats_test)
    
    X_train, y_train = np.expand_dims(X_train, axis=2), np.expand_dims(y_train, axis=2)
    X_test, y_test = np.expand_dims(X_test, axis=2), np.expand_dims(y_test, axis=2)
        
    data_info = {
        'clean_beats': np.array(beats_train),
        'noise_types': np.array(train_types),
        'snr_values': np.array(train_snr),
        'train_noise': {
            'bw': bw_train,
            'em': em_train,
            'ma': ma_train                
        },
        'test_noise': {
            'bw': bw_test,
            'em': em_test,
            'ma': ma_test                
        }
    }

    return X_train, y_train, X_test, y_test, data_info


class ECGDataset(Dataset):
    def __init__(self, n_type=1, use_rmn=False, use_snr=False, config=None):
        self.n_type = n_type
        self.refresh = False
        self.use_rmn = use_rmn
        self.usr_snr = use_snr
        self.config = config
        self._prepare_data()
        
    def _prepare_data(self):
        if self.use_rmn:
            X_train, y_train, X_test, y_test, info = Data_Preparation_RMN(self.n_type)
        else:
           X_train, y_train, X_test, y_test = Data_Preparation(self.n_type)
           
        X_train, y_train = X_train.transpose(0, 2, 1), y_train.transpose(0, 2, 1)
        X_test, y_test = X_test.transpose(0, 2, 1), y_test.transpose(0, 2, 1)

        self.snr = torch.FloatTensor(info['snr_values']) if self.use_rmn else None
           
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        
    def _get_loader(self):
        if not self.usr_snr:
            train_val_set = TensorDataset(self.y_train, self.X_train)
        else:
            train_val_set = TensorDataset(self.y_train, self.X_train, self.snr)
        test_set = TensorDataset(self.y_test, self.X_test)
        
        train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
        train_set = Subset(train_val_set, train_idx)
        val_set = Subset(train_val_set, val_idx)
        
        train_loader = DataLoader(train_set, batch_size=self.config['train']['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.config['train']['batch_size'])
        test_loader = DataLoader(test_set, batch_size=self.config['test']['batch_size'])
        
        return train_loader, val_loader, test_loader


class EMGDataset(Dataset):
    def __init__(self, n_type=1, config=None, train=False):
        self.refresh = False
        self.n_type = n_type
        self.config = config
        self.train = train
        self.refresh = False
        (self.bw_train, self.em_train, self.ma_train,
        self.bw_test, self.em_test, self.ma_test) = load_NSTDB(self.n_type)
        
        self._prepare_data()
        
    def _apply_enhancement(self, X, y):
        np.random.seed(seed=1234)
        N, C, L = X.shape
        scales = np.random.uniform(0.5, 3.0, size=(N, 1, 1))
        seg_num = 8
        seg_len = L // seg_num
        flips = np.random.choice([0, 1], size=(N, seg_num))
        X_out = np.empty_like(X, dtype=float)
        y_out = np.empty_like(y, dtype=float)
        for n in range(N):
            s_scale = scales[n, 0, 0]
            for c in range(C):
                sig = X[n, c, :].astype(float)
                sigy = y[n, c, :].astype(float)
                out = np.empty_like(sig)
                outy = np.empty_like(sigy)
                for s in range(seg_num):
                    st = s * seg_len
                    ed = st + seg_len
                    seg = sig[st:ed]
                    segy = sigy[st:ed]
                    if flips[n, s]:
                        m = seg_len
                        t = np.arange(m)
                        baseline = seg[0] + (seg[-1] - seg[0]) * (t / (m - 1))
                        seg = 2 * baseline - seg
                        baseline_y = segy[0] + (segy[-1] - segy[0]) * (t / (m - 1))
                        segy = 2 * baseline_y - segy
                    out[st:ed] = seg
                    outy[st:ed] = segy
                    
                X_out[n, c, :] = out * s_scale
                y_out[n, c, :] = outy * s_scale
        
        return X_out, y_out

    def _prepare_data(self):
        X_train, y_train, X_test, y_test = load_SimEMG()

        X_train, y_train = X_train.transpose(0, 2, 1), y_train.transpose(0, 2, 1)
        X_test, y_test = X_test.transpose(0, 2, 1), y_test.transpose(0, 2, 1)

        if self.train:
            X_train, y_train, _, snr = Add_RMN(y_train, self.bw_train, self.em_train, self.ma_train, repeat=10, is_train=True)
            X_test, y_test, *_ = Add_RMN(y_test, self.bw_test, self.em_test, self.ma_test, repeat=10)
            
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_test, y_test = np.array(X_test), np.array(y_test)
            
            X_train, y_train = self._apply_enhancement(X_train, y_train)
            X_test, y_test = self._apply_enhancement(X_test, y_test)
            
            # self.snr = torch.FloatTensor(snr)
            
            self.X_train = torch.FloatTensor(X_train)
            self.y_train = torch.FloatTensor(y_train)
            self.X_test = torch.FloatTensor(X_test)
            self.y_test = torch.FloatTensor(y_test)

        else:
            self.X_train = torch.FloatTensor(X_train)
            self.y_train = torch.FloatTensor(y_train)
            self.X_test = torch.FloatTensor(X_test)
            self.y_test = torch.FloatTensor(y_test)
        
    def _get_loader(self):
        train_set = TensorDataset(self.y_train, self.X_train)
        test_set = TensorDataset(self.y_test, self.X_test)
        
        if self.train:
            train_loader = DataLoader(train_set, batch_size=self.config['train']['batch_size'], shuffle=True, drop_last=True)
            test_loader = DataLoader(test_set, batch_size=self.config['test']['batch_size'], shuffle=False)
            
            return train_loader, test_loader, test_loader
        else:
            test_set = test_set
            test_loader = DataLoader(test_set, batch_size=self.config['test']['batch_size'], shuffle=False)
            return test_loader
    
    
class ArrhythmiaDataset(Dataset):
    def __init__(self, n_type=1, config=None):
        self.refresh = False
        self.n_type = n_type
        self.config = config
        self.refresh = False
        (self.bw_train, self.em_train, self.ma_train,
        self.bw_test, self.em_test, self.ma_test) = load_NSTDB(self.n_type)
        
        self._prepare_data()
        
    def _prepare_data(self):
        X_test, *_ = load_Arrhythmia()
        
        X_test = X_test.transpose(0, 2, 1)
        
        X_test, y_test, *_ = Add_RMN(X_test, self.bw_test, self.em_test, self.ma_test)
           
        X_test, y_test = np.array(X_test), np.array(y_test)
        self.X_test, self.y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        
    def _get_loader(self):
        test_set = TensorDataset(self.y_test, self.X_test)
        test_loader = DataLoader(test_set, batch_size=64)
        
        return test_loader