import numpy as np
import yaml
from pathlib import Path
import _pickle as pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split

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
    samples = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
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
        noise = noise_train[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100

    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save(config['rnd_test_path'], rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        
        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
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

def Data_Preparation_RMN(n_type=1):

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
        bw_noise_test = bw_noise_channel2_b
        bw_noise_train = bw_noise_channel1_a
        em_noise_test = em_noise_channel2_b
        em_noise_train = em_noise_channel1_a
        ma_noise_test = ma_noise_channel2_b
        ma_noise_train = ma_noise_channel1_a
    elif n_type == 2:
        bw_noise_test = bw_noise_channel1_b
        bw_noise_train = bw_noise_channel2_a
        em_noise_test = em_noise_channel1_b
        em_noise_train = em_noise_channel2_a
        ma_noise_test = ma_noise_channel1_b
        ma_noise_train = ma_noise_channel2_a
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
    samples = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)

    def add_mixed_noise(beats, bw_noise, em_noise, ma_noise, is_train=False):
        """
        Add mixed noise combinations to ECG beats
        Args:
            beats: clean ECG beats
            bw_noise, em_noise, ma_noise: three types of noise
            is_train: whether this is for test set
        Returns:
            sn_list: noisy signals
            noise_combinations: noise type combinations used
            snr_values: SNR values used
            noise_list: detailed noise segments used
        """
        sn_list = []
        noise_combinations = []
        snr_values = []
        noise_list = []

        noise_index_bw = 0
        noise_index_em = 0
        noise_index_ma = 0
        
        for i in range(len(beats)):
            noise_type = np.random.randint(1, 8)
            noise_combinations.append(noise_type)
            
            # Randomly select SNR value (-6 to 18 dB)
            snr_db = np.random.randint(-6, 19)
            snr_values.append(snr_db)
            snr_linear = 10**(snr_db / 10.0)
            
            # Initialize mixed noise
            mixed_noise = np.zeros(samples)
            noise_comb = {'bw': None, 'em': None, 'ma': None}
            
            # Add BW noise if selected
            if noise_type & 1:
                bw_segment = bw_noise[noise_index_bw:noise_index_bw + samples]
                noise_comb['bw'] = bw_segment
                mixed_noise += bw_segment
                noise_index_bw += samples
                if noise_index_bw > (len(bw_noise) - samples):
                    noise_index_bw = 0
            
            # Add EM noise if selected
            if noise_type & 2:
                em_segment = em_noise[noise_index_em:noise_index_em + samples]
                noise_comb['em'] = em_segment
                mixed_noise += em_segment
                noise_index_em += samples
                if noise_index_em > (len(em_noise) - samples):
                    noise_index_em = 0
            
            # Add MA noise if selected
            if noise_type & 4:
                ma_segment = ma_noise[noise_index_ma:noise_index_ma + samples]
                noise_comb['ma'] = ma_segment
                mixed_noise += ma_segment
                noise_index_ma += samples
                if noise_index_ma > (len(ma_noise) - samples):
                    noise_index_ma = 0
            
            # Scale noise based on SNR
            signal_power = np.mean(beats[i]**2)
            noise_power = np.mean(mixed_noise**2)
            
            if noise_power > 0:
                noise_scaling = np.sqrt(signal_power / (snr_linear * noise_power))
                scaled_noise = mixed_noise * noise_scaling
            else:
                scaled_noise = mixed_noise
            
            # Generate noisy signal
            signal_noise = beats[i] + scaled_noise
            sn_list.append(signal_noise)
            noise_list.append(noise_comb)
        
        return sn_list, noise_combinations, snr_values, noise_list if is_train else None

    # Add mixed noise to training set
    sn_train, train_noise_combinations, train_snr_values, train_noise_list = add_mixed_noise(
        beats_train, bw_noise_train, em_noise_train, ma_noise_train, is_train=True)

    # Add mixed noise to test set
    sn_test, test_noise_combinations, test_snr_values, _ = add_mixed_noise(
        beats_test, bw_noise_test, em_noise_test, ma_noise_test, is_train=False)

    np.save(config['rnd_test_path'], test_snr_values)
    np.save(config['rnd_test_path'].replace('.npy', '_noise_combinations.npy'), test_noise_combinations)

    # Convert to numpy arrays and add channel dimension
    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    
    X_test = np.array(sn_test) # shape: N, L, C
    y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)
    
    train_info = {
        'clean_beats': np.array(beats_train),
        'noise_types': np.array(train_noise_combinations),
        'snr_values': np.array(train_snr_values),
        'mixed_noise': train_noise_list
    }

    return X_train, y_train, X_test, y_test, train_info


class ECGDataset(Dataset):
    def __init__(self, n_type=1, use_rmn=False, config=None):
        self.n_type = n_type
        self.refresh = False
        self.use_rmn = use_rmn
        self.config = config
        self._prepare_data()
        
    def _prepare_data(self):
        if self.use_rmn:
            X_train, y_train, X_test, y_test, _ = Data_Preparation_RMN(self.n_type)
        else:
           X_train, y_train, X_test, y_test = Data_Preparation(self.n_type)
           
        self.X_train = torch.FloatTensor(X_train).permute(0,2,1)
        self.y_train = torch.FloatTensor(y_train).permute(0,2,1)
        self.X_test = torch.FloatTensor(X_test).permute(0,2,1)
        self.y_test = torch.FloatTensor(y_test).permute(0,2,1)
        
    def _get_loader(self):
        train_val_set = TensorDataset(self.y_train, self.X_train)
        test_set = TensorDataset(self.y_test, self.X_test)
        
        train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3, random_state=42)
        train_set = Subset(train_val_set, train_idx)
        val_set = Subset(train_val_set, val_idx)
        
        train_loader = DataLoader(train_set, batch_size=self.config['train']['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.config['train']['batch_size'])
        test_loader = DataLoader(test_set, batch_size=self.config['test']['batch_size'])
        
        return train_loader, val_loader, test_loader
    
    
class Ada_ECGDataset(ECGDataset):
    def __init__(self, n_type=1, use_rmn=False, config=None):
        self.n_type = n_type
        self.refresh = True
        self.use_rmn = use_rmn
        self.config = config
        self.weights = np.array([1.0, 1.0, 1.0])
        
        self._prepare_data()
        
    def _prepare_data(self):
        if self.use_rmn:
            _, _, X_test, y_test, train_info = Data_Preparation_RMN(self.n_type)
            self.clean_beats = train_info['clean_beats']
            self.noise_types = train_info['noise_types']
            self.snr_values = train_info['snr_values']
            self.mixed_noise = train_info['mixed_noise']
        else:
           _, _, X_test, y_test = Data_Preparation(self.n_type)
           
        self.X_test = torch.FloatTensor(X_test).permute(0,2,1)
        self.y_test = torch.FloatTensor(y_test).permute(0,2,1)
        
    def __len__(self):
        return len(self.clean_beats)
    
    def __getitem__(self, idx):
        clean = self.clean_beats[idx]
        noise_type = self.noise_types[idx]
        snr_db = self.snr_values[idx]
        noise = self.mixed_noise[idx]
        
        mixed_noise = np.zeros(512)
        if noise_type & 1:
            mixed_noise += noise['bw'] * self.weights[0]
        if noise_type & 2:
            mixed_noise += noise['em'] * self.weights[1]
        if noise_type & 4:
            mixed_noise += noise['ma'] * self.weights[2]
        
        snr_linear = 10**(snr_db / 10.0)
        signal_power = np.mean(clean**2)
        noise_power = np.mean(mixed_noise**2)
        
        if noise_power > 0:
            noise_scaling = np.sqrt(signal_power / (snr_linear * noise_power))
            scaled_noise = mixed_noise * noise_scaling
        else:
            scaled_noise = mixed_noise
        
        noisy = clean + scaled_noise
        
        y_train = torch.FloatTensor(clean).unsqueeze(0)
        X_train = torch.FloatTensor(noisy).unsqueeze(0)
        
        return y_train, X_train
    
    def update_weights(self, weights):
        self.weights = weights

    def _get_loader(self):
        test_set = TensorDataset(self.y_test, self.X_test)
        
        train_idx, val_idx = train_test_split(list(range(len(self))), test_size=0.3)

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.val_noise_types = self.noise_types[val_idx]
        
        train_set = Subset(self, train_idx)
        val_set = Subset(self, val_idx)
        
        train_loader = DataLoader(train_set, batch_size=self.config['train']['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.config['train']['batch_size'])
        test_loader = DataLoader(test_set, batch_size=self.config['test']['batch_size'])

        return train_loader, val_loader, test_loader

    def update(self, loss_list, batch_size=64):
        noise_losses = [0, 0, 0]
        noise_counts = [0, 0, 0]
        
        for i, loss in enumerate(loss_list):
            noise_type = self.val_noise_types[i]
            if noise_type & 1:
                noise_losses[0] += loss
                noise_counts[0] += 1
            if noise_type & 2:
                noise_losses[1] += loss
                noise_counts[1] += 1
            if noise_type & 4:
                noise_losses[2] += loss
                noise_counts[2] += 1
        
        weights = F.softmax(torch.tensor([l/c for l, c in zip(noise_losses, noise_counts)]), dim=0).numpy()
        self.update_weights(weights)
        # print(f"Updated weights: BW={weights[0]:.3f}, EM={weights[1]:.3f}, MA={weights[2]:.3f}")
        
        train_set = Subset(self, self.train_idx)
        return DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)


class EMGDataset(Dataset):
    def __init__(self, config=None, augment=False):
        self.refresh = False
        self.augment = augment
        self.config = config
        self._prepare_data()
        
    def _prepare_data(self):
        print('Getting the EMG Data ready with noise augmentation ... ')
        
        config_path = Path("./config") / "data_prepare.yaml"
        with open(config_path, "r") as f:
            data_config = yaml.safe_load(f)
        
        seed = 1234
        np.random.seed(seed=seed)
        
        with open(data_config['nstdb_path'], 'rb') as input:
            nstdb = pickle.load(input)
        
        [bw_signals, _, _] = nstdb
        bw_signals = np.array(bw_signals)
        bw_noise = bw_signals[:, 0]
        
        from load_openset import load_SimEMG_Train
        X_train, y_train = load_SimEMG_Train()
        X_train, y_train = X_train.transpose(0, 2, 1), y_train.transpose(0, 2, 1)            
        samples = X_train.shape[2]
        
        X_augmented = []
        y_augmented = []
        
        noise_index = 0
        for i in range(len(X_train)):
            clean_signal = y_train[i, 0, :]

            for i in range(3):
                snr_db = np.random.randint(-6, 7)
                if i > 0:
                    while(snr_db==last_snr_db):
                        snr_db = np.random.randint(-6, 7)
                else:
                    last_snr_db = snr_db
                snr_linear = 10**(snr_db / 10.0)

                noise = bw_noise[noise_index:noise_index + samples]
                if len(noise) < samples:
                    noise_index = 0
                    noise = bw_noise[noise_index:noise_index + samples]
                signal_power = np.mean(clean_signal**2)
                noise_power = np.mean(noise**2)
                
                if noise_power > 0:
                    noise_scaling = np.sqrt(signal_power / (snr_linear * noise_power))
                    scaled_noise = noise * noise_scaling
                else:
                    scaled_noise = noise
                noisy_signal = clean_signal + scaled_noise

                y_augmented.append(clean_signal)
                X_augmented.append(noisy_signal)

                noise_index += samples
                if noise_index > (len(bw_noise) - samples):
                    noise_index = 0

        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)

        X_augmented = np.expand_dims(X_augmented, axis=1)
        y_augmented = np.expand_dims(y_augmented, axis=1)

        self.X_train = torch.FloatTensor(X_augmented)
        self.y_train = torch.FloatTensor(y_augmented)
        
        print(f'EMG Dataset ready with {len(self.X_train)} samples (original + augmented)')
        
    def _get_loader(self):
        train_val_set = TensorDataset(self.y_train, self.X_train)
        
        train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3, random_state=42)
        train_set = Subset(train_val_set, train_idx)
        val_set = Subset(train_val_set, val_idx)
        
        train_loader = DataLoader(train_set, batch_size=self.config['train']['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.config['train']['batch_size'])
        
        return train_loader, val_loader, val_loader