import argparse
import torch
import numpy as np
import yaml
import csv
import os
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from datasets.load_data import load_Icentiak11, load_Arrhythmia, load_CPSC
from datasets.dataset import Data_Preparation, Data_Preparation_RMN, ECGDataset, EMGDataset, ArrhythmiaDataset
from utils.metrics import SSD, PRD, SNR_improvement

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def load_pretrained_model(args, config, foldername):
    # DeScoD-ECG
    if args.exp_name == "DeScoD":
        from models.generation_filters.DeScoD_model import ConditionalModel
        from models.generation_filters.DeScoD_diffusion import DDPM
        
        base_model = ConditionalModel(config['train']['feats']).to(args.device)
        model = DDPM(base_model, config, args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))

    # EDDM
    elif (args.exp_name == "EDDM"):
        from models.generation_filters.EDDM_model import UnetRes
        from models.generation_filters.EDDM_diffusion import ResidualDiffusion
        
        base_model = UnetRes(**config['base_model']).to(args.device)
        model = ResidualDiffusion(model=base_model, **config['diffusion']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))

    # FlowMatching
    elif (args.exp_name == "ARiRGen"):        
        #######################################  ARiR-Gen ########################################
        from models.generation_filters.FlowBackbone import Unet
        from models.generation_filters.FlowMatching import CFM
        from models.dl_filters.AR import AR

        base_model = Unet(**config['base_model'])
        autoencoder = AR().to(args.device)

        model = CFM(base_model=base_model, autoencoder=autoencoder, **config['flow'])
        model_path = foldername + args.model_path
        checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint)
        
        model = model.to(args.device)
        #############################################################################################
        
    # DRNN
    elif args.exp_name == "DRNN":
        from models.dl_filters.DRNN import DRDNN
        model = DRDNN(**config['model']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
        
    # FCN_DAE
    elif args.exp_name == "FCN_DAE":
        from models.dl_filters.FCN_DAE import FCN_DAE
        model = FCN_DAE(filters=config['model']['filters']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    
    # ACDAE
    elif args.exp_name == "ACDAE":
        from models.dl_filters.ACDAE import ACDAE
        model = ACDAE(in_channels=config['model']['in_channels']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
        
    # CBAM_DAE
    elif args.exp_name == "CBAM_DAE":
        from models.dl_filters.CBAM_DAE import AttentionSkipDAE2
        model = AttentionSkipDAE2(**config['model']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    
    # TCDAE
    elif (args.exp_name == "TCDAE"):
        from models.dl_filters.TCDAE import TCDAE
        model = TCDAE(**config['model']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
                
    # DeepFilter
    elif args.exp_name == "DeepFilter":
        from models.dl_filters.DeepFilter import DeepFilterModelLANLDilated
        model = DeepFilterModelLANLDilated().to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
        
    # ECG_GAN
    elif args.exp_name == "ECG_GAN":
        from models.generation_filters.ECGAN import Generator
        model = Generator(input_channels=config['generator']['feats']).to(args.device)
        model_path = foldername + "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))

    # AR
    elif args.exp_name == "AR":
        from models.dl_filters.AR import AR
        model = AR().to(args.device)
        model_path = foldername + args.model_path
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))

    return model


def dl_evaluate(args, model, noisy_batch):
    if args.exp_name == "DeScoD":
        shots = args.shots
        if shots > 1:
            denoised_batch = 0
            for _ in range(shots):
                denoised_batch += model.denoising(noisy_batch)
            denoised_batch = denoised_batch / shots
        else:
            denoised_batch = model.denoising(noisy_batch)

    elif args.exp_name == "EDDM":
        shots = args.shots
        if shots > 1:
            denoised_batch = 0
            for _ in range(shots):
                [_, denoised] = model.sample([noisy_batch, 0], batch_size=noisy_batch.shape[0])
                denoised_batch += denoised
            denoised_batch = denoised_batch / shots
        else:
            [_, denoised_batch] = model.sample([noisy_batch, 0], batch_size=noisy_batch.shape[0])
            
    elif args.exp_name == "ARiRGen":
        shots = args.shots
        if shots > 1:
            denoised_batch = 0
            for _ in range(shots):
                [denoised, _] = model.sample(noisy_batch, self_cond=noisy_batch)
                denoised_batch += denoised
            denoised_batch = denoised_batch / shots
        else:
            [denoised_batch, _] = model.sample(noisy_batch, self_cond=noisy_batch)
        
    elif args.exp_name == "ECG_GAN":
        batch_size = noisy_batch.shape[0]
        z = torch.randn(batch_size, 512, 8).to(args.device)
        denoised_batch = model(noisy_batch, z)
        
    elif args.exp_name == "AR":
        denoised_batch = model.inference(noisy_batch)

    else:
        denoised_batch = model(noisy_batch)

    return denoised_batch


def evaluate_QT(args):
    if args.train_set == 'SimEMG':
        save_path = f"results/EMG_Train/{args.exp_name}"
    else:
        save_path = f"results/{args.dataset}/{args.exp_name}"
    
    os.makedirs(save_path, exist_ok=True)
    all_metrics = {}
    all_noise_levels = []
    
    # Load Data
    for n_type in [1, 2]:
        if args.use_rmn:
            [_, _, X_test, y_test, _] = Data_Preparation_RMN(n_type)
        else:
            [_, _, X_test, y_test] = Data_Preparation(n_type)
            
        try:
            noise_level = np.load('./Data/prepared/rnd_test.npy')
            all_noise_levels.append(noise_level)
        except FileNotFoundError:
            print(f"Warning: rnd_test.npy not found for noise type {n_type}")
        
        # FIR & IIR filters
        if args.exp_name in ["FIR", "IIR"]:
            if args.exp_name == "FIR":
                from models.digital_filters.FIR_filter import FIR_test_Dataset
                [x_input, y_true, y_pred] = FIR_test_Dataset([None, None, X_test, y_test])
            elif args.exp_name == "IIR":
                from models.digital_filters.IIR_filter import IIR_test_Dataset
                [x_input, y_true, y_pred] = IIR_test_Dataset([None, None, X_test, y_test])

            x_input = x_input.transpose(0, 2, 1)
            y_true = y_true.transpose(0, 2, 1)
            y_pred = y_pred.transpose(0, 2, 1)

            results = {
                'y_pred': y_pred,
                'y_true': y_true,
                'x_input': x_input
            }
            np.save(f"results/{args.dataset}/{args.exp_name}/results_{n_type}.npy", results)
            
        # DL_filters
        else:
            config_path = Path("./config") / f"{args.exp_name}.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            foldername = f"./check_points/{args.train_set}/{args.exp_name}/noise_type_" + str(n_type) + "/"
        
            X_test = torch.FloatTensor(X_test).permute(0, 2, 1)         
            y_test = torch.FloatTensor(y_test).permute(0, 2, 1)
            
            test_set = TensorDataset(y_test, X_test)
            test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'])

            print(f"Loading {args.exp_name} model for noise type {n_type}...")
            model = load_pretrained_model(args, config, foldername)
            model.eval()

            print(f"Evaluating {args.exp_name} model for noise type {n_type}...")
            y_pred, y_true, x_input = [], [], []
            with torch.no_grad():
                for clean_batch, noisy_batch in tqdm(test_loader):
                    clean_batch, noisy_batch = clean_batch.to(args.device), noisy_batch.to(args.device)

                    denoised_batch = dl_evaluate(args, model, noisy_batch)

                    y_pred.append(denoised_batch.cpu().numpy())
                    y_true.append(clean_batch.cpu().numpy())
                    x_input.append(noisy_batch.cpu().numpy())
            
            y_pred = np.concatenate(y_pred, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            x_input = np.concatenate(x_input, axis=0)

            if args.save:
                results = {
                    'y_pred': y_pred,
                    'y_true': y_true,
                    'x_input': x_input
                }
                np.save(f"{save_path}/results_{n_type}_{args.save_path}.npy", results)

        metrics = {
            "SSD": SSD(y_true, y_pred),
            "PRD": PRD(y_true, y_pred),
            "ImSNR": SNR_improvement(x_input, y_pred, y_true)
        }

        if n_type == 1:
            all_metrics = {k: v for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] = np.concatenate([all_metrics[k], v])
                
        if args.type == 1:
            break

    metrics_stats = {}
    for name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        metrics_stats[name] = f"{mean_val:.3f}±{std_val:.3f}"
        
    if all_noise_levels:
        n_level = np.concatenate(all_noise_levels)
        segs = [-6, 0, 6, 12, 18]
        segmented_results = {}
        
        for name in all_metrics:
            segmented_results[name] = {}
            for idx_seg in range(len(segs) - 1):
                seg_label = f"{segs[idx_seg]}-{segs[idx_seg+1]}dB"
                idx = np.where(np.logical_and(n_level >= segs[idx_seg], n_level <= segs[idx_seg + 1]))[0]
                
                if len(idx) == 0:
                    segmented_results[name][seg_label] = "N/A"
                    continue
                
                seg_values = all_metrics[name][idx]
                mean_val = np.mean(seg_values)
                std_val = np.std(seg_values)
                segmented_results[name][seg_label] = f"{mean_val:.3f}±{std_val:.3f}"

    with open(f"{save_path}/Metrics_{args.exp_name}.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        headers = ["Dataset", "Model", "ImSNR (dB) ↑", "SSD (au) ↓", "PRD (%) ↓"]
        writer.writerow(headers)
        writer.writerow([args.dataset] + [f"{args.exp_name}"] + [metrics_stats[m] for m in ["ImSNR", "SSD", "PRD"]])
        
        writer.writerow([])

        if all_noise_levels:
            seg_labels = [f"{segs[i]}-{segs[i+1]}dB" for i in range(len(segs)-1)]
            writer.writerow(["Metrics"] + seg_labels)
            
            for metric in ["SSD", "PRD", "ImSNR"]:
                row = [metric]
                for seg_label in seg_labels:
                    if seg_label in segmented_results[metric]:
                        row.append(segmented_results[metric][seg_label])
                    else:
                        row.append("N/A")
                writer.writerow(row)
    
    print(f"Resultes saved to ./{save_path}/Metrics_{args.exp_name}.csv")


def evaluate_SimEMG(args):
    if args.train_set == 'SimEMG':
        save_path = f"results/EMG_Train/{args.exp_name}"
    else:
        save_path = f"results/{args.dataset}/{args.exp_name}"
    os.makedirs(save_path, exist_ok=True)

    # FIR & IIR filters
    if args.exp_name in ["FIR", "IIR"]:
        dataset = EMGDataset(n_type=1, config=None, train=True)
        X_test, y_test = np.array(dataset.X_test).transpose(0, 2, 1), np.array(dataset.y_test).transpose(0, 2, 1)
        
        if args.exp_name == "FIR":
            from models.digital_filters.FIR_filter import FIR_test_Dataset
            [x_input, y_true, y_pred] = FIR_test_Dataset([None, None, X_test, y_test])
        elif args.exp_name == "IIR":
            from models.digital_filters.IIR_filter import IIR_test_Dataset
            [x_input, y_true, y_pred] = IIR_test_Dataset([None, None, X_test, y_test])

        x_input = x_input.transpose(0, 2, 1)
        y_true = y_true.transpose(0, 2, 1)
        y_pred = y_pred.transpose(0, 2, 1)

        results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'x_input': x_input
        }
        np.save(f"{save_path}/results_1.npy", results)

        metrics = {
            "SSD": SSD(y_true, y_pred),
            "PRD": PRD(y_true, y_pred),
            "ImSNR": SNR_improvement(x_input, y_pred, y_true)
        }

        metrics_stats = {name: f"{np.mean(values):.3f}±{np.std(values):.3f}" for name, values in metrics.items()}

        with open(f"{save_path}/Metrics_{args.exp_name}.csv", 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            headers = ["Dataset", "Model", "ImSNR (dB) ↑", "SSD (au) ↓", "PRD (%) ↓"]
            writer.writerow(headers)
            writer.writerow([args.dataset] + [f"{args.exp_name}"] + [metrics_stats[m] for m in ["ImSNR", "SSD", "PRD"]])


    # DL_filters
    else:
        all_metrics = {}
        for n_type in [1, 2]:
            config_path = Path("./config") / f"{args.exp_name}.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            foldername = f"./check_points/{args.train_set}/{args.exp_name}/noise_type_" + str(n_type) + "/"
            
            dataset = EMGDataset(n_type=n_type, config=config, train=False)
            test_loader = dataset._get_loader()

            print(f"Loading {args.exp_name} model for noise type {n_type}...")
            model = load_pretrained_model(args, config, foldername)
            model.eval()

            print(f"Evaluating {args.exp_name} model for noise type {n_type}...")
            y_pred, y_true, x_input = [], [], []
            with torch.no_grad():
                for clean_batch, noisy_batch in tqdm(test_loader):
                    clean_batch, noisy_batch = clean_batch.to(args.device), noisy_batch.to(args.device)

                    denoised_batch = dl_evaluate(args, model, noisy_batch)

                    y_pred.append(denoised_batch.cpu().numpy())
                    y_true.append(clean_batch.cpu().numpy())
                    x_input.append(noisy_batch.cpu().numpy())

            y_pred = np.concatenate(y_pred, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            x_input = np.concatenate(x_input, axis=0)

            if args.save:
                results = {
                    'y_pred': y_pred,
                    'y_true': y_true,
                    'x_input': x_input
                }
                np.save(f"{save_path}/results_{n_type}_{args.save_path}.npy", results)

            metrics = {
                "SSD": SSD(y_true, y_pred),
                "PRD": PRD(y_true, y_pred),
                "ImSNR": SNR_improvement(x_input, y_pred, y_true)
            }

            if n_type == 1:
                all_metrics = {k: v for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    all_metrics[k] = np.concatenate([all_metrics[k], v])
                    
            if args.type == 1:
                break

        metrics_stats = {}
        for name, values in all_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            metrics_stats[name] = f"{mean_val:.3f}±{std_val:.3f}"

        with open(f"{save_path}/Metrics_{args.exp_name}.csv", 'w', newline='') as f:
            import csv
            writer = csv.writer(f)

            headers = ["Dataset", "Model", "ImSNR (dB) ↑", "SSD (au) ↓", "PRD (%) ↓"]
            writer.writerow(headers)
            writer.writerow([args.dataset] + [f"{args.exp_name}"] + [metrics_stats[m] for m in ["ImSNR", "SSD", "PRD"]])

        print(f"Evaluation complete for {args.exp_name}")
        print(f"Resultes saved to ./{save_path}/Metrics_{args.exp_name}.csv")


def evaluate_Icentiak11(args):
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    X_test, ids = load_Icentiak11()

    if args.exp_name in ["FIR", "IIR"]:
        if args.exp_name == "FIR":
            from models.digital_filters.FIR_filter import FIR_test_Dataset
            [X_test, _, y_pred] = FIR_test_Dataset([None, None, X_test, None])
        elif args.exp_name == "IIR":
            from models.digital_filters.IIR_filter import IIR_test_Dataset
            [X_test, _, y_pred] = IIR_test_Dataset([None, None, X_test, None])

        y_pred = y_pred.transpose(0, 2, 1)

        results = {
            'y_pred': y_pred
        }
        np.save(f"results/{args.dataset}/{args.exp_name}/Denoised.npy", results)

    else:
        n_type = 1
        config_path = Path("./config") / f"{args.exp_name}.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        foldername = f"./check_points/{args.train_set}/{args.exp_name}/noise_type_" + str(n_type) + "/"

        X_test = torch.FloatTensor(X_test).permute(0, 2, 1)
        test_set = TensorDataset(X_test)
        test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'], shuffle=False)

        print(f"Loading {args.exp_name} model for noise type {n_type}...")
        model = load_pretrained_model(args, config, foldername)
        model.eval()

        y_pred = []
        x_input = []

        print(f"Evaluating {args.exp_name} model for noise type {n_type}...")
        with torch.no_grad():
            for noisy_batch, in tqdm(test_loader):
                noisy_batch = noisy_batch.to(args.device)
                
                denoised_batch = dl_evaluate(args, model, noisy_batch)
                
                y_pred.append(denoised_batch.cpu().numpy())
                x_input.append(noisy_batch.cpu().numpy())
            
        y_pred = np.concatenate(y_pred, axis=0) 
        x_input = np.concatenate(x_input, axis=0)
        
        y_pred = np.reshape(y_pred, (len(ids), -1, 1, 512))
        x_input = np.reshape(x_input, (len(ids), -1, 1, 512))
        
        if args.save:
            results = {}
            for id in ids:
                results[id] = {
                    'y_pred': y_pred[ids.index(id)].reshape(-1),
                    'x_input': x_input[ids.index(id)].reshape(-1)
                }
            np.save(f"results/{args.dataset}/Denoised_{args.exp_name}_{args.save_path}.npy", results)
            

def evaluate_Arrhythmia(args):
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    X_test, ids, cnts = load_Arrhythmia()

    if args.exp_name in ["FIR", "IIR"]:
        if args.exp_name == "FIR":
            from models.digital_filters.FIR_filter import FIR_test_Dataset
            [X_test, _, y_pred] = FIR_test_Dataset([None, None, X_test, None])
        elif args.exp_name == "IIR":
            from models.digital_filters.IIR_filter import IIR_test_Dataset
            [X_test, _, y_pred] = IIR_test_Dataset([None, None, X_test, None])

        y_pred = y_pred.transpose(0, 2, 1)

        results = {}
        start_idx = 0
        
        for id, N in zip(ids, cnts):
            end_idx = start_idx + N

            y = y_pred[start_idx:end_idx]
            y = np.concatenate((y[:N//2, :, :], y[N//2:, :, :]), axis=1)    
            
            results[id] = y
            
            start_idx = end_idx
            
        np.save(f"results/{args.dataset}/{args.exp_name}/Denoised.npy", results)

    else:
        n_type = 1
        config_path = Path("./config") / f"{args.exp_name}.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        foldername = f"./check_points/{args.train_set}/{args.exp_name}/noise_type_" + str(n_type) + "/"
        dataset = ArrhythmiaDataset()
        test_loader = dataset._get_loader()

        print(f"Loading {args.exp_name} model for noise type {n_type}...")
        model = load_pretrained_model(args, config, foldername)
        model.eval()

        y_pred = []
        x_input = []

        print(f"Evaluating {args.exp_name} model for noise type {n_type}...")
        with torch.no_grad():
            for clean_bacth, noisy_batch, in tqdm(test_loader):
                noisy_batch = noisy_batch.to(args.device)
                
                denoised_batch = dl_evaluate(args, model, noisy_batch)
                
                y_pred.append(denoised_batch.cpu().numpy())
                x_input.append(noisy_batch.cpu().numpy())
            
        y_pred = np.concatenate(y_pred, axis=0) 
        x_input = np.concatenate(x_input, axis=0)
        
        results = {}
        start_idx = 0
        
        for id, N in zip(ids, cnts):
            end_idx = start_idx + N

            y = y_pred[start_idx:end_idx]
            y = np.concatenate((y[:N//2, :, :], y[N//2:, :, :]), axis=1)
            y = np.concatenate((y[:, 0, :].reshape(-1, 1), y[:, 1, :].reshape(-1, 1)), axis=1) 
            
            results[id] = y
            
            start_idx = end_idx
        
        np.save(f"results/{args.dataset}/Denoised_{args.exp_name}.npy", results)
        

def evaluate_CPSC(args):
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    X_test, ids, infos = load_CPSC()

    if args.exp_name in ["FIR", "IIR"]:
        if args.exp_name == "FIR":
            from models.digital_filters.FIR_filter import FIR_test_Dataset
            [X_test, _, y_pred] = FIR_test_Dataset([None, None, X_test, None])
        elif args.exp_name == "IIR":
            from models.digital_filters.IIR_filter import IIR_test_Dataset
            [X_test, _, y_pred] = IIR_test_Dataset([None, None, X_test, None])

        y_pred = y_pred.transpose(0, 2, 1)

        results = {
            'y_pred': y_pred
        }
        np.save(f"results/{args.dataset}/{args.exp_name}/Denoised.npy", results)

    else:
        n_type = 1
        config_path = Path("./config") / f"{args.exp_name}.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        foldername = f"./check_points/{args.train_set}/{args.exp_name}/noise_type_" + str(n_type) + "/"
        print(f"Loading {args.exp_name} model for noise type {n_type}...")
        model = load_pretrained_model(args, config, foldername)
        model.eval()

        X_test = torch.FloatTensor(X_test).permute(0, 2, 1)
        test_set = TensorDataset(X_test)
        test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'], shuffle=False)
        
        y_pred = []
        x_input = []

        with torch.no_grad():
            for noisy_batch, in tqdm(test_loader):
                noisy_batch = noisy_batch.to(args.device)
                
                denoised_batch = dl_evaluate(args, model, noisy_batch)
                
                y_pred.append(denoised_batch.cpu().numpy())
                x_input.append(noisy_batch.cpu().numpy())
            
        y_pred = np.concatenate(y_pred, axis=0) 
        x_input = np.concatenate(x_input, axis=0)
        
        results = {}
        ind = 0
        
        for id, info in zip(ids, infos):
            results[id] = {
                'data': y_pred[ind : ind + info['shape']].reshape(12, -1)[:, :info['length']],
                'label': info['label']
            }

            ind += info['shape']

        np.save(f"results/{args.dataset}/Denoised_{args.exp_name}.npy", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Denoising Evaluation")
    parser.add_argument("--exp_name", type=str, choices=[
        "FIR",
        "IIR",
        "DeScoD",
        "EDDM",
        "ARiRGen",
        "DRNN",
        "FCN_DAE",
        "ACDAE",
        "CBAM_DAE",
        "TCDAE", 
        "DeepFilter",
        "ECG_GAN",
        "AR"
    ], default="ARiRGen", help="Experiment name")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shots', type=int, default=1, help='Number of shots for repeated sampling')
    parser.add_argument('--type', type=int, default=1)
    parser.add_argument('--train_set', type=str, choices=['QT', 'SimEMG'], default='QT')
    parser.add_argument('--use_rmn', type=bool, default=True, help='Add Random Mixed Noise')
    parser.add_argument('--dataset', type=str, choices=['QT', 
                                                        'SimEMG', 
                                                        'Icentiak11',
                                                        'Arrhythmia',
                                                        'CPSC'], default='QT', help='Dataset to evaluate on')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path of saved base_model weights')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='AETest')
    
    args = parser.parse_args()
    
    if args.dataset == 'QT':
        evaluate_QT(args)
    elif args.dataset == 'SimEMG':
        evaluate_SimEMG(args)
    elif args.dataset == 'Icentiak11':
        evaluate_Icentiak11(args)
    elif args.dataset == 'Arrhythmia':
        evaluate_Arrhythmia(args)
    elif args.dataset == 'CPSC':
        evaluate_CPSC(args)
    