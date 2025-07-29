import argparse
import torch
import numpy as np
import pandas as pd
import yaml
import os
import csv
from pathlib import Path
from tqdm import tqdm

from data_preparation import Data_Preparation, Data_Preparation_RMN
from utils.metrics import SSD, MAD, PRD, COS_SIM, SNR, SNR_improvement
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(args):
    os.makedirs("results", exist_ok=True)
    
    all_metrics = {}
    all_noise_levels = []
    
    for n_type in [1, 2]:
        # Load Data
        [_, _, X_test, y_test] = Data_Preparation(n_type) if not args.use_rmn else Data_Preparation_RMN(n_type)
        
        try:
            noise_level = np.load('./Data/prepared/rnd_test.npy')
            all_noise_levels.append(noise_level)
        except FileNotFoundError:
            print(f"Warning: rnd_test.npy not found for noise type {n_type}")
        
        # FIR & IIR filters
        if args.exp_name == "FIR":
            from digital_filters.FIR_filter import FIR_test_Dataset
            print(f"Evaluating FIR filter for noise type {n_type}...")
            [X_test, y_test, y_pred] = FIR_test_Dataset([None, None, X_test, y_test])
        
        elif args.exp_name == "IIR":
            from digital_filters.IIR_filter import IIR_test_Dataset
            print(f"Evaluating IIR filter for noise type {n_type}...")
            [X_test, y_test, y_pred] = IIR_test_Dataset([None, None, X_test, y_test])
        
        # dl_filters
        else:
            config_path = Path("./config") / f"{args.exp_name}.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            foldername = f"./check_points/{args.exp_name}/noise_type_" + str(n_type) + "/"
     
            X_test_tensor = torch.FloatTensor(X_test)
            X_test_tensor = X_test_tensor.permute(0, 2, 1)
            
            y_test_tensor = torch.FloatTensor(y_test)
            y_test_tensor = y_test_tensor.permute(0, 2, 1)
            
            test_set = TensorDataset(y_test_tensor, X_test_tensor)
            test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'])

            print(f"Loading {args.exp_name} model for noise type {n_type}...")
            
            # DeScoD-ECG
            if args.exp_name == "DeScoD":
                from generation_filters.DeScoD_model import ConditionalModel
                from generation_filters.DeScoD_diffusion import DDPM
                
                base_model = ConditionalModel(config['train']['feats']).to(args.device)
                model = DDPM(base_model, config, args.device)

            # EDDM
            elif (args.exp_name == "EDDM"):
                from generation_filters.EDDM_model import UnetRes
                from generation_filters.EDDM_diffusion import ResidualDiffusion
                
                base_model = UnetRes(**config['base_model']).to(args.device)
                model = ResidualDiffusion(model=base_model, **config['diffusion']).to(args.device)
    
            # FlowMatching
            elif (args.exp_name == "FlowMatching"):
                from generation_filters.FlowBackbone import Unet
                from generation_filters.FlowMatching import CFM, AdaCFM
                
                base_model = Unet(**config['base_model']).to(args.device)
                model = CFM(base_model=base_model, **config['flow']).to(args.device)
                
            # DRNN
            elif args.exp_name == "DRNN":
                from dl_filters.DRNN import DRDNN
                model = DRDNN(**config['model']).to(args.device)
                
            # FCN_DAE
            elif args.exp_name == "FCN_DAE":
                from dl_filters.FCN_DAE import FCN_DAE
                model = FCN_DAE(filters=config['model']['filters']).to(args.device)
            
            # ACDAE
            elif args.exp_name == "ACDAE":
                from dl_filters.ACDAE import ACDAE
                model = ACDAE(in_channels=config['model']['in_channels']).to(args.device)
                
            # CBAM_DAE
            elif args.exp_name == "CBAM_DAE":
                from dl_filters.CBAM_DAE import AttentionSkipDAE2
                model = AttentionSkipDAE2(signal_size=config['model']['signal_size']).to(args.device)
            
            # TCDAE
            elif (args.exp_name == "TCDAE"):
                from dl_filters.TCDAE import TCDAE
                model = TCDAE(**config['model']).to(args.device)
                        
            # DeepFilter
            elif args.exp_name == "DeepFilter":
                from dl_filters.DeepFilter import DeepFilterModelLANLDilated
                model = DeepFilterModelLANLDilated().to(args.device)
                
            # ECG_GAN
            elif args.exp_name == "ECG_GAN":
                from generation_filters.ECGAN import Generator
                model = Generator(input_channels=config['generator']['feats']).to(args.device)

            model_path = foldername + "/model.pth"
            model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
            model.eval()

            y_pred = []
            trajectory = []
            print(f"Evaluating {args.exp_name} model for noise type {n_type}...")
            with torch.no_grad():
                for clean_batch, noisy_batch in tqdm(test_loader):
                    clean_batch = clean_batch.to(args.device)
                    noisy_batch = noisy_batch.to(args.device)
                    
                    if args.exp_name == "DeScoD":
                        shots = args.shots
                        if shots > 1:
                            denoised_batch = 0
                            for _ in range(shots):
                                denoised_batch += model.denoise(noisy_batch)
                            denoised_batch = denoised_batch / shots
                        else:
                            denoised_batch = model.denoise(noisy_batch)
                            
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
                            
                    elif args.exp_name == "FlowMatching":
                        shots = args.shots
                        if shots > 1:
                            denoised_batch = 0
                            for _ in range(shots):
                                [denoised, _] = model.sample(noisy_batch)
                                denoised_batch += denoised
                            denoised_batch = denoised_batch / shots
                        else:
                            [denoised_batch, denoised_trajectory] = model.sample(noisy_batch)
                            denoised_trajectory = torch.cat((denoised_trajectory, clean_batch.unsqueeze(0)), dim=0)
                        
                    elif args.exp_name == "ECG_GAN":
                        batch_size = noisy_batch.shape[0]
                        z = torch.randn(batch_size, 512, 8).to(args.device)
                        denoised_batch = model(noisy_batch, z)
                        
                    else:
                        denoised_batch = model(noisy_batch)
                    
                    y_pred.append(denoised_batch.cpu().numpy())
                    trajectory.append(denoised_trajectory.cpu().numpy().transpose(1, 0, 2, 3))
            
            y_pred = np.concatenate(y_pred, axis=0)
            y_pred = np.transpose(y_pred, (0, 2, 1))
            trajectory = np.concatenate(trajectory, axis=0)
            np.save(f"results/{args.exp_name}_trajectory_{n_type}.npy", trajectory)

        metrics = {
            "SSD": SSD(y_test, y_pred),
            "MAD": MAD(y_test, y_pred),
            "PRD": PRD(y_test, y_pred),
            "COS_SIM": COS_SIM(y_test, y_pred),
            "SNR out": SNR(y_test, y_pred),
            "ImSNR": SNR_improvement(X_test, y_pred, y_test)
        }

        if n_type == 1:
            all_metrics = {k: v for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] = np.concatenate([all_metrics[k], v])

    metrics_stats = {}
    for name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        metrics_stats[name] = f"{mean_val:.3f}±{std_val:.3f}"

    if all_noise_levels:
        n_level = np.concatenate(all_noise_levels)
        # segs = [0.2, 0.6, 1.0, 1.5, 2.0]
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

    with open(f"results/{args.exp_name}_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        headers = ["Model", "SSD (au) ↓", "MAD (au) ↓", "PRD (%) ↓", "Cosine Sim ↑", "SNR out (dB) ↑", "ImSNR (dB) ↑"]
        writer.writerow(headers)
        writer.writerow([args.exp_name] + [metrics_stats[m] for m in ["SSD", "MAD", "PRD", "COS_SIM", "SNR out", "ImSNR"]])

        writer.writerow([])

        if all_noise_levels:
            seg_labels = [f"{segs[i]}-{segs[i+1]}dB" for i in range(len(segs)-1)]
            writer.writerow(["Metrics"] + seg_labels)
            
            for metric in ["SSD", "MAD", "PRD", "COS_SIM", "SNR out", "ImSNR"]:
                row = [metric]
                for seg_label in seg_labels:
                    if seg_label in segmented_results[metric]:
                        row.append(segmented_results[metric][seg_label])
                    else:
                        row.append("N/A")
                writer.writerow(row)

    print(f"Evaluation complete for {args.exp_name}")
    print(f"Results saved to results/{args.exp_name}_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Denoising Evaluation")
    parser.add_argument("--exp_name", type=str, choices=[
        "FIR",
        "IIR",
        "DeScoD",
        "EDDM",
        "FlowMatching",
        "DRNN",
        "FCN_DAE",
        "ACDAE",
        "CBAM_DAE",
        "TCDAE",
        "DeepFilter",
        "ECG_GAN",
    ], default="FlowMatching", help="Experiment name")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--use_rmn', type=bool, default=True, help='Use Random Mixed Noise')
    parser.add_argument('--shots', type=int, default=1, help='Number of shots for Diffusion model')
    
    args = parser.parse_args()
    
    evaluate_model(args)