import argparse
import torch
import numpy as np
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import pickle

from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(args):
    os.makedirs(f"results/{args.exp_name}", exist_ok=True)

    with open('./SimEMG_LowSNR_Paired_Processed/SimEMG_LowSNR_Paired_QTMethod.pkl', 'rb') as f:
        processed_signal = pickle.load(f)
        
    clean_signal = processed_signal['clean']
    noised_signal = processed_signal['noisy']

    clean = []
    noised = []

    for items in clean_signal.values():
        for beat in items:
            clean.append(beat / 200)
    for items in noised_signal.values():
        for beat in items:
            noised.append(beat / 200)
        
    y_test = np.expand_dims(np.array(clean), axis=2)
    X_test = np.expand_dims(np.array(noised), axis=2)
    
    # FIR & IIR filters
    if args.exp_name == "FIR":
        from digital_filters.FIR_filter import FIR_test_Dataset
        [X_test, y_test, y_pred] = FIR_test_Dataset([None, None, X_test, y_test])
    
    elif args.exp_name == "IIR":
        from digital_filters.IIR_filter import IIR_test_Dataset
        [X_test, y_test, y_pred] = IIR_test_Dataset([None, None, X_test, y_test])
        
        X_test = X_test.transpose(0, 2, 1)
        y_test = y_test.transpose(0, 2, 1)
        y_pred = y_pred.transpose(0, 2, 1)
        results = {
                'y_pred': y_pred,
                'y_true': y_test,
                'x_input': X_test
            }
        np.save(f"results/{args.exp_name}/results_{args.n_type}.npy", results)

    # dl_filters
    else:
        config_path = Path("./config") / f"{args.exp_name}.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        foldername = f"./check_points/{args.exp_name}/noise_type_" + str(args.n_type) + "/"

        X_test_tensor = torch.FloatTensor(X_test)
        X_test_tensor = X_test_tensor.permute(0, 2, 1)
        
        y_test_tensor = torch.FloatTensor(y_test)
        y_test_tensor = y_test_tensor.permute(0, 2, 1)
        
        test_set = TensorDataset(y_test_tensor, X_test_tensor)
        test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'])

        print(f"Loading {args.exp_name} model for noise type {args.n_type}...")

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
        y_true = []
        x_input = []
        trajectory = []
        print(f"Evaluating {args.exp_name} model for noise type {args.n_type}...")
        with torch.no_grad():
            for clean_batch, noisy_batch in tqdm(test_loader):
                clean_batch = clean_batch.to(args.device)
                noisy_batch = noisy_batch.to(args.device)
                
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
                        trajectory.append(denoised_trajectory.cpu().numpy().transpose(1, 0, 2, 3))
                    
                elif args.exp_name == "ECG_GAN":
                    batch_size = noisy_batch.shape[0]
                    z = torch.randn(batch_size, 512, 8).to(args.device)
                    denoised_batch = model(noisy_batch, z)
                        
                else:
                    denoised_batch = model(noisy_batch)
                
                y_pred.append(denoised_batch.cpu().numpy())
                y_true.append(clean_batch.cpu().numpy())
                x_input.append(noisy_batch.cpu().numpy())
                
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        x_input = np.concatenate(x_input, axis=0)
        
        if args.exp_name == "FlowMatching":
            trajectory = np.concatenate(trajectory, axis=0)
            np.save(f"results/{args.exp_name}/trajectory_{args.n_type}.npy", trajectory)
        # else:
        # results = {
        #     'y_pred': y_pred,
        #     'y_true': y_true,
        #     'x_input': x_input
        # }
        # np.save(f"results/{args.exp_name}/results_{args.n_type}.npy", results)


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
    parser.add_argument('--shots', type=int, default=1, help='Number of shots for Diffusion model')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    
    args = parser.parse_args()
    
    evaluate_model(args)