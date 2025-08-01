import argparse
import torch
import datetime
import json
import yaml
import os
from pathlib import Path

from data_preparation import Data_Preparation, Data_Preparation_RMN

from trainer import train_diffusion, train_gan, train_dl, train_eddm, train_flow

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="ECG Denoising")
    parser.add_argument("--exp_name", type=str, choices=[
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
        ""
    ], default="FlowMatching", help="Experiment name")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--use_rmn', type=bool, default=True, help='Add Random Mixed Noise')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    
    parser.add_argument('--val_interval', type=int, default=1)
    args = parser.parse_args()
    
    config_path = Path("./config") / f"{args.exp_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    foldername = f"./check_points/{args.exp_name}/noise_type_" + str(args.n_type) + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}/noise_type_" + str(args.n_type) + f"/{timestamp}"
    print('log:', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load Data
    [X_train, y_train, X_test, y_test] = Data_Preparation(args.n_type) if not args.use_rmn else Data_Preparation_RMN(args.n_type)
    
    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0,2,1)
    
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0,2,1)
    
    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)
    
    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)
    
    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'])
    
    
    # Load model
    print('Loading model...')
    # DeScoD-ECG
    if (args.exp_name == "DeScoD"):
        from generation_filters.DeScoD_model import ConditionalModel
        from generation_filters.DeScoD_diffusion import DDPM
        
        base_model = ConditionalModel(config['train']['feats']).to(args.device)
        model = DDPM(base_model, config, args.device)
        train_diffusion(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # EDDM
    if (args.exp_name == "EDDM"):
        from generation_filters.EDDM_model import UnetRes
        from generation_filters.EDDM_diffusion import ResidualDiffusion
        
        base_model = UnetRes(**config['base_model']).to(args.device)
        model = ResidualDiffusion(model=base_model, **config['diffusion']).to(args.device)
        train_eddm(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)

    # FlowMatching
    if (args.exp_name == "FlowMatching"):
        from generation_filters.FlowBackbone import Unet
        from generation_filters.FlowMatching import CFM
        
        base_model = Unet(**config['base_model']).to(args.device)
        from generation_filters.AttnUnet import AutoEncoder
        autoencoder = AutoEncoder()
        autoencoder.load_state_dict(torch.load("./check_points/VAE/noise_type_1/model.pth", weights_only=True))
        autoencoder.to(args.device)
        autoencoder.eval()
        model = CFM(base_model=base_model, autoencoder=autoencoder, **config['flow']).to(args.device)
        
        train_flow(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # DRNN
    elif (args.exp_name == "DRNN"):
        from dl_filters.DRNN import DRDNN
        model = DRDNN(**config['model']).to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # FCN_DAE
    elif (args.exp_name == "FCN_DAE"):
        from dl_filters.FCN_DAE import FCN_DAE
        model = FCN_DAE(filters=config['model']['filters']).to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)

    # ACDAE
    elif (args.exp_name == "ACDAE"):
        from dl_filters.ACDAE import ACDAE
        model = ACDAE(in_channels=config['model']['in_channels']).to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # CBAM_DAE
    elif (args.exp_name == "CBAM_DAE"):
        from dl_filters.CBAM_DAE import AttentionSkipDAE2
        model = AttentionSkipDAE2(**config['model']).to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # TCDAE
    elif (args.exp_name == "TCDAE"):
        from dl_filters.TCDAE import TCDAE
        model = TCDAE(**config['model']).to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # DeepFilter
    elif (args.exp_name == "DeepFilter"):
        from dl_filters.DeepFilter import DeepFilterModelLANLDilated
        model = DeepFilterModelLANLDilated().to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # ECG_GAN
    elif (args.exp_name == "ECG_GAN"):
        from generation_filters.ECGAN import Generator, Discriminator

        generator = Generator(input_channels=config['generator']['feats']).to(args.device)
        discriminator = Discriminator(input_channels=config['discriminator']['feats']).to(args.device)
        
        train_gan(generator, discriminator, config['train'], train_loader ,args.device,
                  valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
    