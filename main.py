import argparse
import torch
import datetime
import yaml
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from datasets.dataset import ECGDataset, EMGDataset

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="ECG Denoising")
    parser.add_argument("--exp_name", type=str, choices=[
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
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--train_set', type=str, default='QT', choices=['QT', 'SimEMG'], help='Dataset to use for training')
    parser.add_argument('--use_rmn', action='store_true', default=True, help='Add Random Mixed Noise')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')

    args = parser.parse_args()
    
    config_path = Path("./config") / f"{args.exp_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    foldername = f"./check_points/{args.train_set}/{args.exp_name}/noise_type_" + str(args.n_type) + "/"
    print('check_points_folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}/noise_type_" + str(args.n_type) + f"/{timestamp}"
    print('log_dir:', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load Data
    if args.train_set == 'QT':
        if args.exp_name == 'FlowMatching' :
            dataset = ECGDataset(n_type=args.n_type, use_rmn=args.use_rmn, use_snr=True, config=config)
        else:
            dataset = ECGDataset(n_type=args.n_type, use_rmn=args.use_rmn, config=config)
    elif args.train_set == 'SimEMG':
        dataset = EMGDataset(n_type=args.n_type, config=config, train=True)

    print('Data ready to use.')
    
    # Load model
    print('Loading model...')
    # DeScoD-ECG
    if (args.exp_name == "DeScoD"):
        from models.generation_filters.DeScoD_model import ConditionalModel
        from models.generation_filters.DeScoD_diffusion import DDPM
        from trainer import train_diffusion
        
        base_model = ConditionalModel(config['train']['feats']).to(args.device)
        model = DDPM(base_model, config, args.device)
        train_diffusion(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # EDDM
    if (args.exp_name == "EDDM"):
        from models.generation_filters.EDDM_model import UnetRes
        from models.generation_filters.EDDM_diffusion import ResidualDiffusion
        from trainer import train_eddm
        
        base_model = UnetRes(**config['base_model']).to(args.device)
        model = ResidualDiffusion(model=base_model, **config['diffusion']).to(args.device)
        train_eddm(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)

    # ARiR-Gen
    if (args.exp_name == "ARiRGen"):
        from models.generation_filters.FlowBackbone import Unet
        from models.generation_filters.FlowMatching import CFM
        from models.dl_filters.AR import AR
        from trainer import train_flow
        
        base_model = Unet(**config['base_model']).to(args.device)

        encoder_path = f"./check_points/{args.train_set}/AR/noise_type_" + str(args.n_type) + "/" + "model.pth"
        autoencoder = AR().to(args.device)
        autoencoder.load_state_dict(torch.load(encoder_path, map_location=args.device))

        model = CFM(base_model=base_model, autoencoder=autoencoder, **config['flow']).to(args.device)
        
        train_flow(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # DRNN
    elif (args.exp_name == "DRNN"):
        from models.dl_filters.DRNN import DRDNN
        from trainer import train_dl
        
        model = DRDNN(**config['model']).to(args.device)
        
        train_dl(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # FCN_DAE
    elif (args.exp_name == "FCN_DAE"):
        from models.dl_filters.FCN_DAE import FCN_DAE
        from trainer import train_dl
        
        model = FCN_DAE(**config['model']).to(args.device)
        
        train_dl(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)

    # ACDAE
    elif (args.exp_name == "ACDAE"):
        from models.dl_filters.ACDAE import ACDAE
        from trainer import train_dl
        
        model = ACDAE(**config['model']).to(args.device)
        
        train_dl(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # CBAM_DAE
    elif (args.exp_name == "CBAM_DAE"):
        from models.dl_filters.CBAM_DAE import AttentionSkipDAE2
        from trainer import train_dl
        
        model = AttentionSkipDAE2(**config['model']).to(args.device)
        
        train_dl(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # TCDAE
    elif (args.exp_name == "TCDAE"):
        from models.dl_filters.TCDAE import TCDAE
        from trainer import train_dl
        
        model = TCDAE(**config['model']).to(args.device)
        
        train_dl(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # DeepFilter
    elif (args.exp_name == "DeepFilter"):
        from models.dl_filters.DeepFilter import DeepFilterModelLANLDilated
        from trainer import train_dl
        
        model = DeepFilterModelLANLDilated().to(args.device)
        
        train_dl(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
        
    # ECG_GAN
    elif (args.exp_name == "ECG_GAN"):
        from models.generation_filters.ECGAN import Generator, Discriminator 
        from trainer import train_gan

        generator = Generator(input_channels=config['generator']['feats']).to(args.device)
        discriminator = Discriminator(input_channels=config['discriminator']['feats']).to(args.device)
        
        train_gan(generator, discriminator, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)
    
    # AR 
    elif (args.exp_name == "AR"):
        from models.dl_filters.AR import AR
        from trainer import train_ar
        
        model = AR().to(args.device)
        
        train_ar(model, config['train'], dataset, args.device, foldername=foldername, log_dir=log_dir)