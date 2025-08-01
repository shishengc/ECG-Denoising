import argparse
import torch
import datetime
import json
import yaml
import os
from pathlib import Path
from tqdm import tqdm

from data_preparation import Data_Preparation, Data_Preparation_RMN

from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from generation_filters.AttnUnet import AutoEncoder
import torch.nn.functional as F


def vae_loss(recon_x, x, mean, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = 1e-4 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss - kl_loss, recon_loss


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="ECG Denoising")
    parser.add_argument("--exp_name", type=str, default="VAE", help="Experiment name")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--use_rmn', type=bool, default=True, help='Add Random Mixed Noise')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_interval', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    foldername = f"./check_points/{args.exp_name}/noise_type_" + str(args.n_type) + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}/noise_type_" + str(args.n_type) + f"/{timestamp}"
    print('log:', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    [X_train, y_train, X_test, y_test] = Data_Preparation(args.n_type) if not args.use_rmn else Data_Preparation_RMN(args.n_type)
    
    X_train = torch.FloatTensor(X_train).permute(0,2,1)
    y_train = torch.FloatTensor(y_train).permute(0,2,1)
    X_test = torch.FloatTensor(X_test).permute(0,2,1)
    y_test = torch.FloatTensor(y_test).permute(0,2,1)
    
    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)
    
    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True)
    
    model = AutoEncoder(img_ch=1, output_ch=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
    from generation_filters.FlowBackbone import Unet
    from generation_filters.FlowMatching import CFM
    config_path = Path("./config") / "FlowMatching.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    base_model = Unet(**config['base_model']).to(args.device)
    cfm = CFM(base_model=base_model, **config['flow'])
    cfm.load_state_dict(torch.load("./check_points/FlowMatching/noise_type_1/model.pth", weights_only=True))
    cfm.to(device)
    cfm.eval()
    
    output_path = foldername + "/model.pth"
    final_path = foldername + "/final.pth"
    
    best_valid_loss = float('inf')
    writer = SummaryWriter(log_dir=log_dir)
    
    total_steps = args.epochs * len(train_loader)
    current_step = 0
    
    with tqdm(total=total_steps, desc="Training", unit="batch") as pbar:
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            
            for batch_idx, (clean_batch, noisy_batch) in enumerate(train_loader):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                
                optimizer.zero_grad()
                with torch.no_grad():
                    [noisy_batch, _] = cfm.sample(noisy_batch)
                
                recon_batch, mean, logvar = model(noisy_batch)
                loss, _ = vae_loss(recon_batch, clean_batch, mean, logvar)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                current_step += 1
                
                pbar.set_postfix({
                    "epoch": f"{epoch + 1}/{args.epochs}",
                    "batch": f"{batch_idx + 1}/{len(train_loader)}",
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}"
                })
                pbar.update(1)
            
            avg_train_loss = epoch_loss / len(train_loader)
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            
            if (epoch + 1) % args.val_interval == 0 and val_loader is not None:
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_idx, (clean_batch, noisy_batch) in enumerate(val_loader):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        
                        noisy_batch, _ = cfm.sample(noisy_batch)
                        
                        recon_batch, mean, logvar = model(noisy_batch)
                        _, loss = vae_loss(recon_batch, clean_batch, mean, logvar)
                        val_loss += loss.item()
                        
                        pbar.set_postfix({
                            "epoch": f"{epoch + 1}/{args.epochs}",
                            "mode": "validation",
                            "val_loss": f"{val_loss / (batch_idx + 1):.4f}"
                        })
                
                avg_val_loss = val_loss / len(val_loader)
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
                
                if avg_val_loss < best_valid_loss:
                    best_valid_loss = avg_val_loss
                    pbar.write(f"Best loss updated to {avg_val_loss:.4f} at epoch {epoch + 1}")
                    torch.save(model.state_dict(), output_path)