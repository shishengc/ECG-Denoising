import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.loss_function import SSDLoss, CombinedSSDMADLoss, MADLoss, HuberFreqLoss
from utils.train_utils import LRScheduler, EarlyStopping

def train_diffusion(model, config, dataset, device, valid_epoch_interval=5, foldername="", log_dir=None):

    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    optimizer = optimizer_type(model.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})

    #ema = EMA(0.9)
    #ema.register(model)
    
    train_loader, valid_loader, _ = dataset._get_loader()
    
    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    if config['lr_scheduler'].get("use", False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=150, gamma=.1, verbose=True
        )
    
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                loss = model(clean_batch, noisy_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                
                #ema.update(model)
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": f"{avg_loss / batch_no:.4f}",
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            if lr_scheduler is not None:
                lr_scheduler.step()
        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
            
        if (epoch_no + 1) % config['val_interval'] == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        denoised_batch = model.denoising(noisy_batch)
                        loss = SSDLoss()(denoised_batch, clean_batch).item()
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": f"{avg_loss_valid / batch_no:.4f}",
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            
            writer.add_scalar('Loss/Validation', avg_loss_valid / batch_no, epoch_no)
            
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at Epoch", epoch_no+1)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
    
    torch.save(model.state_dict(), final_path)


def train_gan(generator, discriminator, config, dataset, device, foldername="", log_dir=None):
    
    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "RMSprop"))
    
    optimizer_G = optimizer_type(generator.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    optimizer_D = optimizer_type(discriminator.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    
    def lsgan_loss(pred, target):
        return torch.mean((pred - target) ** 2)
    
    if foldername != "":
        gen_output_path = foldername + "/generator.pth"
        disc_output_path = foldername + "/discriminator.pth"
        gen_final_path = foldername + "/generator_final.pth"
        disc_final_path = foldername + "/discriminator_final.pth"
    
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    lambda_l1 = config.get('lambda_l1', 100.0)
    
    # Get train and validation loaders
    train_loader, valid_loader, _ = dataset._get_loader()
    
    for epoch_no in range(config["epochs"]):
        avg_g_loss = 0
        avg_d_loss = 0
        generator.train()
        discriminator.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                batch_size = clean_batch.shape[0]
                
                z = torch.randn(batch_size, 512, 8).to(device)
                denoised_ecg = generator(noisy_batch, z)
                
                optimizer_D.zero_grad()
                
                real_pair = torch.cat([clean_batch, noisy_batch], dim=1)
                real_pred = discriminator(real_pair)
                real_loss = lsgan_loss(real_pred, torch.ones_like(real_pred))
                
                fake_pair = torch.cat([denoised_ecg.detach(), noisy_batch], dim=1)
                fake_pred = discriminator(fake_pair)
                fake_loss = lsgan_loss(fake_pred, torch.zeros_like(fake_pred))
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()
                
                optimizer_G.zero_grad()
                
                fake_pair = torch.cat([denoised_ecg, noisy_batch], dim=1)
                fake_pred = discriminator(fake_pair)
                g_adv_loss = lsgan_loss(fake_pred, torch.ones_like(fake_pred))
                
                g_l1_loss = torch.nn.L1Loss()(denoised_ecg, clean_batch)
                
                g_loss = g_adv_loss + lambda_l1 * g_l1_loss
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()
                
                avg_g_loss += g_loss.item()
                avg_d_loss += d_loss.item()
                
                it.set_postfix(
                    ordered_dict={
                        "avg_g_loss": f"{avg_g_loss / batch_no:.3f}",
                        "avg_d_loss": f"{avg_d_loss / batch_no:.3f}",
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
        writer.add_scalar('Loss/Generator', avg_g_loss / batch_no, epoch_no)
        writer.add_scalar('Loss/Discriminator', avg_d_loss / batch_no, epoch_no)
        
        if (epoch_no + 1) % config['val_interval'] == 0:
            generator.eval()
            discriminator.eval()
            avg_valid_loss = 0
            
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        batch_size = clean_batch.shape[0]
                        
                        z = torch.randn(batch_size, 512, 8).to(device)
                        denoised_ecg = generator(noisy_batch, z)
                        
                        valid_loss = torch.nn.L1Loss()(denoised_ecg, clean_batch)
                        avg_valid_loss += valid_loss.item()
                        
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_loss": f"{avg_valid_loss / batch_no:.3f}",
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            writer.add_scalar('Loss/Validation', avg_valid_loss / batch_no, epoch_no)
            
            if best_valid_loss > avg_valid_loss / batch_no:
                best_valid_loss = avg_valid_loss / batch_no
                print("\n best loss is updated to", f"{avg_valid_loss / batch_no:.4f}", "at Epoch", epoch_no + 1)
                
                if foldername != "":
                    torch.save(generator.state_dict(), gen_output_path)
                    torch.save(discriminator.state_dict(), disc_output_path)
    
    if foldername != "":
        torch.save(generator.state_dict(), gen_final_path)
        torch.save(discriminator.state_dict(), disc_final_path)    
    

def train_dl(model, config, dataset, device, foldername="", log_dir=None):

    # optimizer config
    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    optimizer = optimizer_type(model.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    
    # criterion config
    criterion = config.get('criterion', 'MSELoss')
    if criterion == 'MSELoss':
        criterion = torch.nn.MSELoss()
    elif criterion == 'SSDLoss':
        criterion = SSDLoss()
    elif criterion == 'CombinedSSDMADLoss':
        criterion = CombinedSSDMADLoss()
    elif criterion == 'HuberFreqLoss':
        criterion = HuberFreqLoss()
    
    # metrics for tracking
    ssd_metric = SSDLoss()
    mad_metric = MADLoss()
    
    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    
    # lr_scheduler config
    if config['lr_scheduler'].get("use", False):
        lr_scheduler = LRScheduler(config.get('lr_scheduler', {}))
    else:
        lr_scheduler = None
        
    # Early stopping config
    if config['early_stopping'].get("use", False):
        early_stopping = EarlyStopping(config.get('early_stopping', {}))
    else:
        early_stopping = None
    
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    
    # Get train and validation loaders
    train_loader, valid_loader, _ = dataset._get_loader()
    
    # training loop
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        avg_ssd = 0
        avg_mad = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                denoised_batch = model(noisy_batch)
                loss = criterion(denoised_batch, clean_batch)
                loss.backward()
                optimizer.step()
                
                ssd_value = ssd_metric(denoised_batch, clean_batch).item()
                mad_value = mad_metric(denoised_batch, clean_batch).item()
                
                avg_loss += loss.item()
                avg_ssd += ssd_value
                avg_mad += mad_value
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": f"{avg_loss / batch_no:.4f}",
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            if lr_scheduler is not None and config['lr_scheduler']['type'] != "ReduceLROnPlateau":
                lr_scheduler.step()
        
        # Log training metrics
        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
        writer.add_scalar('SSD/Train', avg_ssd / batch_no, epoch_no)
        writer.add_scalar('MAD/Train', avg_mad / batch_no, epoch_no)
            
        if (epoch_no + 1) % config['val_interval'] == 0:
            model.eval()
            avg_loss_valid = 0
            avg_ssd_valid = 0
            avg_mad_valid = 0
            
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        denoised_batch = model(noisy_batch)
                        
                        loss = criterion(denoised_batch, clean_batch)
                        ssd_value = ssd_metric(denoised_batch, clean_batch).item()
                        mad_value = mad_metric(denoised_batch, clean_batch).item()
                        
                        avg_loss_valid += loss.item()
                        avg_ssd_valid += ssd_value
                        avg_mad_valid += mad_value
                        
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": f"{avg_loss_valid / batch_no:.4f}",
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            
            # Log validation metrics
            writer.add_scalar('Loss/Validation', avg_loss_valid / batch_no, epoch_no)
            writer.add_scalar('SSD/Validation', avg_ssd_valid / batch_no, epoch_no)
            writer.add_scalar('MAD/Validation', avg_mad_valid / batch_no, epoch_no)
            
            if config['lr_scheduler']['type'] == "ReduceLROnPlateau":
                lr_scheduler.step(avg_loss_valid / batch_no)
            
            if best_valid_loss > avg_ssd_valid/batch_no:
                best_valid_loss = avg_ssd_valid/batch_no
                print("\n best loss is updated to ",f"{avg_ssd_valid / batch_no:.4f}","at Epoch", epoch_no+1)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                    
            if early_stopping is not None:
                if early_stopping.step(avg_ssd_valid/batch_no):
                    print(f"\nEarly stopping triggered after {epoch_no+1} epochs!")
                    break
    
    torch.save(model.state_dict(), final_path)
    

def train_eddm(model, config, dataset, device, foldername="", log_dir=None):
    from ema_pytorch import EMA

    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    opt0 = optimizer_type(model.base_model.unet0.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    opt1 = optimizer_type(model.base_model.unet1.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})

    if config['use_ema']:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.to(device)

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    
    step = 0
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    
    # Get train and validation loaders
    train_loader, valid_loader, _ = dataset._get_loader()
    data_iter = iter(train_loader)
    
    with tqdm(initial=step, total=config['steps']) as pbar:
        while step < config['steps']:
            model.train()
            
            if config['num_unet'] == 1:
                total_loss = [0]
            elif config['num_unet'] == 2:
                total_loss = [0, 0]
            
            for _ in range(config['gradient_accumulate_every']):
                try:
                    if config['condition']:
                        clean_batch, noisy_batch = next(data_iter)
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        data = [clean_batch, noisy_batch]
                    else:
                        clean_batch = next(data_iter)
                        clean_batch = clean_batch[0] if isinstance(clean_batch, list) else clean_batch
                        clean_batch = clean_batch.to(device)
                        data = clean_batch
                        
                except StopIteration:
                    data_iter = iter(train_loader)
                    if config['condition']:
                        clean_batch, noisy_batch = next(data_iter)
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        data = [clean_batch, noisy_batch]
                    else:
                        clean_batch = next(data_iter)
                        clean_batch = clean_batch[0] if isinstance(clean_batch, list) else clean_batch
                        clean_batch = clean_batch.to(device)
                        data = clean_batch
                
                loss = model(data)
                
                for i in range(config['num_unet']):
                    loss[i] = loss[i] / config['gradient_accumulate_every']
                    total_loss[i] = total_loss[i] + loss[i].item()
                
                for i in range(config['num_unet']):
                    loss[i].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if config['num_unet'] == 1:
                opt0.step()
                opt0.zero_grad()
            elif config['num_unet'] == 2:
                opt0.step()
                opt0.zero_grad()
                opt1.step()
                opt1.zero_grad()

            if config['use_ema']:
                ema.update()

            step += 1
            pbar.update(1)
            
            if config['num_unet'] == 1:
                pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
            elif config['num_unet'] == 2:
                pbar.set_description(f'loss_unet0: {total_loss[0]:.4f},loss_unet1: {total_loss[1]:.4f}')

            for i in range(config['num_unet']):
                writer.add_scalar(f'Loss/Train/UNet{i}', total_loss[i], step)

            if step % config['val_interval'] == 0:
                if valid_loader is not None:
                    model.eval()
                    avg_loss_valid = 0
                    
                    with torch.no_grad():
                        with tqdm(valid_loader) as it:
                            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

                                [_, denoised_batch] = ema.ema_model.sample([noisy_batch, 0], batch_size=noisy_batch.shape[0])
                                loss = SSDLoss()(denoised_batch, clean_batch).item()
                                
                                avg_loss_valid += loss
                                
                                it.set_postfix(
                                ordered_dict={
                                    "valid_avg_epoch_loss": f"{avg_loss_valid / batch_no:.4f}",
                                    "epoch": step,
                                },
                                refresh=True,
                                )

                        writer.add_scalar('Loss/Validation', avg_loss_valid / batch_no, step)
                    
                    if best_valid_loss > avg_loss_valid / batch_no:
                        best_valid_loss = avg_loss_valid / batch_no
                        print(f"\n Best loss updated to {avg_loss_valid / batch_no:.4f} at step {step}")
                        
                        if foldername != "":
                            if config['use_ema']:
                                torch.save(ema.ema_model.state_dict(), output_path)
                            else:
                                torch.save(model.state_dict(), output_path)

    if foldername != "":
        if config['use_ema']:
            torch.save(ema.ema_model.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)
    


def train_flow(model, config, dataset, device, foldername="", log_dir=None):
    from ema_pytorch import EMA
    
    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    optimizer = optimizer_type(model.base_model.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})

    if config['use_ema']:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.to(device)

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    
    step = 0
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)

    # Get train and validation loaders
    train_loader, valid_loader, _ = dataset._get_loader()
    data_iter = iter(train_loader)
    
    with tqdm(initial=step, total=config['steps']) as pbar:
        while step < config['steps']:
            model.train()
            total_loss = 0.
            optimizer.zero_grad()
       
            for _ in range(config['gradient_accumulate_every']):
                try:
                    clean_batch, noisy_batch = next(data_iter)
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                except StopIteration:
                    data_iter = iter(train_loader)
                    clean_batch, noisy_batch = next(data_iter)
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
      
                loss, _, _ = model(noisy_batch, clean_batch)
                loss/= config['gradient_accumulate_every']
                total_loss += loss.item()
                
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if config['use_ema']:
                ema.update()

            step += 1
            pbar.update(1)
            pbar.set_description(f'train_loss: {total_loss:.4f}')
            
            writer.add_scalar(f'Loss/Train', total_loss, step)

            if step % config['val_interval'] == 0:
                if valid_loader is not None:
                    model.eval()
                    loss_list = []
                    avg_loss_valid = 0
                    
                    with torch.no_grad():
                        with tqdm(valid_loader) as it:
                            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

                                [denoised_batch, _] = ema.ema_model.sample(noisy_batch)
                                
                                if dataset.refresh == False:
                                    loss = SSDLoss()(denoised_batch, clean_batch).item()
                                else:
                                    loss = torch.sum((denoised_batch - clean_batch)**2, dim=-1).squeeze(1)
                                    loss_list.append(np.array(loss.cpu().numpy()))
                                    loss = loss.mean().item()
                                    
                                avg_loss_valid += loss
                                
                                it.set_postfix(
                                ordered_dict={
                                    "valid_avg_loss": f"{avg_loss_valid / batch_no:.4f}",
                                    "Step": step,
                                },
                                refresh=True,
                                )

                        writer.add_scalar('Loss/Validation', avg_loss_valid / batch_no, step)
                    
                    if loss_list:
                        loss_list = np.concatenate(loss_list)
                        train_loader = dataset.update(loss_list)
                        data_iter = iter(train_loader)
                    
                    if best_valid_loss > avg_loss_valid / batch_no:
                        best_valid_loss = avg_loss_valid / batch_no
                        print(f"\n Best loss updated to {avg_loss_valid / batch_no:.4f} at step {step}")
                        
                        if foldername != "":
                            if config['use_ema']:
                                torch.save(ema.ema_model.state_dict(), output_path)
                            else:
                                torch.save(model.state_dict(), output_path)

    if foldername != "":
        if config['use_ema']:
            torch.save(ema.ema_model.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)
      