import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA
from tqdm import tqdm

from utils.loss_function import SSDLoss, CombinedSSDMADLoss, MADLoss, HuberFreqLoss
from utils.train_utils import LRScheduler, EarlyStopping


def train_diffusion(model, config, dataset, device, foldername="", log_dir=None):

    opt = getattr(optim, config['optimizer'].get("type", "Adam"))
    optimizer = opt(model.parameters(), **{k: v for k, v in config['optimizer'].items() if k != 'type'})
    
    if config['use_ema']:
        ema = EMA(model, beta=0.9, update_every=10)
        ema.to(device)
    
    train_loader, val_loader, _ = dataset._get_loader()
    
    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    if config['lr_scheduler'].get("use", False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=150, gamma=.1, verbose=False
        )
    
    best_val_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    pbar = tqdm(total=len(train_loader), desc="Training")
    
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()

        pbar.reset(total=len(train_loader))
        pbar.set_description(f"Epoch {epoch_no + 1}/{config['epochs']}")
        
        for batch_no, (clean_batch, noisy_batch) in enumerate(train_loader, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            optimizer.zero_grad()
            
            loss = model(clean_batch, noisy_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            
            if config['use_ema']:
                ema.update()
            
            avg_loss += loss
            
            pbar.set_postfix(
                ordered_dict={
                    "Train_loss": f"{avg_loss / batch_no:.4f}",
                },
                refresh=True,
            )
            pbar.update(1)

        if lr_scheduler is not None:
            lr_scheduler.step()

        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
            
        if (epoch_no + 1) % config['val_interval'] == 0:
            if config['use_ema']:
                ema.ema_model.eval()
            else:
                model.eval()

            avg_val_loss = 0
            
            with torch.no_grad():
                with tqdm(val_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        
                        if config['use_ema']:
                            denoised_batch = ema.ema_model.denoising(noisy_batch)
                        else:
                            denoised_batch = model.denoising(noisy_batch)
                        loss = SSDLoss()(denoised_batch, clean_batch).item()
                        avg_val_loss += loss

                        it.set_postfix(
                            ordered_dict={
                                "Val_loss": f"{avg_val_loss / batch_no:.4f}",
                                "Epoch": epoch_no + 1,
                            },
                            refresh=True,
                        )
            
            writer.add_scalar('Loss/Validation', avg_val_loss / batch_no, epoch_no)
            
            if best_val_loss > avg_val_loss/batch_no:
                best_val_loss = avg_val_loss/batch_no
                print("\n best loss is updated to ",f"{avg_val_loss / batch_no:.4f}","at Epoch", epoch_no + 1)
                
                if foldername != "":
                    if config['use_ema']:
                        torch.save(ema.ema_model.state_dict(), output_path)
                    else:
                        torch.save(model.state_dict(), output_path)
    
    torch.save(model.state_dict(), final_path)


def train_gan(generator, discriminator, config, dataset, device, foldername="", log_dir=None):
    
    opt = getattr(optim, config['optimizer'].get("type", "RMSprop"))
    
    optimizer_G = opt(generator.parameters(), **{k: v for k, v in config['optimizer'].items() if k not in ['type']})
    optimizer_D = opt(discriminator.parameters(), **{k: v for k, v in config['optimizer'].items() if k not in ['type']})
    
    def lsgan_loss(pred, target):
        return torch.mean((pred - target) ** 2)
    
    if foldername != "":
        gen_output_path = foldername + "/model.pth"
        disc_output_path = foldername + "/discriminator.pth"
        gen_final_path = foldername + "/final.pth"
        disc_final_path = foldername + "/discriminator_final.pth"
    
    best_val_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    lambda_l1 = config.get('lambda_l1', 100.0)
    
    # Get train and validation loaders
    train_loader, val_loader, _ = dataset._get_loader()
    
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
                        "G_loss": f"{avg_g_loss / batch_no:.4f}",
                        "D_loss": f"{avg_d_loss / batch_no:.4f}",
                        "Epoch": epoch_no + 1,
                    },
                    refresh=True,
                )
        writer.add_scalar('Loss/Generator', avg_g_loss / batch_no, epoch_no)
        writer.add_scalar('Loss/Discriminator', avg_d_loss / batch_no, epoch_no)
        
        if (epoch_no + 1) % config['val_interval'] == 0:
            generator.eval()
            discriminator.eval()
            avg_Val_loss = 0
            
            with torch.no_grad():
                with tqdm(val_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        batch_size = clean_batch.shape[0]
                        
                        z = torch.randn(batch_size, 512, 8).to(device)
                        denoised_ecg = generator(noisy_batch, z)
                        
                        Val_loss = torch.nn.L1Loss()(denoised_ecg, clean_batch)
                        avg_Val_loss += Val_loss.item()
                        
                        it.set_postfix(
                            ordered_dict={
                                "Val_loss": f"{avg_Val_loss / batch_no:.3f}",
                                "Epoch": epoch_no + 1,
                            },
                            refresh=True,
                        )
            writer.add_scalar('Loss/Validation', avg_Val_loss / batch_no, epoch_no)
            
            if best_val_loss > avg_Val_loss / batch_no:
                best_val_loss = avg_Val_loss / batch_no
                print("\n best loss is updated to", f"{avg_Val_loss / batch_no:.4f}", "at Epoch", epoch_no + 1)
                
                if foldername != "":
                    torch.save(generator.state_dict(), gen_output_path)
                    torch.save(discriminator.state_dict(), disc_output_path)
    
    if foldername != "":
        torch.save(generator.state_dict(), gen_final_path)
        torch.save(discriminator.state_dict(), disc_final_path)    
    

def train_dl(model, config, dataset, device, foldername="", log_dir=None):

    opt = getattr(optim, config['optimizer'].get("type", "Adam"))
    optimizer = opt(model.parameters(), **{k: v for k, v in config['optimizer'].items() if k != 'type'})
    
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
    
    best_val_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    
    # Get train and validation loaders
    train_loader, val_loader, _ = dataset._get_loader()
    pbar = tqdm(total=len(train_loader), desc="Training")
    
    # training loop
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        avg_ssd = 0
        model.train()

        pbar.reset(total=len(train_loader))
        pbar.set_description(f"Epoch {epoch_no + 1}/{config['epochs']}")
        
        for batch_no, (clean_batch, noisy_batch) in enumerate(train_loader, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            optimizer.zero_grad()
            
            denoised_batch = model(noisy_batch)
            loss = criterion(denoised_batch, clean_batch)
            loss.backward()
            optimizer.step()
            
            ssd_value = SSDLoss()(denoised_batch, clean_batch).item()
            
            avg_loss += loss.item()
            avg_ssd += ssd_value
            
            pbar.set_postfix(
                ordered_dict={
                    "Train_loss": f"{avg_loss / batch_no:.4f}",
                },
                refresh=True,
            )
            pbar.update(1)

        if lr_scheduler is not None and config['lr_scheduler']['type'] != "ReduceLROnPlateau":
            lr_scheduler.step()
        
        # Log training metrics
        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
        writer.add_scalar('SSD/Train', avg_ssd / batch_no, epoch_no)
            
        if (epoch_no + 1) % config['val_interval'] == 0:
            model.eval()
            avg_val_loss = 0
            avg_ssd_valid = 0
            
            with torch.no_grad():
                with tqdm(val_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        denoised_batch = model(noisy_batch)
                        
                        loss = criterion(denoised_batch, clean_batch)
                        ssd_value = SSDLoss()(denoised_batch, clean_batch).item()
                        
                        avg_val_loss += loss.item()
                        avg_ssd_valid += ssd_value
                        
                        it.set_postfix(
                            ordered_dict={
                                "Val_loss": f"{avg_val_loss / batch_no:.4f}",
                                "Epoch": epoch_no + 1,
                            },
                            refresh=True,
                        )
            
            # Log validation metrics
            writer.add_scalar('Loss/Validation', avg_val_loss / batch_no, epoch_no)
            writer.add_scalar('SSD/Validation', avg_ssd_valid / batch_no, epoch_no)
            
            if lr_scheduler is not None and config['lr_scheduler']['type'] != "ReduceLROnPlateau":
                lr_scheduler.step(avg_val_loss / batch_no)
            
            if best_val_loss > avg_ssd_valid/batch_no:
                best_val_loss = avg_ssd_valid/batch_no
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

    opt = getattr(optim, config['optimizer'].get("type", "Adam"))
    opt0 = opt(model.base_model.unet0.parameters(), **{k: v for k, v in config['optimizer'].items() if k not in ['type']})
    opt1 = opt(model.base_model.unet1.parameters(), **{k: v for k, v in config['optimizer'].items() if k not in ['type']})

    if config['use_ema']:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.to(device)

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    
    step = 0
    best_val_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)

    train_loader, val_loader, test_loader = dataset._get_loader()
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
                pbar.set_description(f'Loss_unet0: {total_loss[0]:.4f}')
            elif config['num_unet'] == 2:
                pbar.set_description(f'Loss_unet0: {total_loss[0]:.4f},Loss_unet1: {total_loss[1]:.4f}')

            for i in range(config['num_unet']):
                writer.add_scalar(f'Loss/Train/UNet{i}', total_loss[i], step)

            if step % config['val_interval'] == 0:
                if val_loader is not None:
                    if config['use_ema']:
                        ema.ema_model.eval()
                    else:
                        model.eval()
                        
                    avg_val_loss = 0
                    
                    with torch.no_grad():
                        with tqdm(test_loader) as it:
                            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

                                if config['use_ema']:
                                    [_, denoised_batch] = ema.ema_model.sample([noisy_batch, 0], batch_size=noisy_batch.shape[0])
                                else:
                                    [_, denoised_batch] = model.sample([noisy_batch, 0], batch_size=noisy_batch.shape[0])
                                loss = SSDLoss()(denoised_batch, clean_batch).item()
                                avg_val_loss += loss
                                
                                it.set_postfix(
                                ordered_dict={
                                    "Val_loss": f"{avg_val_loss / batch_no:.4f}",
                                    "Step": step,
                                },
                                refresh=True,
                                )

                        writer.add_scalar('Loss/Validation', avg_val_loss / batch_no, step)
                    
                    if best_val_loss > avg_val_loss / batch_no:
                        best_val_loss = avg_val_loss / batch_no
                        print(f"\n Best loss updated to {avg_val_loss / batch_no:.4f} at step {step}")
                        
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
    
    opt = getattr(optim, config['optimizer'].get("type", "Adam"))
    optimizer = opt(model.parameters(), **{k: v for k, v in config['optimizer'].items() if k != 'type'})

    if config['use_ema']:
        ema = EMA(model, beta=0.9, update_every=10)
        ema.to(device)

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    
    step = 0
    best_val_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)

    train_loader, val_loader, _ = dataset._get_loader()
    data_iter = iter(train_loader)
    
    with tqdm(initial=step, total=config['steps']) as pbar:
        while step < config['steps']:
            model.train()
            total_loss = 0.
            optimizer.zero_grad()
       
            for _ in range(config['gradient_accumulate_every']):
                try:
                    clean_batch, noisy_batch, snr = next(data_iter)
                    clean_batch, noisy_batch, snr = clean_batch.to(device), noisy_batch.to(device), snr.to(device)
                except StopIteration:
                    if dataset.refresh:
                        dataset.update()
                        train_loader, val_loader, _ = dataset._get_loader()
                    else:
                        train_loader, _, _ = dataset._get_loader()
                    data_iter = iter(train_loader)
                    clean_batch, noisy_batch, snr = next(data_iter)
                    clean_batch, noisy_batch, snr = clean_batch.to(device), noisy_batch.to(device), snr.to(device)

                loss = model(noisy_batch, clean_batch, self_cond=noisy_batch, snr=snr, step=step)
                loss/= config['gradient_accumulate_every']
                total_loss += loss.item()
                
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if config['use_ema']:
                ema.update()

            step += 1
            pbar.update(1)
            pbar.set_description(f'Train_loss: {total_loss:.4f}')
            
            writer.add_scalar(f'Loss/Train', total_loss, step)

            if step % config['val_interval'] == 0:
                if val_loader is not None:
                    if config['use_ema']:
                        ema.ema_model.eval()
                    else:
                        model.eval()
                        
                    avg_val_loss = 0.
                    
                    with torch.no_grad():
                        with tqdm(val_loader) as it:
                            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

                                if config['use_ema']:
                                    [denoised_batch, _] = ema.ema_model.sample(noisy_batch, self_cond=noisy_batch)
                                else:
                                    [denoised_batch, _] = model.sample(noisy_batch, self_cond=noisy_batch)
                                
                                loss = SSDLoss()(denoised_batch, clean_batch).item()
                                avg_val_loss += loss
                                    
                                it.set_postfix(
                                ordered_dict={
                                    "Val_loss": f"{avg_val_loss / batch_no:.4f}",
                                    "Step": step,
                                },
                                refresh=True,
                                )

                        writer.add_scalar('Loss/Validation', avg_val_loss, step)
                    
                    if best_val_loss > avg_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"\n Best loss updated to {avg_val_loss:.4f} at step {step}")
                        
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


def train_ar(model, config, dataset, device, foldername="", log_dir=None):

    opt = getattr(optim, config['optimizer'].get("type", "Adam"))
    optimizer = opt(model.parameters(), **{k: v for k, v in config['optimizer'].items() if k != 'type'})
    criterion = SSDLoss()
    
    if foldername != "":
        output_path = foldername + "/model.pth"

    if config['lr_scheduler'].get("use", False):
        lr_scheduler = LRScheduler(config.get('lr_scheduler', {}))
    else:
        lr_scheduler = None

    if config['early_stopping'].get("use", False):
        early_stopping = EarlyStopping(config.get('early_stopping', {}))
    else:
        early_stopping = None
    
    best_val_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)

    train_loader, val_loader, _ = dataset._get_loader()

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        avg_loss_c = 0
        avg_loss_ar = 0
        model.train()
        train_loader, _, _ = dataset._get_loader()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                loss, loss_c, loss_ar = model(noisy_batch, clean_batch)
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item() 
                avg_loss_c += loss_c.item()
                avg_loss_ar += loss_ar.item()      
                it.set_postfix(
                    ordered_dict={
                        "Train_loss": f"{avg_loss / batch_no:.4f}",
                        "Loss_c": f"{avg_loss_c / batch_no:.4f}",
                        "Loss_ar": f"{avg_loss_ar / batch_no:.4f}",
                        "Epoch": epoch_no + 1,
                    },
                    refresh=True,
                )
            if lr_scheduler is not None and config['lr_scheduler']['type'] != "ReduceLROnPlateau":
                lr_scheduler.step()

        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
        torch.cuda.empty_cache()
            
        if (epoch_no + 1) % config['val_interval'] == 0:
            model.eval()
            avg_val_loss = 0
            avg_loss_c = 0
            avg_loss_ar = 0
            
            with torch.no_grad():
                with tqdm(val_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        
                        loss, loss_c, loss_ar = model(noisy_batch, clean_batch)
                        avg_val_loss += loss.item()
                        avg_loss_c += loss_c.item()
                        avg_loss_ar += loss_ar.item()
                        
                        it.set_postfix(
                            ordered_dict={
                                "Val_loss": f"{avg_val_loss / batch_no:.4f}",
                                "Loss_c": f"{avg_loss_c / batch_no:.4f}",
                                "Loss_ar": f"{avg_loss_ar / batch_no:.4f}",
                                "Epoch": epoch_no + 1,
                            },
                            refresh=True,
                        )
            
            writer.add_scalar('Loss/Validation', avg_val_loss / batch_no, epoch_no)
      
            if config['lr_scheduler']['type'] == "ReduceLROnPlateau":
                lr_scheduler.step(avg_val_loss / batch_no)
            
            if best_val_loss > avg_val_loss/batch_no:
                best_val_loss = avg_val_loss/batch_no
                print("\n best loss is updated to ",f"{avg_val_loss / batch_no:.4f}","at Epoch", epoch_no+1)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                    
            if early_stopping is not None:
                if early_stopping.step(avg_val_loss/batch_no):
                    print(f"\nEarly stopping triggered after {epoch_no+1} epochs!")
                    break
