import os
import sys
import json
import argparse
import itertools
import math
import gc
import pickle
from datetime import datetime, time
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from flight_dynamics.normalizer import NormalizerFactory

from models.mamba import Mamba, MambaConfig

FLIGHT_DYNAMICS_DIR = 'flight_dynamics'  
RUNS_DIRECTORY = os.path.join(FLIGHT_DYNAMICS_DIR, "runs")
os.makedirs(RUNS_DIRECTORY, exist_ok=True)

def create_directory_structure(experiment_directory, experiment_name, conf_idx=None):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    dir_name = f"{experiment_name}{conf_idx}_{timestamp}" if conf_idx is not None else f"{experiment_name}_{timestamp}"
    current_instance_dir = os.path.join(experiment_directory, dir_name)
    checkpoint_dir = os.path.join(current_instance_dir, 'checkpoints')
    
    os.makedirs(current_instance_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return current_instance_dir, checkpoint_dir

def save_grid_state(filepath, state_dict):
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(state_dict, f, indent=4)
    os.replace(temp_path, filepath)

def pad_collate(batch):
    Xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    
    lengths = torch.tensor([len(x) for x in Xs])
    max_len_actual = lengths.max().item()
    next_pow_2 = 2 ** math.ceil(math.log2(max_len_actual))
    
    X_padded = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    
    pad_amount = next_pow_2 - max_len_actual
    if pad_amount > 0:
        X_padded = F.pad(X_padded, (0, 0, 0, pad_amount), value=0.0)
        y_padded = F.pad(y_padded, (0, 0, 0, pad_amount), value=0.0)
    
    mask = torch.arange(next_pow_2).expand(len(lengths), next_pow_2) < lengths.unsqueeze(1)
    return X_padded, y_padded, mask

class InsectFlightSeq2SeqDataset(Dataset):
    def __init__(self, X_data, y_data, feature_scaler=None, target_scaler=None, fit_scalers=True):
        self.X = X_data
        self.y = y_data
        assert len(self.X) == len(self.y), "Mismatch between feature and target trajectories."

        if feature_scaler is not None:
            if fit_scalers: self.X = feature_scaler.fit_transform(self.X)
            else: self.X = feature_scaler.transform(self.X)

        if target_scaler is not None:
            if fit_scalers: self.y = target_scaler.fit_transform(self.y)
            else: self.y = target_scaler.transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MambaSeq2SeqModel(nn.Module):
    def __init__(self, d_input=12, d_output=6, d_model=128, n_layers=4):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        config = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=16, expand_factor=2, pscan=True)
        self.mamba_core = Mamba(config)
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)  
        x = self.mamba_core(x)
        x = self.decoder(x)  
        return x

def setup_optimizer(model, lr, weight_decay, epochs):
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if getattr(p, "_no_weight_decay", False) or p.ndim <= 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
            
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def train_epoch(epoch, model, dataloader, optimizer, device, disable_tqdm=False):
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
    
    for batch_idx, (inputs, targets, mask) in pbar:
        inputs, targets, mask = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32), mask.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1) 
        masked_loss = loss_unreduced * mask
        loss = masked_loss.sum() / mask.sum()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        pbar.set_description(f'Train Epoch: {epoch} | Masked MSE: {train_loss/(batch_idx+1):.4f}')

def evaluate(epoch, model, dataloader, device, is_val=True, disable_tqdm=False):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
        for batch_idx, (inputs, targets, mask) in pbar:
            inputs, targets, mask = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32), mask.to(device, dtype=torch.float32)
            outputs = model(inputs) 
            
            loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1)
            masked_loss = loss_unreduced * mask
            loss = masked_loss.sum() / mask.sum()
            eval_loss += loss.item()
            
            mode = "Val" if is_val else "Test"
            pbar.set_description(f'{mode} Epoch: {epoch} | Masked MSE: {eval_loss/(batch_idx+1):.4f}')
            
    return eval_loss / len(dataloader)

def run_training_pipeline(config, X_train, y_train, X_val, y_val, current_instance_dir, checkpoint_dir, device, conf_idx=None, disable_tqdm=False, is_resume=False):
    run_label = f"Config {conf_idx}" if conf_idx else "Single Config"
    print(f"\n[{run_label}] Output Directory: {current_instance_dir}")
    
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY = config["epochs"], config["batch_size"], config["lr"], config["weight_decay"]
    NUM_WORKERS, D_MODEL, N_LAYERS = config["num_workers"], config["d_model"], config["n_layers"]
    
    with open(os.path.join(current_instance_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    feature_scaler_path = os.path.join(current_instance_dir, 'feature_scaler.pkl')
    target_scaler_path = os.path.join(current_instance_dir, 'target_scaler.pkl')
    
    if is_resume and os.path.exists(feature_scaler_path):
        print(f"[{run_label}] Loading existing scalers...")
        with open(feature_scaler_path, 'rb') as f: feature_scaler = pickle.load(f)
        with open(target_scaler_path, 'rb') as f: target_scaler = pickle.load(f)
        fit_scalers = False
    else:
        feature_scaler = NormalizerFactory.create(config["feature_normalizer"], global_normalizer=True)
        target_scaler = NormalizerFactory.create(config["target_normalizer"], global_normalizer=True)
        fit_scalers = True

    trainset = InsectFlightSeq2SeqDataset(X_train, y_train, feature_scaler, target_scaler, fit_scalers=fit_scalers)
    valset = InsectFlightSeq2SeqDataset(X_val, y_val, feature_scaler, target_scaler, fit_scalers=False)

    if fit_scalers:
        with open(feature_scaler_path, 'wb') as f: pickle.dump(feature_scaler, f)
        with open(target_scaler_path, 'wb') as f: pickle.dump(target_scaler, f)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_collate)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_collate)

    model = MambaSeq2SeqModel(d_input=12, d_output=6, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
    optimizer, scheduler = setup_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS)

    start_epoch = 1
    best_val_loss = float('inf')
    latest_ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    if is_resume and os.path.exists(latest_ckpt_path):
        print(f"[{run_label}] Found latest checkpoint. Resuming training state...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        torch.set_rng_state(ckpt['rng_state'])
        if 'cuda_rng_state' in ckpt and device == 'cuda':
            torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
            
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"[{run_label}] Resuming from Epoch {start_epoch} with Best Val Loss: {best_val_loss:.4f}")

    print(f"[{run_label}] Starting Training Loop...")
    for epoch in range(start_epoch, EPOCHS + 1):
        train_epoch(epoch, model, trainloader, optimizer, device, disable_tqdm=disable_tqdm)
        val_loss = evaluate(epoch, model, valloader, device, is_val=True, disable_tqdm=disable_tqdm)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if device == 'cuda' else None
        }, latest_ckpt_path)
            
        print(f"[{run_label}] Epoch {epoch} | Val MSE: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    print(f"[{run_label}] Completed. Checkpoints saved to: {checkpoint_dir}")
    del model, optimizer, trainloader, valloader
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()

    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train Mamba Model with Grid Search and Resume capabilities")
    parser.add_argument('--config', type=str, help="Path to the JSON configuration file (Required for new runs)")
    parser.add_argument('--name', type=str, default="default_mamba_run", help="Name prefix for new output directory")
    parser.add_argument('--resume_dir', type=str, help="Path to an existing experiment directory to resume (Ignores --config)")
    parser.add_argument('--disable_tqdm', action='store_true', help="Manually turn off progress bars")
    args = parser.parse_args()

    DISABLE_PBARS = args.disable_tqdm or ('SLURM_JOB_ID' in os.environ)
    if DISABLE_PBARS: print("==> Slurm environment detected. Disabling tqdm progress bars.")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        print(f"==> CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True

    if args.resume_dir:
        print(f"==> RESUME MODE INITIATED: {args.resume_dir}")
        EXPERIMENT_DIR = args.resume_dir
        
        grid_state_path = os.path.join(EXPERIMENT_DIR, 'grid_state.json')
        if not os.path.exists(grid_state_path):
            raise FileNotFoundError(f"Cannot resume. No grid_state.json found in {EXPERIMENT_DIR}")
        with open(grid_state_path, 'r') as f:
            grid_state = json.load(f)
            
        print("==> Loading exact Train/Val indices from previous run...")
        train_indices = torch.load(os.path.join(EXPERIMENT_DIR, 'train_indices.pt')).tolist()
        val_indices = torch.load(os.path.join(EXPERIMENT_DIR, 'val_indices.pt')).tolist()
        
        FEATURES_FILE = grid_state.get('features_file', "data/features.pt")
        TARGETS_FILE = grid_state.get('targets_file', "data/targets.pt")
        
    else:
        if not args.config: raise ValueError("Must provide --config for a fresh run, or --resume_dir to resume.")
        print("==> FRESH RUN INITIATED")
        
        EXPERIMENT_DIR = os.path.join(RUNS_DIRECTORY, f"{args.name}_{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}")
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        
        with open(args.config, 'r') as f: raw_config = json.load(f)
        FEATURES_FILE = raw_config.pop("features_file", "data/features.pt")
        TARGETS_FILE = raw_config.pop("targets_file", "data/targets.pt")
        TRAIN_RATIO = raw_config.pop("train_split_ratio", 0.85)

        listified_config = {k: (v if isinstance(v, list) else [v]) for k, v in raw_config.items()}
        keys, values = zip(*listified_config.items())
        config_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        print("==> Generating and saving new Train/Val indices...")
        total_samples = len(torch.load(FEATURES_FILE, map_location='cpu'))
        torch.manual_seed(42)
        indices = torch.randperm(total_samples)
        train_size = int(TRAIN_RATIO * total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        torch.save(train_indices, os.path.join(EXPERIMENT_DIR, 'train_indices.pt'))
        torch.save(val_indices, os.path.join(EXPERIMENT_DIR, 'val_indices.pt'))
        train_indices, val_indices = train_indices.tolist(), val_indices.tolist()

        grid_state = {
            "overall_best_loss": float('inf'),
            "overall_best_run_name": None,
            "current_idx": 0,
            "features_file": FEATURES_FILE,
            "targets_file": TARGETS_FILE,
            "runs": [{"config": combo, "dir": None} for combo in config_combinations]
        }
        save_grid_state(os.path.join(EXPERIMENT_DIR, 'grid_state.json'), grid_state)

    print(f"==> Loading master dataset files...")
    X_full = torch.load(FEATURES_FILE, map_location='cpu')
    y_full = torch.load(TARGETS_FILE, map_location='cpu')

    X_train, y_train = [X_full[i] for i in train_indices], [y_full[i] for i in train_indices]
    X_val, y_val = [X_full[i] for i in val_indices], [y_full[i] for i in val_indices]
    print(f"Dataset split: {len(X_train)} Train | {len(X_val)} Val")

    num_configs = len(grid_state["runs"])
    print(f"==> Executing {num_configs - grid_state['current_idx']} pending configuration(s).")

    for i in range(grid_state["current_idx"], num_configs):
        run_info = grid_state["runs"][i]
        config_instance = run_info["config"]
        run_idx = i + 1 
        
        if num_configs > 1:
            print(f"\n==========================================")
            print(f"   STARTING RUN {run_idx} of {num_configs}")
            print(f"==========================================")
            
        if run_info["dir"] is not None and os.path.exists(run_info["dir"]):
            current_instance_directory = run_info["dir"]
            checkpoint_dir = os.path.join(current_instance_directory, 'checkpoints')
            is_resume = True
        else:
            current_instance_directory, checkpoint_dir = create_directory_structure(EXPERIMENT_DIR, args.name if not args.resume_dir else "resume_run", run_idx if num_configs > 1 else None)
            run_info["dir"] = current_instance_directory
            save_grid_state(os.path.join(EXPERIMENT_DIR, 'grid_state.json'), grid_state)
            is_resume = False
            
        try:
            start_time = time.time()
            run_loss = run_training_pipeline(
                config_instance, 
                X_train, y_train, X_val, y_val, 
                current_instance_directory, checkpoint_dir, 
                DEVICE, conf_idx=run_idx, disable_tqdm=DISABLE_PBARS, is_resume=is_resume
            )
            
            elapsed_seconds = time.time() - start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_string = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
            print(f"\n[Config {run_idx}] Finished in: {time_string}")

            with open(os.path.join(EXPERIMENT_DIR, 'all_configs_results.txt'), 'a') as f:
                f.write(f"configuration {run_idx} best loss - {run_loss:.6f} | Time: {time_string}\n")

            if run_loss < grid_state["overall_best_loss"]:
                grid_state["overall_best_loss"] = run_loss
                grid_state["overall_best_run_name"] = os.path.basename(current_instance_directory)
                with open(os.path.join(EXPERIMENT_DIR, 'best_so_far.txt'), 'w') as f:
                    f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Current Best Model: {grid_state['overall_best_run_name']}\n")
                    f.write(f"Current Lowest Validation MSE: {grid_state['overall_best_loss']:.6f}\n")

            grid_state["current_idx"] = i + 1
            save_grid_state(os.path.join(EXPERIMENT_DIR, 'grid_state.json'), grid_state)

        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] CUDA Out of Memory on Config {run_idx}. Skipping...")
            if DEVICE == 'cuda': torch.cuda.empty_cache()
            grid_state["current_idx"] = i + 1
            save_grid_state(os.path.join(EXPERIMENT_DIR, 'grid_state.json'), grid_state)
            continue
            
        except Exception as e:
            print(f"\n[!] An unexpected error occurred on Config {run_idx}: {e}")
            raise e

    if num_configs > 1 and grid_state["overall_best_run_name"] is not None:
        summary_path = os.path.join(EXPERIMENT_DIR, 'best_model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Grid Search Completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"BEST MODEL RUN: {grid_state['overall_best_run_name']}\n")
            f.write(f"LOWEST VALIDATION MSE: {grid_state['overall_best_loss']:.6f}\n")
        print(f"\n==> Best model summary saved to: {summary_path}")

if __name__ == '__main__':
    main()