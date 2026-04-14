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

# Import custom modules
from flight_dynamics.normalizer import NormalizerFactory

# --- MAMBA IMPORT REPLACING S4 ---
from models.mamba import Mamba, MambaConfig

FLIGHT_DYNAMICS_DIR = 'flight_dynamics'  # Default working directory
RUNS_DIRECTORY = os.path.join(FLIGHT_DYNAMICS_DIR, "runs")
os.makedirs(RUNS_DIRECTORY, exist_ok=True)

# ==========================================
# 1. DIRECTORY & CONFIG MANAGEMENT
# ==========================================
def create_directory_structure(experiment_directory, experiment_name, conf_idx=None):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    if conf_idx is not None:
        dir_name = f"{experiment_name}{conf_idx}_{timestamp}"
    else:
        dir_name = f"{experiment_name}_{timestamp}"

    current_instance_dir = os.path.join(experiment_directory, dir_name)
    checkpoint_dir = os.path.join(current_instance_dir, 'checkpoints')
    
    os.makedirs(current_instance_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return current_instance_dir, checkpoint_dir

# ==========================================
# 2. DATASET & COLLATION
# ==========================================
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
    def __init__(self, X_data, y_data, feature_scaler=None, target_scaler=None, is_train=True):
        self.X = X_data
        self.y = y_data
        assert len(self.X) == len(self.y), "Mismatch between number of feature and target trajectories."

        if feature_scaler is not None:
            if is_train:
                self.X = feature_scaler.fit_transform(self.X)
            else:
                self.X = feature_scaler.transform(self.X)

        if target_scaler is not None:
            if is_train:
                self.y = target_scaler.fit_transform(self.y)
            else:
                self.y = target_scaler.transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. MODEL ARCHITECTURE (MAMBA)
# ==========================================
class MambaSeq2SeqModel(nn.Module):
    def __init__(self, d_input=12, d_output=6, d_model=128, n_layers=4):
        super().__init__()
        
        # 1. Input Projection
        self.encoder = nn.Linear(d_input, d_model)
        
        # 2. Mamba Core
        config = MambaConfig(
            d_model=d_model, 
            n_layers=n_layers, 
            d_state=16, 
            expand_factor=2,
            pscan=True  # Uses your parallel scan for fast training
        )
        self.mamba_core = Mamba(config)
        
        # 3. Output Projection
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        # x shape: (B, L, d_input)
        x = self.encoder(x)  
        x = self.mamba_core(x)
        x = self.decoder(x)  
        return x

# ==========================================
# 4. UTILITIES & TRAINING LOOPS
# ==========================================
def setup_optimizer(model, lr, weight_decay, epochs):
    """
    Adapted specifically for Mamba: 
    Extracts parameters tagged with _no_weight_decay (like the A matrix and D)
    as well as 1D tensors (biases/norms) to prevent them from decaying.
    """
    decay_params = []
    no_decay_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # mamba.py tags specific SSM parameters
        if getattr(p, "_no_weight_decay", False):
            no_decay_params.append(p)
        # standard practice: no weight decay for biases and norm weights
        elif p.ndim <= 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
            
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

# def train_epoch(epoch, model, dataloader, optimizer, device, disable_tqdm=False):
#     model.train()
#     train_loss = 0
#     pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
    
#     for batch_idx, (inputs, targets, mask) in pbar:
#         inputs = inputs.to(device, dtype=torch.float32)
#         targets = targets.to(device, dtype=torch.float32)
#         mask = mask.to(device, dtype=torch.float32)
        
#         # 1. Check Data for NaNs (Culprit: Normalizer)
#         if torch.isnan(inputs).any():
#             print(f"\n[DEBUG] FATAL: NaNs detected in input data at batch {batch_idx}! Check your Normalizer.")
#             sys.exit(1)
#         if torch.isnan(targets).any():
#             print(f"\n[DEBUG] FATAL: NaNs detected in target data at batch {batch_idx}! Check your Normalizer.")
#             sys.exit(1)

#         optimizer.zero_grad()
#         outputs = model(inputs)
        
#         # 2. Check Forward Pass for NaNs (Culprit: Bad Model Initialization or previous bad gradient)
#         if torch.isnan(outputs).any():
#             print(f"\n[DEBUG] FATAL: NaNs detected in model OUTPUTS at batch {batch_idx} (Epoch {epoch})!")
#             sys.exit(1)
        
#         loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1) 
#         masked_loss = loss_unreduced * mask
        
#         # 3. Check Mask (Culprit: Divide by Zero in Loss)
#         if mask.sum() == 0:
#             print(f"\n[DEBUG] FATAL: mask.sum() is zero at batch {batch_idx}!")
#             sys.exit(1)
            
#         loss = masked_loss.sum() / mask.sum()
        
#         loss.backward()
        
#         # 4. Check Gradients (Culprit: Exploding Gradients before clipping)
#         for name, param in model.named_parameters():
#             if param.grad is not None and torch.isnan(param.grad).any():
#                 print(f"\n[DEBUG] FATAL: NaN gradient detected in parameter: {name} at batch {batch_idx}!")
#                 sys.exit(1)
                
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         train_loss += loss.item()
#         pbar.set_description(f'Train Epoch: {epoch} | Masked MSE: {train_loss/(batch_idx+1):.4f}')

def train_epoch(epoch, model, dataloader, optimizer, device, disable_tqdm=False):
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
    
    for batch_idx, (inputs, targets, mask) in pbar:
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1) 
        masked_loss = loss_unreduced * mask
        loss = masked_loss.sum() / mask.sum()
        
        loss.backward()
        
        # ---> MAMBA ADDITION: CRITICAL FOR STABILITY <---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        train_loss += loss.item()
        pbar.set_description(f'Train Epoch: {epoch} | Masked MSE: {train_loss/(batch_idx+1):.4f}')

def evaluate(epoch, model, dataloader, device, optimizer, checkpoint_dir, is_val=True, best_val_loss=float('inf'), disable_tqdm=False):
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
        for batch_idx, (inputs, targets, mask) in pbar:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)
            
            outputs = model(inputs) 
            
            loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1)
            masked_loss = loss_unreduced * mask
            loss = masked_loss.sum() / mask.sum()

            eval_loss += loss.item()
            avg_loss = eval_loss / (batch_idx + 1)

            mode = "Val" if is_val else "Test"
            pbar.set_description(f'{mode} Epoch: {epoch} | Masked MSE: {avg_loss:.4f}')

    if is_val and avg_loss < best_val_loss:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'val_loss': avg_loss
        }
        save_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, save_path)
        best_val_loss = avg_loss

    return avg_loss, best_val_loss

# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
def run_training_pipeline(config, X_train, y_train, X_val, y_val, current_instance_dir, checkpoint_dir, device, conf_idx=None, disable_tqdm=False):
    run_label = f"Config {conf_idx}" if conf_idx else "Single Config"
    print(f"\n[{run_label}] Output Directory: {current_instance_dir}")
    
    try:
        EPOCHS = config["epochs"]
        BATCH_SIZE = config["batch_size"]
        LR = config["lr"]
        WEIGHT_DECAY = config["weight_decay"]
        NUM_WORKERS = config["num_workers"]
        D_MODEL = config["d_model"]
        N_LAYERS = config["n_layers"]
        FEATURE_NORMALIZER = config["feature_normalizer"]
        TARGET_NORMALIZER = config["target_normalizer"]
    except KeyError as e:
        raise KeyError(f"Missing required hyperparameter in JSON config file: {e}")

    config_path_out = os.path.join(current_instance_dir, 'config.json')
    with open(config_path_out, 'w') as f:
        json.dump(config, f, indent=4)

    feature_scaler = NormalizerFactory.create(FEATURE_NORMALIZER, global_normalizer=True)
    target_scaler = NormalizerFactory.create(TARGET_NORMALIZER, global_normalizer=True)

    trainset = InsectFlightSeq2SeqDataset(X_train, y_train, feature_scaler=feature_scaler, target_scaler=target_scaler, is_train=True)
    valset = InsectFlightSeq2SeqDataset(X_val, y_val, feature_scaler=feature_scaler, target_scaler=target_scaler, is_train=False)

    with open(os.path.join(current_instance_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(current_instance_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_collate)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_collate)

    # Initialize Mamba Model
    model = MambaSeq2SeqModel(d_input=12, d_output=6, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
    optimizer, scheduler = setup_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS)

    best_val_loss = float('inf')
    print(f"[{run_label}] Starting Training Loop for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_epoch(epoch, model, trainloader, optimizer, device, disable_tqdm=disable_tqdm)
        val_loss, best_val_loss = evaluate(
            epoch, model, valloader, device, 
            optimizer=optimizer, checkpoint_dir=checkpoint_dir, 
            is_val=True, best_val_loss=best_val_loss, disable_tqdm=disable_tqdm
        )
        scheduler.step()
        print(f"[{run_label}] Epoch {epoch} | Val MSE: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    print(f"[{run_label}] Completed. Best model saved to: {checkpoint_dir}")

    del model
    del optimizer
    del trainloader
    del valloader
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train Mamba Model with Grid Search")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument('--name', type=str, default="default_mamba_run", help="Name prefix for the output directory")
    parser.add_argument('--disable_tqdm', action='store_true', help="Manually turn off progress bars")
    args = parser.parse_args()

    DISABLE_PBARS = args.disable_tqdm or ('SLURM_JOB_ID' in os.environ)
    if DISABLE_PBARS:
        print("==> Slurm environment detected (or flag passed). Disabling tqdm progress bars to keep logs clean.")

    with open(args.config, 'r') as f:
        raw_config = json.load(f)

    FEATURES_FILE = raw_config.pop("features_file", "data/features.pt")
    TARGETS_FILE = raw_config.pop("targets_file", "data/targets.pt")
    TRAIN_RATIO = raw_config.pop("train_split_ratio", 0.85)

    listified_config = {k: (v if isinstance(v, list) else [v]) for k, v in raw_config.items()}
    keys, values = zip(*listified_config.items())
    config_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    num_configs = len(config_combinations)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        print(f"==> CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.backends.cuda.cufft_plan_cache.max_size = 0

    print(f"==> Initiating on {DEVICE.upper()}")
    print(f"==> Found {num_configs} configuration(s) to execute.")

    print(f"==> Loading master dataset files...")
    X_full = torch.load(FEATURES_FILE, map_location='cpu')
    y_full = torch.load(TARGETS_FILE, map_location='cpu')

    total_samples = len(X_full)
    assert total_samples == len(y_full), "Mismatch in total features and targets."

    torch.manual_seed(42)
    indices = torch.randperm(total_samples).tolist()
    
    train_size = int(TRAIN_RATIO * total_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train = [X_full[i] for i in train_indices]
    y_train = [y_full[i] for i in train_indices]
    X_val = [X_full[i] for i in val_indices]
    y_val = [y_full[i] for i in val_indices]
    print(f"Dataset split: {len(X_train)} Train | {len(X_val)} Val")

    EXPERIMENT_DIR = os.path.join(RUNS_DIRECTORY, args.name)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    overall_best_loss = float('inf')
    overall_best_run_name = None

    for i, config_instance in enumerate(config_combinations, start=1):
        if num_configs > 1:
            print(f"\n==========================================")
            print(f"   STARTING RUN {i} of {num_configs}")
            print(f"==========================================")
            
        run_idx = i if num_configs > 1 else None
        current_instance_directory, checkpoint_dir = create_directory_structure(EXPERIMENT_DIR, args.name, run_idx)
        
        try:
            start_time = time.time()
            run_loss = run_training_pipeline(
                config_instance, 
                X_train, y_train, X_val, y_val, 
                current_instance_directory, checkpoint_dir, 
                DEVICE, conf_idx=run_idx,
                disable_tqdm=DISABLE_PBARS
            )
            end_time = time.time()
            elapsed_seconds = end_time - start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_string = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"

            print(f"\n[Config {run_idx}] Finished in: {time_string}")

            all_results_path = os.path.join(EXPERIMENT_DIR, 'all_configs_results.txt')
            with open(all_results_path, 'a') as f:
                f.write(f"configuration {run_idx} best loss - {run_loss:.6f} | Time: {time_string}\n")

            if run_loss < overall_best_loss:
                overall_best_loss = run_loss
                overall_best_run_name = os.path.basename(current_instance_directory)
                
                live_tracker_path = os.path.join(EXPERIMENT_DIR, 'best_so_far.txt')
                with open(live_tracker_path, 'w') as f:
                    f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Current Best Model: {overall_best_run_name}\n")
                    f.write(f"Current Lowest Validation MSE: {overall_best_loss:.6f}\n")

        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] CUDA Out of Memory on Config {run_idx}. This hyperparameter combination is too large for the GPU. Skipping...")
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            continue
            
        except Exception as e:
            print(f"\n[!] An unexpected error occurred on Config {run_idx}: {e}")
            continue

    if num_configs > 1 and overall_best_run_name is not None:
        summary_path = os.path.join(EXPERIMENT_DIR, 'best_model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Grid Search Completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total configurations tested: {num_configs}\n")
            f.write("-" * 40 + "\n")
            f.write(f"BEST MODEL RUN: {overall_best_run_name}\n")
            f.write(f"LOWEST VALIDATION MSE: {overall_best_loss:.6f}\n")
        print(f"\n==> Best model summary saved to: {summary_path}")

if __name__ == '__main__':
    main()