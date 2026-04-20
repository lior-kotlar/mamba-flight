import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Add parent directory to sys.path to allow imports from models and flight_dynamics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from flight_dynamics.main_flight import MambaSeq2SeqModel, InsectFlightSeq2SeqDataset, pad_collate


def is_run_directory(path):
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, 'config.json'))
        and os.path.exists(os.path.join(path, 'feature_scaler.pkl'))
        and os.path.exists(os.path.join(path, 'target_scaler.pkl'))
        and os.path.isdir(os.path.join(path, 'checkpoints'))
    )


def resolve_run_directories(directory_path):
    directory_path = os.path.abspath(directory_path)

    if is_run_directory(directory_path):
        return [directory_path]

    grid_state_path = os.path.join(directory_path, 'grid_state.json')
    run_dirs = []

    if os.path.exists(grid_state_path):
        with open(grid_state_path, 'r') as f:
            grid_state = json.load(f)

        for run in grid_state.get('runs', []):
            run_dir = run.get('dir')
            if run_dir:
                abs_run_dir = os.path.abspath(run_dir)
                if is_run_directory(abs_run_dir):
                    run_dirs.append(abs_run_dir)

    if not run_dirs:
        for child in sorted(os.listdir(directory_path)):
            child_path = os.path.join(directory_path, child)
            if is_run_directory(child_path):
                run_dirs.append(os.path.abspath(child_path))

    if not run_dirs:
        raise FileNotFoundError(
            f"No valid run directories were found in: {directory_path}. "
            "Provide either a specific configuration directory or a parent experiment directory."
        )

    return run_dirs


def load_prediction_inputs(dataset_path):
    dataset_obj = torch.load(dataset_path, map_location='cpu')

    if isinstance(dataset_obj, dict):
        features = None
        targets = None

        for key in ('features', 'X', 'inputs', 'x'):
            if key in dataset_obj:
                features = dataset_obj[key]
                break

        for key in ('targets', 'y', 'labels'):
            if key in dataset_obj:
                targets = dataset_obj[key]
                break

        if features is None:
            raise ValueError(
                "Dataset dict format is not supported. Expected one of keys: "
                "['features', 'X', 'inputs', 'x']."
            )

        return features, targets

    if isinstance(dataset_obj, (list, tuple)):
        if len(dataset_obj) == 2:
            return dataset_obj[0], dataset_obj[1]
        return dataset_obj, None

    if isinstance(dataset_obj, torch.Tensor):
        return dataset_obj, None

    raise ValueError(
        "Unsupported dataset format in .pt file. "
        "Expected Tensor, list/tuple, or dict with feature keys."
    )


def to_sequence_list(sequence_obj, source_name):
    if isinstance(sequence_obj, list):
        return [seq.detach().cpu() if isinstance(seq, torch.Tensor) else torch.tensor(seq) for seq in sequence_obj]

    if isinstance(sequence_obj, torch.Tensor):
        if sequence_obj.ndim != 3:
            raise ValueError(f"{source_name} tensor must have shape (N, T, 6), got shape {tuple(sequence_obj.shape)}")
        return [sequence_obj[i].detach().cpu() for i in range(sequence_obj.shape[0])]

    raise ValueError(f"Unsupported {source_name} format. Expected list[Tensor] or Tensor(N,T,6).")


def load_ground_truth_sequences(ground_truth_path):
    gt_obj = torch.load(ground_truth_path, map_location='cpu')

    if isinstance(gt_obj, dict):
        for key in ('targets', 'y', 'labels', 'ground_truth', 'gt'):
            if key in gt_obj:
                return to_sequence_list(gt_obj[key], f"ground truth dict key '{key}'")
        raise ValueError(
            "Ground truth dict format is not supported. Expected one of keys: "
            "['targets', 'y', 'labels', 'ground_truth', 'gt']."
        )

    if isinstance(gt_obj, (list, tuple)):
        if len(gt_obj) == 2:
            first, second = gt_obj[0], gt_obj[1]
            if isinstance(second, torch.Tensor) and second.ndim >= 2:
                return to_sequence_list(second, 'ground truth tuple second element')
            return to_sequence_list(first, 'ground truth tuple first element')
        return to_sequence_list(list(gt_obj), 'ground truth list/tuple')

    if isinstance(gt_obj, torch.Tensor):
        return to_sequence_list(gt_obj, 'ground truth tensor')

    raise ValueError(
        "Unsupported ground truth .pt format. Expected Tensor, list/tuple, or dict with target keys."
    )

def load_run(run_dir, device='cpu'):
    # Load config
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        # Try to find it in the parent directory (for grid search runs)
        config_path = os.path.join(os.path.dirname(run_dir), 'grid_state.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                grid_state = json.load(f)
                # Find which run this is
                for run in grid_state['runs']:
                    if run['dir'] and os.path.abspath(run['dir']) == os.path.abspath(run_dir):
                        config = run['config']
                        break
                else:
                    raise FileNotFoundError(f"Could not find config for run {run_dir}")
        else:
            raise FileNotFoundError(f"Could not find config.json or grid_state.json for {run_dir}")
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Load scalers
    with open(os.path.join(run_dir, 'feature_scaler.pkl'), 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(os.path.join(run_dir, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)

    # Load model
    checkpoint_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(run_dir, 'checkpoints', 'latest_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint file in {os.path.join(run_dir, 'checkpoints')}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MambaSeq2SeqModel(
        d_input=12, 
        d_output=6, 
        d_model=config['d_model'], 
        n_layers=config['n_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, feature_scaler, target_scaler

def build_dataset(features, targets, feature_scaler, target_scaler):
    if targets is None:
        # Build placeholder targets so existing Dataset/Collate pipeline can be reused.
        if isinstance(features, torch.Tensor):
            targets = torch.zeros(features.shape[0], features.shape[1], 6, dtype=features.dtype)
        elif isinstance(features, list):
            targets = [torch.zeros(x.shape[0], 6, dtype=x.dtype) for x in features]
        else:
            raise ValueError("Unsupported feature container type for dataset construction.")

    return InsectFlightSeq2SeqDataset(features, targets, feature_scaler, target_scaler, fit_scalers=False)

def predict(model, dataset, device, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, mask in dataloader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            
            # Apply mask to keep only valid timesteps
            for i in range(len(inputs)):
                valid_len = mask[i].sum().item()
                all_preds.append(outputs[i, :int(valid_len)].cpu())
                all_targets.append(targets[i, :int(valid_len)].cpu())
                
    return all_preds, all_targets


def inverse_transform_sequence_list(scaler, seq_list):
    return [scaler.inverse_transform(seq) for seq in seq_list]


def make_wing_angle_figure(pred_seq, gt_seq, sample_idx, run_name):
    subplot_titles = [
        'Left Wing Angle 1', 'Right Wing Angle 1',
        'Left Wing Angle 2', 'Right Wing Angle 2',
        'Left Wing Angle 3', 'Right Wing Angle 3',
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    for row_idx in range(3):
        left_dim = row_idx
        right_dim = row_idx + 3
        row = row_idx + 1

        gt_x = np.arange(gt_seq.shape[0])
        pred_x = np.arange(pred_seq.shape[0])

        fig.add_trace(
            go.Scatter(x=gt_x, y=gt_seq[:, left_dim], mode='lines', name='Ground Truth', line=dict(color='#1f77b4')),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_seq[:, left_dim], mode='lines', name='Prediction', line=dict(color='#d62728', dash='dash')),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=gt_x, y=gt_seq[:, right_dim], mode='lines', name='Ground Truth', line=dict(color='#1f77b4'), showlegend=False),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_seq[:, right_dim], mode='lines', name='Prediction', line=dict(color='#d62728', dash='dash'), showlegend=False),
            row=row,
            col=2,
        )

        fig.update_yaxes(title_text='Angle', row=row, col=1)
        fig.update_yaxes(title_text='Angle', row=row, col=2)

    fig.update_xaxes(title_text='Time Step', row=3, col=1)
    fig.update_xaxes(title_text='Time Step', row=3, col=2)

    fig.update_layout(
        title_text=f"Prediction vs Ground Truth | {run_name} | Sample {sample_idx}",
        height=1000,
        width=1400,
        template='plotly_white',
    )

    return fig


def save_prediction_plots(predictions, ground_truth_sequences, run_dir):
    if len(predictions) == 0:
        print("No predictions found. Skipping plot generation.")
        return

    total = min(len(predictions), len(ground_truth_sequences))

    for sample_idx in range(total):
        pred_seq = predictions[sample_idx].detach().cpu()
        gt_seq = ground_truth_sequences[sample_idx].detach().cpu()

        if pred_seq.ndim != 2 or gt_seq.ndim != 2 or pred_seq.shape[1] != 6 or gt_seq.shape[1] != 6:
            print(
                f"[!] Warning: Skipping sample {sample_idx} due to invalid shape. "
                f"pred={tuple(pred_seq.shape)}, gt={tuple(gt_seq.shape)}"
            )
            continue

        common_len = min(pred_seq.shape[0], gt_seq.shape[0])
        pred_seq = pred_seq[:common_len]
        gt_seq = gt_seq[:common_len]

        fig = make_wing_angle_figure(
            pred_seq=pred_seq.numpy(),
            gt_seq=gt_seq.numpy(),
            sample_idx=sample_idx,
            run_name=os.path.basename(run_dir),
        )

        html_path = os.path.join(run_dir, f'sample_{sample_idx:04d}.html')
        fig.write_html(html_path, include_plotlyjs='cdn')

    print(f"Saved {total} prediction plot html file(s) to: {run_dir}")

def run_prediction_for_directory(run_dir, dataset_features, dataset_targets, ground_truth_sequences, device, output_name):
    print(f"\n=== Predicting for run directory: {run_dir}")
    model, config, feature_scaler, target_scaler = load_run(run_dir, device)

    dataset = build_dataset(dataset_features, dataset_targets, feature_scaler, target_scaler)
    preds, targets = predict(model, dataset, device, batch_size=config.get('batch_size', 32))

    unnormalized_predictions = inverse_transform_sequence_list(target_scaler, preds)

    save_payload = {
        'predictions': unnormalized_predictions,
        'run_dir': run_dir,
        'checkpoint_batch_size': config.get('batch_size', 32),
    }

    if dataset_targets is not None:
        unnormalized_targets = inverse_transform_sequence_list(target_scaler, targets)
        mse_list = []
        for p, t in zip(unnormalized_predictions, unnormalized_targets):
            mse_list.append(torch.mean((p - t) ** 2).item())
        mean_mse = float(np.mean(mse_list))
        save_payload['targets'] = unnormalized_targets
        save_payload['mean_mse'] = mean_mse
        print(f"Mean MSE (unnormalized): {mean_mse:.6f}")

    output_path = os.path.join(run_dir, output_name)
    torch.save(save_payload, output_path)
    print(f"Saved predictions to: {output_path}")

    save_prediction_plots(unnormalized_predictions, ground_truth_sequences, run_dir)

def main():
    parser = argparse.ArgumentParser(description="Predict using saved Mamba model(s)")
    parser.add_argument('directory', type=str, help="Path to either a parent experiment directory or a specific configuration run directory")
    parser.add_argument('dataset_pt', type=str, help="Path to a .pt file containing prediction dataset")
    parser.add_argument('ground_truth_pt', type=str, help="Path to a .pt file containing ground truth trajectories")
    parser.add_argument('--output_name', type=str, default='predictions.pt', help="Output .pt filename to save inside each run directory")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_pt_path = os.path.abspath(args.dataset_pt)
    if not os.path.exists(dataset_pt_path):
        raise FileNotFoundError(f"Dataset .pt file was not found: {dataset_pt_path}")

    ground_truth_pt_path = os.path.abspath(args.ground_truth_pt)
    if not os.path.exists(ground_truth_pt_path):
        raise FileNotFoundError(f"Ground truth .pt file was not found: {ground_truth_pt_path}")

    dataset_features, dataset_targets = load_prediction_inputs(dataset_pt_path)
    ground_truth_sequences = load_ground_truth_sequences(ground_truth_pt_path)
    run_dirs = resolve_run_directories(args.directory)

    print(f"Using device: {device}")
    print(f"Dataset source: {dataset_pt_path}")
    print(f"Ground truth source: {ground_truth_pt_path}")
    print(f"Discovered {len(run_dirs)} run directory(ies) to predict.")

    for run_dir in run_dirs:
        try:
            run_prediction_for_directory(
                run_dir=run_dir,
                dataset_features=dataset_features,
                dataset_targets=dataset_targets,
                ground_truth_sequences=ground_truth_sequences,
                device=device,
                output_name=args.output_name,
            )
        except FileNotFoundError as e:
            # Catches missing checkpoints, configs, or scalers and skips the directory
            print(f"\n[!] Warning: Skipping {run_dir}\n    Reason: {e}")
            continue
        except Exception as e:
            # Catches any other unexpected crashes so the whole script doesn't die
            print(f"\n[!] Unexpected Error: Skipping {run_dir}\n    Reason: {e}")
            continue

if __name__ == '__main__':
    main()
