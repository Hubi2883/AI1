#!/usr/bin/env python
# run_inference.py

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models import Autoformer, DLinear, TimeLLM  # Ensure these modules are accessible
from data_provider.data_factory import data_provider  # Ensure this module is accessible
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import pandas as pd


def plot_full_series(timeline, actual_series, predicted_series, feature_name, save_path=None):
    """
    Plots the actual data and predictions over the entire dataset.

    Args:
        timeline (np.ndarray): Time indices for the x-axis.
        actual_series (np.ndarray): Actual data points.
        predicted_series (np.ndarray): Predicted data points.
        feature_name (str): Name of the feature being plotted.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(20, 8))
    
    # Plot Actual Data
    plt.plot(timeline, actual_series, label='Actual OT', color='blue', linewidth=2)
    
    # Plot Predicted Data
    plt.plot(timeline, predicted_series, label='Predicted OT', color='red', linewidth=2, alpha=0.7)
    
    # Highlight Prediction Horizon
    pred_start = len(actual_series) - len(predicted_series)
    plt.axvline(x=pred_start, color='gray', linestyle='--', linewidth=1)
    plt.text(pred_start, plt.ylim()[1], 'Prediction Start', rotation=90, verticalalignment='top', color='gray')
    
    # Titles and Labels
    plt.title(f'Full Series Forecasting Results for Feature: {feature_name}', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    
    # Grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save or Show Plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Forecast plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Time-LLM Inference for Full Series Forecasting')

    # [All your existing arguments]

    # Required arguments
    parser.add_argument('--model', type=str, required=True, choices=['Autoformer', 'DLinear', 'TimeLLM'],
                        help='Model name: Autoformer, DLinear, or TimeLLM')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset (e.g., electricity.csv)')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset name (e.g., ECL, ETTh1, etc.)')

    # Model and data configuration
    parser.add_argument('--features', type=str, required=True, choices=['M', 'S', 'MS'],
                        help='Features to use: M (multivariate), S (univariate), MS (multivariate to univariate)')
    parser.add_argument('--seq_len', type=int, required=True,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, required=True,
                        help='Label sequence length')
    parser.add_argument('--pred_len', type=int, required=True,
                        help='Prediction sequence length')
    parser.add_argument('--enc_in', type=int, required=True,
                        help='Encoder input size (number of input features)')
    parser.add_argument('--dec_in', type=int, required=True,
                        help='Decoder input size (number of decoder input features)')
    parser.add_argument('--c_out', type=int, required=True,
                        help='Output size (set to 1 for target feature)')
    parser.add_argument('--d_model', type=int, required=True,
                        help='Dimension of the model')
    parser.add_argument('--n_heads', type=int, required=True,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, required=True,
                        help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, required=True,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, required=True,
                        help='Dimension of the feedforward network')
    parser.add_argument('--factor', type=int, required=True,
                        help='Attention factor')
    parser.add_argument('--embed', type=str, required=True, choices=['timeF', 'fixed', 'learned'],
                        help='Type of embedding: timeF, fixed, or learned')
    parser.add_argument('--freq', type=str, required=True,
                        help='Frequency for time features encoding, options: [s, t, h, d, b, w, m]')
    parser.add_argument('--target', type=str, required=True,
                        help='Target feature to forecast')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Number of workers for data loading')
    parser.add_argument('--percent', type=float, required=True,
                        help='Percentage of data to use (e.g., 100 for full data)')

    # TimeLLM specific arguments
    parser.add_argument('--llm_dim', type=int, required=True,
                        help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, required=True,
                        help='Number of layers in the LLM')
    parser.add_argument('--llm_model', type=str, required=True, choices=['LLAMA', 'GPT2', 'BERT'],
                        help='LLM model type')
    parser.add_argument('--prompt_domain', type=int, required=True, choices=[0, 1],
                        help='Prompt domain flag (0 or 1)')
    parser.add_argument('--content', type=str, required=True,
                        help='Content for prompt domain')
    parser.add_argument('--patch_len', type=int, required=True,
                        help='Patch length for embedding')
    parser.add_argument('--stride', type=int, required=True,
                        help='Stride for patch embedding')

    # Additional arguments
    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout rate')
    parser.add_argument('--task_name', type=str, required=True, choices=['long_term_forecast', 'short_term_forecast', 'imputation'],
                        help='Task name, options: [long_term_forecast, short_term_forecast, imputation]')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for inference')
    parser.add_argument('--output_attention', action='store_true',
                        help='Whether to output attention weights')
    parser.add_argument('--device', type=str, required=True, choices=['cuda', 'cpu'],
                        help='Device to run inference on: cuda or cpu')
    parser.add_argument('--save_plot_path', type=str, required=True,
                        help='Path to save the inference plot (e.g., forecasting_results.png)')
    parser.add_argument('--seasonal_patterns', type=str, required=True,
                        help='Seasonal patterns for the dataset (e.g., Monthly)')

    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Set device
    device = accelerator.device if accelerator.device.type != 'cpu' else 'cpu'

    # Construct test_args
    test_args = argparse.Namespace(
        task_name=args.task_name,
        is_training=0,
        model_id='ECL_512_96',  # Modify if needed
        model=args.model,
        data=args.data,
        root_path=os.path.dirname(args.data_path) + '/',
        data_path=os.path.basename(args.data_path),
        features=args.features,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        dec_in=args.dec_in,
        c_out=args.c_out,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        embed=args.embed,
        output_attention=args.output_attention,
        batch_size=args.batch_size,
        freq=args.freq,
        target=args.target,
        num_workers=args.num_workers,
        percent=args.percent,
        seasonal_patterns=args.seasonal_patterns,
        llm_dim=args.llm_dim,
        llm_layers=args.llm_layers,
        llm_model=args.llm_model,
        prompt_domain=args.prompt_domain,
        content=args.content,
        patch_len=args.patch_len,
        stride=args.stride,
        dropout=args.dropout,
        use_amp=False,  # Assuming not using automatic mixed precision
        device=device,
    )

    # Load full data
    test_args.flag = 'test'  # Use the test flag
    test_data, test_loader = data_provider(test_args, flag='test')

    # Initialize the model
    if args.model == 'Autoformer':
        model = Autoformer.Model(test_args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(test_args).float()
    elif args.model == 'TimeLLM':
        model = TimeLLM.Model(test_args).float()
    else:
        print(f"Model {args.model} not recognized.")
        sys.exit(1)

    # Load the checkpoint
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # If the checkpoint contains 'model_state_dict', load accordingly
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is the state_dict
            model.load_state_dict(checkpoint)
        print(f"Checkpoint loaded successfully from {args.checkpoint_path}.")
    else:
        print(f"Checkpoint not found at {args.checkpoint_path}. Exiting.")
        sys.exit(1)

    # Move model to device
    model.to(device)
    model.eval()

    # Prepare the test_loader with accelerator
    test_loader = accelerator.prepare(test_loader)

    # Initialize arrays for full series
    total_length = len(test_data.data_y)
    timeline = np.arange(total_length)
    full_actual = np.zeros(total_length)
    full_predicted = np.full(total_length, np.nan)  # Initialize with NaN

    # Manually inverse transform actual data
    mean_target = test_data.scaler.mean_[-1]
    scale_target = test_data.scaler.scale_[-1]
    full_actual = test_data.data_y[:, -1] * scale_target + mean_target

    # Calculate prediction start index
    pred_start_index = args.seq_len + args.label_len

    # Calculate maximum number of predictions
    max_predictions = total_length - pred_start_index

    # Initialize prediction counter
    assigned_predictions = 0

    # Inference loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Running Inference")):
            if assigned_predictions >= max_predictions:
                break  # Stop if all predictions have been assigned

            # Unpack the batch
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Prepare decoder input (use previous labels)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # Forward pass
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Adjust output dimensions based on features
            if args.features in ['M', 'MS']:
                f_dim = -1  # Select the last feature (target feature)
            else:
                f_dim = 0  # For univariate tasks

            # Select relevant features for output
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            # Reshape to 2D for inverse_transform if scaler exists
            if hasattr(test_data, 'scaler'):
                # Check if scaler has mean_ and scale_ attributes
                if hasattr(test_data.scaler, 'mean_') and hasattr(test_data.scaler, 'scale_'):
                    batch_size, pred_len, features = outputs.shape
                    # Reshape outputs and batch_y to (batch_size * pred_len, features)
                    outputs_reshaped = outputs.cpu().reshape(-1, features)
                    batch_y_reshaped = batch_y.cpu().reshape(-1, features)

                    # Manually inverse transform only the target feature
                    # Assuming target feature is the last feature
                    mean_target = test_data.scaler.mean_[-1]
                    scale_target = test_data.scaler.scale_[-1]

                    # Apply inverse transformation
                    outputs_inversed = outputs_reshaped * scale_target + mean_target
                    batch_y_inversed = batch_y_reshaped * scale_target + mean_target
                else:
                    print("Scaler does not have 'mean_' and 'scale_' attributes. Skipping inverse transform.")
                    outputs_inversed = outputs.cpu()
                    batch_y_inversed = batch_y.cpu()
            else:
                outputs_inversed = outputs.cpu()
                batch_y_inversed = batch_y.cpu()

            # Flatten predictions
            batch_predictions = outputs_inversed.numpy().flatten()

            # Calculate the number of predictions to assign in this batch
            remaining_predictions = max_predictions - assigned_predictions
            if remaining_predictions <= 0:
                break  # All predictions have been assigned

            # Determine how many predictions to take from this batch
            assign_len = min(len(batch_predictions), remaining_predictions)

            # Calculate start and end indices
            start_idx = pred_start_index + assigned_predictions
            end_idx = start_idx + assign_len

            # Assign predictions
            full_predicted[start_idx:end_idx] = batch_predictions[:assign_len]

            # Update the counter
            assigned_predictions += assign_len

            # Optional: Print assignment status
            if i % 10 == 0:
                print(f"Assigned {assigned_predictions}/{max_predictions} predictions.")

    # Define save paths
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results_csv', exist_ok=True)
    save_path = os.path.join('plots', f'{os.path.splitext(args.save_plot_path)[0]}_{args.target}.png')
    csv_filename = os.path.join('results_csv', f'inference_results_{args.target}.csv')

    # Plot the full series forecasting results
    plot_full_series(timeline, full_actual, full_predicted, args.target, save_path=save_path)

    # Prepare DataFrame for CSV
    results_df = pd.DataFrame({
        'Time': timeline,
        'Actual': full_actual,
        'Predicted': full_predicted
    })

    # Save results to CSV
    results_df.to_csv(csv_filename, index=False)
    print(f"Inference results for {args.target} saved to '{csv_filename}'")

    # Compute evaluation metrics on the predicted points
    valid_indices = ~np.isnan(full_predicted)
    mae = np.mean(np.abs(full_actual[valid_indices] - full_predicted[valid_indices]))
    rmse = np.sqrt(np.mean((full_actual[valid_indices] - full_predicted[valid_indices]) ** 2))
    mape = np.mean(np.abs((full_actual[valid_indices] - full_predicted[valid_indices]) / (full_actual[valid_indices] + 1e-10))) * 100  # Prevent division by zero

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    print("All inference tasks completed successfully.")


if __name__ == "__main__":
    main()
