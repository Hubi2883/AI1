#!/usr/bin/env python
# run_inference.py

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models import Autoformer, DLinear, TimeLLM  # Ensure these are accessible
from data_provider.data_factory import data_provider  # Ensure this is accessible
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

def plot_anomalies(time_series, predictions, anomalies, save_path=None):
    """
    Plots the actual data, predictions, and highlights anomalies.

    Args:
        time_series (np.ndarray): Actual data points.
        predictions (np.ndarray): Predicted data points.
        anomalies (np.ndarray): Boolean array indicating anomalies.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(time_series, label='Actual Data')
    plt.plot(predictions, label='Predictions', alpha=0.7)
    # Mark anomalies
    anomaly_indices = np.where(anomalies)[0]
    plt.scatter(anomaly_indices, time_series[anomaly_indices], color='red', label='Anomalies')
    plt.legend()
    plt.title('Anomaly Detection Results')
    plt.xlabel('Time')
    plt.ylabel('Value')
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Time-LLM Inference for Anomaly Detection')

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
    parser.add_argument('--features', type=str, default='M',
                        help='Features to use: M (multivariate), S (univariate), MS (multivariate to univariate)')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Label sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=7,
                        help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=7,
                        help='Output size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Dimension of the model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Dimension of the feedforward network')
    parser.add_argument('--factor', type=int, default=5,
                        help='Attention factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Type of embedding: timeF, fixed, or learned')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features encoding, options: [s, t, h, d, b, w, m]')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target feature in S or MS task')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--percent', type=float, default=100.0,
                        help='Percentage of data to use')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='Seasonal patterns for M4 dataset')

    # TimeLLM specific arguments
    parser.add_argument('--llm_dim', type=int, default=4096,
                        help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=2,
                        help='Number of layers in the LLM')
    parser.add_argument('--llm_model', type=str, default='LLAMA', choices=['LLAMA', 'GPT2', 'BERT'],
                        help='LLM model type')
    parser.add_argument('--prompt_domain', type=int, default=0,
                        help='Prompt domain flag (0 or 1)')
    parser.add_argument('--content', type=str, default='',
                        help='Content for prompt domain')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch length for embedding')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride for patch embedding')

    # Additional arguments
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='Task name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--output_attention', action='store_true',
                        help='Whether to output attention weights')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on: cuda or cpu')
    parser.add_argument('--save_plot_path', type=str, default='anomaly_detection_plots.png',
                        help='Path to save the anomaly detection plot')

    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Set device
    device = accelerator.device if accelerator.device.type != 'cpu' else 'cpu'

    # Construct test_args
    test_args = argparse.Namespace(
        task_name=args.task_name,
        is_training=0,
        model_id='ECL_512_96',
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

    # Load test data
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
        model.load_state_dict(checkpoint)  # Directly load the state_dict
        print(f"Checkpoint loaded successfully from {args.checkpoint_path}.")
    else:
        print(f"Checkpoint not found at {args.checkpoint_path}. Exiting.")
        sys.exit(1)


    # Move model to device
    model.to(device)
    model.eval()

    # Prepare the test_loader with accelerator
    test_loader = accelerator.prepare(test_loader)

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference"):
            # Unpack the batch
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Prepare decoder input (zeros)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # Forward pass
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]

            # Move outputs and targets to CPU and convert to numpy
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truth.append(batch_y[:, -args.pred_len:, f_dim:].cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    # Compute residuals
    residuals = np.abs(all_ground_truth - all_predictions)

    # Determine anomaly threshold (mean + 3*std)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    threshold = mean_residual + 3 * std_residual
    print(f"Anomaly detection threshold set to: {threshold}")

    # Identify anomalies
    anomalies = residuals > threshold  # Boolean array

    # Prepare data for plotting
    feature_index = 0  # Adjust if needed
    actual_series = all_ground_truth[:, :, feature_index].flatten()
    predicted_series = all_predictions[:, :, feature_index].flatten()
    anomalies_series = anomalies[:, :, feature_index].flatten()

    # Plot the results
    plot_anomalies(actual_series, predicted_series, anomalies_series, save_path=args.save_plot_path)

if __name__ == "__main__":
    main()
