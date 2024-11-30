#!/usr/bin/env python
# run_inference.py

import argparse
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models import Autoformer, DLinear, TimeLLM  # Ensure these modules are accessible
from data_provider.data_factory import data_provider  # Ensure this module is accessible
import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # Adjust the value as needed

def plot_classification_results(timeline, true_labels, predicted_labels, save_path=None):
    """
    Plots the true class labels and predicted class labels over time, limited to the first 1,000 data points.

    Args:
        timeline (np.ndarray): Time indices for the x-axis.
        true_labels (np.ndarray): True class labels.
        predicted_labels (np.ndarray): Predicted class labels.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000  # Adjust chunksize to handle large data

    # Limit to the first 1,000 data points
    max_points = 1000
    timeline = timeline[:max_points]
    true_labels = true_labels[:max_points]
    predicted_labels = predicted_labels[:max_points]

    plt.figure(figsize=(15, 6))

    # Plot True Labels
    plt.plot(timeline, true_labels, label='True Class', color='blue', linewidth=2)

    # Plot Predicted Labels
    plt.plot(timeline, predicted_labels, label='Predicted Class', color='red', linewidth=2, alpha=0.7)

    # Titles and Labels
    plt.title('Classification Results Over First 1,000 Data Points', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Class Label', fontsize=14)
    plt.legend(fontsize=12)

    # Grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save or Show Plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Classification plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Time-LLM Inference for Classification Task')

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
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for classification')
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
                        help='Target feature for classification')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Number of workers for data loading')
    parser.add_argument('--percent', type=float, required=True,
                        help='Percentage of data to use (e.g., 100 for full data)')
    parser.add_argument('--csv_name', type=str, required=True,
                        help='Name of the csv file')

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
    parser.add_argument('--task_name', type=str, required=True, choices=['classification', 'anomaly_detection'],
                        help='Task name, options: [classification, anomaly_detection]')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for inference')
    parser.add_argument('--output_attention', action='store_true',
                        help='Whether to output attention weights')
    parser.add_argument('--device', type=str, required=True, choices=['cuda', 'cpu'],
                        help='Device to run inference on: cuda or cpu')
    parser.add_argument('--save_results_path', type=str, required=True,
                        help='Path to save the inference results (e.g., inference_results.csv)')
    parser.add_argument('--save_plot_path', type=str, required=True,
                        help='Path to save the classification plot (e.g., classification_plot.png)')
    parser.add_argument('--seasonal_patterns', type=str, required=True,
                        help='Seasonal patterns for the dataset (e.g., Monthly)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Set device
    device = accelerator.device if accelerator.device.type != 'cpu' else 'cpu'

    # Construct test_args
    test_args = argparse.Namespace(
        task_name=args.task_name,
        is_training=0,
        model_id='inference_model',  # Modify if needed
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
        num_classes=args.num_classes,
        c_out=args.num_classes,  # For classification, output size equals number of classes
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
        csv_name=args.csv_name,
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

    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []

    # Inference loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Running Inference")):
            # Unpack the batch
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark)

            # Reshape outputs and targets to match training
            B, T, num_classes = outputs.shape
            outputs = outputs.reshape(B * T, num_classes)  # Shape: (B*T, num_classes)
            batch_y = batch_y.reshape(B * T)               # Shape: (B*T,)

            # Get predicted classes
            _, predicted = torch.max(outputs, dim=1)  # Shape: (B*T,)

            # Collect predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())

    # Convert to NumPy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Generate timeline
    total_length = len(all_true_labels)
    timeline = np.arange(total_length)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Time': timeline,
        'True_Label': all_true_labels,
        'Predicted_Label': all_predictions
    })
    results_df.to_csv(args.save_results_path, index=False)
    print(f"Inference results saved to '{args.save_results_path}'")

    # Plot the classification results
    plot_classification_results(timeline, all_true_labels, all_predictions, save_path=args.save_plot_path)

    print("All inference tasks completed successfully.")

if __name__ == "__main__":
    main()
