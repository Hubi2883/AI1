# utils/tools.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil
import logging
from tqdm import tqdm

# Configure matplotlib backend
plt.switch_backend('agg')

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    """
    Adjusts the learning rate based on the specified schedule.

    Args:
        accelerator: Accelerator instance from the `accelerate` library.
        optimizer: Optimizer whose learning rate needs to be adjusted.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
        args: Parsed command-line arguments containing learning rate adjustment type and initial learning rate.
        printout: Flag to control logging of learning rate updates.
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    else:
        logging.warning(f"Unknown learning rate adjustment type: {args.lradj}")
        return

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'Updating learning rate to {lr}')
            else:
                print(f'Updating learning rate to {lr}')


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        """
        Args:
            accelerator: Accelerator instance from the `accelerate` library.
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_mode (bool): If True, saves the model checkpoint when validation loss improves.
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        """
        Call method to evaluate whether to perform early stopping.

        Args:
            val_loss (float): Current validation loss.
            model: Model being trained.
            path (str): Path to save the model checkpoint.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
    
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), f"{path}/checkpoint.pth")
        else:
            torch.save(model.state_dict(), f"{path}/checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    Enables dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    """
    Standardizes data by removing the mean and scaling to unit variance.
    """
    def __init__(self, mean, std):
        """
        Args:
            mean (numpy.ndarray): Mean of the data.
            std (numpy.ndarray): Standard deviation of the data.
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Transforms the data by standardizing.

        Args:
            data (numpy.ndarray or torch.Tensor): Data to be transformed.

        Returns:
            Transformed data.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inversely transforms the data back to original scale.

        Args:
            data (numpy.ndarray or torch.Tensor): Data to be inversely transformed.

        Returns:
            Inversely transformed data.
        """
        return (data * self.std) + self.mean


def adjustment(gt, pred):
    """
    Adjusts predictions based on ground truth anomalies.

    Args:
        gt (list or numpy.ndarray): Ground truth labels.
        pred (list or numpy.ndarray): Predicted labels.

    Returns:
        Tuple of adjusted ground truth and predictions.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    Calculates the accuracy of predictions.

    Args:
        y_pred (list or numpy.ndarray): Predicted labels.
        y_true (list or numpy.ndarray): Ground truth labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    """
    Deletes an entire directory tree.

    Args:
        dir_path (str): Path to the directory to be deleted.
    """
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    """
    Validation function to evaluate the model's performance on the validation dataset.

    Args:
        args: Parsed command-line arguments.
        accelerator: Accelerator instance from the `accelerate` library.
        model: The trained model to be evaluated.
        vali_data: The validation dataset.
        vali_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        mae_metric: Metric for Mean Absolute Error.

    Returns:
        Tuple containing average validation loss and average validation MAE loss.
    """
    total_loss = []
    total_mae_loss = []
    model.eval()
    logging.info("Starting validation...")

    try:
        with torch.no_grad():
            # Ensure that vali_loader has data
            if len(vali_loader) == 0:
                logging.warning("Validation loader is empty.")
                return 0.0, 0.0

            # Iterate over the validation data
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc="Validation Batches")):
                try:
                    # Move data to the appropriate device
                    batch_x = batch_x.float().to(accelerator.device)
                    batch_y = batch_y.float().to(accelerator.device)
                    batch_x_mark = batch_x_mark.float().to(accelerator.device)
                    batch_y_mark = batch_y_mark.float().to(accelerator.device)

                    # Prepare decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

                    # Forward pass with or without AMP
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            if args.output_attention:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # Gather outputs and targets from all processes
                    outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

                    # Select feature dimension
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                    # Detach tensors to prevent gradient computation
                    pred = outputs.detach()
                    true = batch_y.detach()

                    # Compute losses
                    loss = criterion(pred, true)
                    mae_loss = mae_metric(pred, true)

                    # Append losses to the lists
                    total_loss.append(loss.item())
                    total_mae_loss.append(mae_loss.item())

                    # Debugging: Log loss values for the current batch
                    logging.debug(f"Batch {i+1}/{len(vali_loader)} - Loss: {loss.item():.6f}, MAE Loss: {mae_loss.item():.6f}")

                except Exception as batch_e:
                    logging.error(f"Error processing batch {i+1}: {batch_e}")
                    # Depending on your preference, you can choose to continue or halt validation
                    # Here, we'll halt validation if a batch fails
                    return None, None

        # Calculate average losses
        if total_loss:
            avg_loss = np.average(total_loss)
            avg_mae_loss = np.average(total_mae_loss)
            logging.info(f"Validation completed. Average Loss: {avg_loss:.6f}, Average MAE Loss: {avg_mae_loss:.6f}")
        else:
            logging.warning("No losses recorded during validation.")
            avg_loss, avg_mae_loss = 0.0, 0.0

    except Exception as e:
        logging.error(f"An error occurred during validation: {e}")
        return None, None
    finally:
        # Ensure the model is set back to training mode
        model.train()

    return avg_loss, avg_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    """
    Test function to evaluate the model's performance on the test dataset.

    Args:
        args: Parsed command-line arguments.
        accelerator: Accelerator instance from the `accelerate` library.
        model: The trained model to be evaluated.
        train_loader: DataLoader for the training dataset.
        vali_loader: DataLoader for the validation/test dataset.
        criterion: Loss function.

    Returns:
        float: Test loss.
    """
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    test_loss = 0.0
    try:
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
            dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
            outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
            id_list = np.arange(0, B, args.eval_batch_size)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = model(
                    x[id_list[i]:id_list[i + 1]],
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                )
            accelerator.wait_for_everyone()
            outputs = accelerator.gather_for_metrics(outputs)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y)).to(accelerator.device)
            batch_y_mark = torch.ones(true.shape).to(accelerator.device)
            true = accelerator.gather_for_metrics(true)
            batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

            loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)
            test_loss = loss.item()
            logging.info(f"Test Loss: {test_loss:.6f}")

    except Exception as e:
        logging.error(f"An error occurred during testing: {e}")
        return None

    finally:
        model.train()

    return test_loss


def load_content(args):
    """
    Loads content from a text file based on the dataset name.

    Args:
        args: Parsed command-line arguments containing the dataset name.

    Returns:
        str: Content of the text file.
    """
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    try:
        with open(f'./dataset/prompt_bank/{file}.txt', 'r') as f:
            content = f.read()
        logging.info(f"Loaded content from ./dataset/prompt_bank/{file}.txt")
    except FileNotFoundError:
        logging.error(f"File ./dataset/prompt_bank/{file}.txt not found.")
        content = ""
    except Exception as e:
        logging.error(f"An error occurred while loading content: {e}")
        content = ""
    return content
