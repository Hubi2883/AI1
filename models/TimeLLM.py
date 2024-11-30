from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
import os

transformers.logging.set_verbosity_error()

class FlattenHead_prediction(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    


class ExtendedFullyConnectedFlattenHead(nn.Module):
    def __init__(self, n_vars, d_ff, num_classes, T, head_dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.T = T
        self.head_dropout = head_dropout

        # Convolutional Layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=n_vars * d_ff, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Dropout for convolutional layers
        self.dropout1 = nn.Dropout(head_dropout)
        self.dropout2 = nn.Dropout(head_dropout)
        self.dropout3 = nn.Dropout(head_dropout)

        # Global pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers (extended)
        self.fc1 = nn.Linear(128 * 2, 512)  # Combine max and avg pooled features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, T * num_classes)

        # Dropout for fully connected layers
        self.fc_dropout1 = nn.Dropout(head_dropout)
        self.fc_dropout2 = nn.Dropout(head_dropout)
        self.fc_dropout3 = nn.Dropout(head_dropout)
        self.fc_dropout4 = nn.Dropout(head_dropout)

    def forward(self, x):
        # Input shape: (B, N, D_ff, L_total)
        B, N, D_ff, L_total = x.shape

        # Flatten over N and D_ff dimensions
        x = x.view(B, N * D_ff, L_total)  # Shape: (B, N*D_ff, L_total)

        # Convolutional layers with ReLU and normalization
        x = F.relu(self.bn1(self.conv1(x)))  # Shape: (B, 128, L_total)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))  # Shape: (B, 256, L_total)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))  # Shape: (B, 128, L_total)
        x = self.dropout3(x)

        # Global pooling
        max_pooled = self.global_max_pool(x).squeeze(-1)  # Shape: (B, 128)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # Shape: (B, 128)

        # Concatenate pooled features
        x = torch.cat([max_pooled, avg_pooled], dim=1)  # Shape: (B, 128*2)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Shape: (B, 512)
        x = self.fc_dropout1(x)
        x = F.relu(self.fc2(x))  # Shape: (B, 256)
        x = F.relu(self.fc3(x))  # Shape: (B, 256)
        x = self.fc_dropout2(x)
        x = F.relu(self.fc4(x))  # Shape: (B, 128)
        x = self.fc_dropout3(x)
        x = F.relu(self.fc5(x))  # Shape: (B, 128)
        x = F.relu(self.fc6(x))  # Shape: (B, 64)
        x = self.fc_dropout4(x)
        x = self.fc7(x)          # Shape: (B, T * num_classes)

        # Reshape to (B, T, num_classes)
        x = x.view(B, self.T, self.num_classes)

        return x

class ComplexFlattenHeadBinaryClassification(nn.Module):
    def __init__(self, n_vars, d_ff, num_classes, T, head_dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.T = T
        self.head_dropout = head_dropout

        # Define convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=n_vars * d_ff, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(head_dropout)
        self.dropout2 = nn.Dropout(head_dropout)
        self.dropout3 = nn.Dropout(head_dropout)

        # Global pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 256)  # Combine max and avg pooled features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, T * num_classes)

    def forward(self, x):
        # Input shape: (B, N, D_ff, L_total)
        B, N, D_ff, L_total = x.shape

        # Flatten over N and D_ff dimensions
        x = x.view(B, N * D_ff, L_total)  # Shape: (B, N*D_ff, L_total)

        # Convolutional layers with ReLU and normalization
        x = F.relu(self.bn1(self.conv1(x)))  # Shape: (B, 128, L_total)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))  # Shape: (B, 256, L_total)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))  # Shape: (B, 128, L_total)
        x = self.dropout3(x)

        # Global pooling
        max_pooled = self.global_max_pool(x).squeeze(-1)  # Shape: (B, 128)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # Shape: (B, 128)

        # Concatenate pooled features
        x = torch.cat([max_pooled, avg_pooled], dim=1)  # Shape: (B, 128*2)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Shape: (B, 256)
        x = F.relu(self.fc2(x))  # Shape: (B, 128)
        x = self.fc3(x)          # Shape: (B, T * num_classes)

        # Reshape to (B, T, num_classes)
        x = x.view(B, self.T, self.num_classes)

        return x

class UltraComplexFlattenHeadBinaryClassification(nn.Module):
    def __init__(self, n_vars, d_ff, num_classes, T, head_dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.T = T
        self.head_dropout = head_dropout

        # Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=n_vars * d_ff, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        # Depthwise-Separable Convolution Block
        self.depthwise_conv = nn.Conv1d(128, 128, kernel_size=3, groups=128, padding=1)
        self.pointwise_conv = nn.Conv1d(128, 128, kernel_size=1)

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Pooling Layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Residual Pathway (Dense Connection)
        self.residual_fc1 = nn.Linear(n_vars * d_ff, 128)
        self.residual_fc2 = nn.Linear(128, 128)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3, 512)  # Combine max, avg pooled, and attention output
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, T * num_classes)

        # Batch Normalization and Dropout
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # Input shape: (B, N, D_ff, L_total)
        B, N, D_ff, L_total = x.shape

        # Flatten over N and D_ff dimensions
        x = x.view(B, N * D_ff, L_total)  # Shape: (B, N*D_ff, L_total)

        # Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))  # Shape: (B, 128, L_total)
        x = F.relu(self.bn2(self.conv2(x)))  # Shape: (B, 256, L_total)
        x = F.relu(self.bn3(self.conv3(x)))  # Shape: (B, 128, L_total)

        # Depthwise-Separable Convolution
        x = F.relu(self.depthwise_conv(x))  # Shape: (B, 128, L_total)
        x = F.relu(self.pointwise_conv(x))  # Shape: (B, 128, L_total)

        # Residual Pathway
        residual = x.mean(dim=-1)  # Shape: (B, 128)
        residual = F.relu(self.residual_fc1(residual))
        residual = F.relu(self.residual_fc2(residual))

        # Attention Mechanism
        attention_input = x.permute(0, 2, 1)  # Shape: (B, L_total, 128)
        attention_out, _ = self.attention(attention_input, attention_input, attention_input)
        attention_out = attention_out.mean(dim=1)  # Shape: (B, 128)

        # Pooling
        max_pooled = self.global_max_pool(x).squeeze(-1)  # Shape: (B, 128)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # Shape: (B, 128)

        # Concatenate Features
        x = torch.cat([max_pooled, avg_pooled, attention_out + residual], dim=1)  # Shape: (B, 128*3)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))  # Shape: (B, 512)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # Shape: (B, 256)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # Shape: (B, 128)
        x = self.dropout(x)
        x = self.fc4(x)          # Shape: (B, T * num_classes)

        # Reshape to (B, T, num_classes)
        x = x.view(B, self.T, self.num_classes)

        return x

class FlattenHead_binary_classification(nn.Module):
    def __init__(self, n_vars, d_ff, num_classes, T, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars        # Number of variables (features), N
        self.d_ff = d_ff            # Feedforward dimension, D_ff
        self.num_classes = num_classes
        self.T = T                  # Original sequence length (specified separately)
        self.head_dropout = head_dropout

        # Hidden layer size (adjust as needed)
        hidden_size = 32

        # Layers
        # Since L_total can vary and we don't include its relationship with T,
        # we'll use global average pooling over L_total dimension.
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc_one = nn.Linear(n_vars * d_ff, hidden_size)
        self.dropout = nn.Dropout(self.head_dropout)
        self.fc_two = nn.Linear(hidden_size, self.T * self.num_classes)

    def forward(self, x):
        # x shape: (B, N, D_ff, L_total)
        B, N, D_ff, L_total = x.shape

        # Flatten over N and D_ff dimensions
        x = x.view(B, N * D_ff, L_total)  # Shape: (B, N*D_ff, L_total)

        # Global average pooling over L_total dimension
        x = self.global_avg_pool(x)  # Shape: (B, N*D_ff, 1)
        x = x.squeeze(-1)            # Shape: (B, N*D_ff) (3, 32)

        # Pass through fully connected layers
        x = F.relu(self.fc_one(x))   # Shape: (B, hidden_size)
        x = self.dropout(x)
        x = self.fc_two(x)           # Shape: (B, T * num_classes)
        x = x.view(B, self.T, self.num_classes)  # Shape: (B, T, num_classes)
        #x = torch.sigmoid(x)

        return x

class CombinedPoolingHeadBinaryClassification(nn.Module):
    def __init__(self, n_vars, d_ff, num_classes, T, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.T = T

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(n_vars * d_ff * 2, T * num_classes)

    def forward(self, x):
        B, N, D_ff, L_total = x.shape
        x = x.view(B, N * D_ff, L_total)  # Shape: (B, N*D_ff, L_total)

        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # Shape: (B, N*D_ff)
        max_pooled = self.global_max_pool(x).squeeze(-1)  # Shape: (B, N*D_ff)

        x = torch.cat([avg_pooled, max_pooled], dim=1)    # Shape: (B, N*D_ff*2)
        x = self.dropout(x)
        x = self.fc(x)                                    # Shape: (B, T * num_classes)
        x = x.view(B, self.T, self.num_classes)

        return x

    
class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.num_classes = configs.num_classes

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead_prediction(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        elif self.task_name == 'classification' or 'anomaly_detection':
            #self.output_projection = FlattenHead_binary_classification(configs.enc_in, self.d_ff, self.num_classes, 
            #                                                           self.seq_len, head_dropout=configs.dropout)
            self.output_projection = ExtendedFullyConnectedFlattenHead(configs.enc_in, self.d_ff, self.num_classes,
                                                                                 self.seq_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification' or self.task_name == 'anomaly_detection':
            dec_out = self.classify(x_enc, x_mark_enc)
            return dec_out
        return None
    
    def classify(self, x_enc, x_mark_enc):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) #Shape: ((B*N), T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0]) #Shape: (B*N, 1)
            max_values_str = str(max_values[b].tolist()[0]) #Shape: (B*N, 1)
            median_values_str = str(medians[b].tolist()[0]) #Shape: (B*N, 1)
            lags_values_str = str(lags[b].tolist()) #Shape: (B*N, 5)
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: classify the {T} length of input data into one of {str(self.num_classes)} categories: 0 for no anomaly and 1 for anomaly;"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous() #Shape: (B, T, N)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids #Shape: (B*N, L_prompt), L_prompt: length of the tokenized prompt after padding
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device)) #Shape: (B × N, L_prompt, D_emb), D_emb: Embedding dimension of the LLM

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) #Shape: (num_tokens, D_emb)

        x_enc = x_enc.permute(0, 2, 1).contiguous() #Shape: (B, N, T)
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16)) # [enc_out Shape: (B*N, L_patches, D_emb)], [n_vars Shape: N], [L_patches = int((T - self.patch_len) / self.stride) + 1]
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) #Shape: (B × N, L_patches, D_emb)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1) #Shape: (B × N, L_total, D_emb), L_total: L_prompt + L_patches
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state # Shape: (B*N, L_total, D_llm)

        dec_out = dec_out[:, :, :self.d_ff] # Shape: (B*N, L_total, D_ff)

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])) #Shape: (B, N, L_total, D_ff)
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() #Shape: (B, N, D_ff, L_total)
        
        dec_out = self.output_projection(dec_out)

        #dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        #dec_out = dec_out.permute(0, 2, 1).contiguous()

        return dec_out

    def forecast(self, x_enc, x_mark_enc):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state

        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() #Shape: (B, N, D_ff, L_total)

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        #dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags



class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
