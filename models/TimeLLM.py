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

#class FlattenHead_binary_classification(nn.Module):
#    def __init__(self, nf, num_classes, head_dropout=0):
        #Number of hidden layers: 1
        #Number of neurons per hidden layer = 4
#        super().__init__()
#        hidden_layer_size = 4
#        self.flatten = nn.Flatten(start_dim=-2)
#        self.fc_one = nn.Linear(nf, hidden_layer_size)
#        self.dropout = nn.Dropout(head_dropout)
#        self.fc_two = nn.Linear(hidden_layer_size, num_classes)

#    def forward(self, x):
#        x = self.flatten(x)
#        x = F.relu(self.fc_one(x))
#        x = self.dropout(x)
#        x = F.sigmoid(self.fc_two(x))
#        return x

class FlattenHead_binary_classification(nn.Module):
    def __init__(self, nf, num_classes, head_dropout=0):
        super().__init__()
        self.num_classes = num_classes
        hidden_layer_size = 4
        self.fc_one = nn.Linear(nf, hidden_layer_size)
        self.dropout = nn.Dropout(head_dropout)
        self.fc_two = nn.Linear(hidden_layer_size, self.num_classes)

    def forward(self, x):
        # x shape: (B, n_vars, d_ff, T)
        x = x.permute(0, 3, 1, 2)  # Shape: (B, T, n_vars, d_ff)
        B, T, n_vars, d_ff = x.shape
        x = x.reshape(B * T, n_vars * d_ff)  # Flatten n_vars and d_ff
        x = F.relu(self.fc_one(x))  # Shape: (B * T, hidden_layer_size)
        x = self.dropout(x)
        x = self.fc_two(x)  # Shape: (B * T, num_classes)
        x = torch.sigmoid(x)
        x = x.view(B, T, self.num_classes)  # Reshape back to (B, T, num_classes)
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
            self.output_projection = FlattenHead_binary_classification(self.head_nf, self.num_classes,
                                                 head_dropout=configs.dropout)
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
                f"Task description: classify the input data into one of {str(self.num_classes)} categories;"
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
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        #dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out
    
    def get_long_term_forecast(self, x_enc, x_mark_enc=None):
        """
        Generates long-term forecast for the given input data.

        Parameters:
        - x_enc (numpy array or torch.Tensor): Input time series data of shape (B, T, N),
          where B is batch size, T is the sequence length, and N is the number of variables/features.
        - x_mark_enc (numpy array or torch.Tensor, optional): Time features corresponding to x_enc,
          of shape (B, T, D), where D is the number of time features.

        Returns:
        - forecast (numpy array): The forecasted data of shape (B, pred_len, N).
        """
        # Ensure x_enc is a torch tensor
        if not isinstance(x_enc, torch.Tensor):
            x_enc = torch.tensor(x_enc, dtype=torch.float32)

        # Add batch dimension if necessary
        if x_enc.dim() == 2:
            x_enc = x_enc.unsqueeze(0)  # Shape: (1, T, N)

        # Process x_mark_enc similarly
        if x_mark_enc is not None:
            if not isinstance(x_mark_enc, torch.Tensor):
                x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32)
            if x_mark_enc.dim() == 2:
                x_mark_enc = x_mark_enc.unsqueeze(0)  # Shape: (1, T, D)
        else:
            x_mark_enc = None  # Adjust based on your model's requirements

        # Create placeholders for decoder inputs (not used in forecast)
        x_dec = torch.zeros((x_enc.shape[0], self.pred_len, x_enc.shape[2]), device=x_enc.device)
        x_mark_dec = torch.zeros_like(x_dec) if x_mark_enc is not None else None

        # Move data to the same device as the model
        device = next(self.parameters()).device
        x_enc = x_enc.to(device)
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.to(device)
            x_mark_dec = x_mark_dec.to(device)
        else:
            x_mark_dec = None
        x_dec = x_dec.to(device)

        # Generate forecast
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            forecast = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Move forecast to CPU and convert to numpy array
        forecast = forecast.cpu().detach().numpy()

        return forecast


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