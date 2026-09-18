from typing import Any
from rotary_encodings import RotaryEncodings
import torch.nn as nn
import torch
import math

class KVCacheMemory():
    def __init__(self, num_kv_heads, seq_len, head_dim):
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.curr_pos = 0
        self.init_cache()

    def reset_pos(self):
        self.curr_pos = 0
        self.init_cache()

    def init_cache(self):
        self.k_cached = torch.zeros((1,  self.num_kv_heads, self.seq_len, self.head_dim))
        self.q_cached = torch.zeros((1, self.num_kv_heads, self.seq_len, self.head_dim))

    def update(self, k, q):
        self.k_cached[:, :, self.curr_pos : self.curr_pos + 1, :] = k
        self.q_cached[:, :, self.curr_pos : self.curr_pos + 1, :] = q
        self.curr_pos += 1

    def __call__(self):
        return self.k_cached[:, :, :self.curr_pos, :], self.q_cached[:, :,  :self.curr_pos, :]        

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, num_kv_heads, seq_len, d_k, weights, layer_n, dropout=0.1) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = num_hidden // num_heads

        self.rotary_encodings = RotaryEncodings(seq_len, self.head_dim)

        self.seq_len = seq_len
        self.d_k = d_k
        
        #caching KV for inference time
        self.cache = KVCacheMemory(num_kv_heads, seq_len, self.head_dim)

        self.W_q = nn.Linear(num_hidden, num_heads * self.head_dim)      # 288 -> 6*48 = 288
        self.W_k = nn.Linear(num_hidden, num_kv_heads * self.head_dim)   # 288 -> num_kv_heads*48
        self.W_v = nn.Linear(num_hidden, num_kv_heads * self.head_dim)   # 288 -> num_kv_heads*48
        self.W_o = nn.Linear(num_heads * self.head_dim, num_hidden) 

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mask = self.get_mask(self.seq_len)

        with torch.no_grad():
            self.W_q.weight.copy_(weights[f"model.layers.{layer_n}.self_attn.q_proj.weight"])
            self.W_k.weight.copy_(weights[f"model.layers.{layer_n}.self_attn.k_proj.weight"])
            self.W_v.weight.copy_(weights[f"model.layers.{layer_n}.self_attn.v_proj.weight"])
            self.W_o.weight.copy_(weights[f"model.layers.{layer_n}.self_attn.o_proj.weight"])
    
    def get_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.tril(torch.ones(size, size, device=device), diagonal=0)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, keys, values, mask=False):
        # Reshaping expanded to n_heads or n_kv_heads
        seq_len, num_hidden = query.shape[1], query.shape[2]
        query  = self.W_q(query).view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys   = self.W_k(keys).view(-1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = self.W_v(values).view(-1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)        

        # shape [batch, seq_len, kv_heads, hidden]
        query = self.rotary_encodings(query)
        rope_keys = self.rotary_encodings(keys) 

        # if evaluation, [batch, seq_len = 1, ....], so we need to cache the KV    
        if not self.training:
            # in this case keys and values span the whole sequence but we cache only the last one
            # so we have the keys and values for last token in the sequence 
            self.cache.update(rope_keys, values)

            # then we need to get the whole cached keys and values        
            rope_keys, rope_values = self.cache()


        # bring them to the same shape as original key and values
        rope_keys = rope_keys.repeat_interleave(self.num_rep, dim=1)
        values = values.repeat_interleave(self.num_rep, dim=1)

        # Q * K_T, in this case its [batch, 1, heads, seq_len] * [batch, seq_len, heads, hidden]
        QK_T = torch.matmul(query,  rope_keys.mT)

        # QK_T / sqrt(dk)
        QK_T = QK_T / math.sqrt(self.d_k)

        # mask
        if mask and self.training:
            QK_T = QK_T.masked_fill(mask == 0, float("-inf"))

        # softmax(QK_T / sqrt(d_k)
        attention_scores = self.softmax(QK_T)

        #dropout
        if self.training:
            attention_scores = self.dropout(attention_scores)

        output = torch.matmul(attention_scores, values)  

        # Reshape and apply output linear layer  
        output = output.transpose(1, 2).contiguous().view(-1, seq_len, self.num_heads * self.head_dim)  
        output = self.W_o(output)  
          
        return output