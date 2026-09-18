import torch
import torch.nn as nn
import torch.nn.functional as F
from grouped_query_attention import GroupedQueryAttention
from rms_norm import RMSNorm
from modules.quantization_cpu_np_infer import QLinear

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class FeedForward(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden, weights, layer_n, logger, args) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_ffn_hidden = num_ffn_hidden
        
        self.silu = SiLU()
        
        self.gate_proj = QLinear(num_hidden, num_ffn_hidden, logger=logger,
                    wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                    wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                    subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target)
        self.up_proj = QLinear(num_hidden, num_ffn_hidden, logger=logger,
                    wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                    wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                    subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target)
        self.down_proj = QLinear(num_ffn_hidden, num_hidden, logger=logger,
                    wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                    wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                    subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target)

        with torch.no_grad():
            self.gate_proj.weight.copy_(weights[f"model.layers.{layer_n}.mlp.gate_proj.weight"])
            self.up_proj.weight.copy_(weights[f"model.layers.{layer_n}.mlp.up_proj.weight"])
            self.down_proj.weight.copy_(weights[f"model.layers.{layer_n}.mlp.down_proj.weight"])

    def forward(self, x):
        gate = self.silu(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)
        return x

class LlamaModel(nn.Module):
    def __init__(self, num_layers, n_heads, num_kv_heads, seq_len, num_hidden, num_ffn_hidden, weights, logger, args) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.decoders = nn.ModuleList([LlamaLayer(num_hidden, num_ffn_hidden, n_heads, num_kv_heads, seq_len, weights, i, logger, args) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.decoders:
            x = layer(x)
        return x

class LlamaLayer(nn.Module):
    def __init__(self, num_hidden, num_ffn_hidden, num_heads, num_kv_heads, seq_len, weights, layer_n, logger, args) -> None:
        super().__init__()
        self.grouped_query_attention = GroupedQueryAttention(num_hidden=num_hidden, num_heads=num_heads, num_kv_heads=num_kv_heads, seq_len=seq_len, d_k=1, weights=weights, layer_n=layer_n) 
        self.feed_forward = FeedForward(num_hidden=num_hidden, num_ffn_hidden=num_ffn_hidden, weights=weights, layer_n=layer_n, logger=logger, args=args)
    
        self.rms_norm1 = RMSNorm(num_hidden, weights, f"model.layers.{layer_n}.input_layernorm.weight")
        self.rms_norm2 = RMSNorm(num_hidden, weights, f"model.layers.{layer_n}.post_attention_layernorm.weight")
    
    def forward(self, output_with_pos):
        #first norm
        x = self.rms_norm1(output_with_pos)

        #attention
        x = self.grouped_query_attention(x, x, x)

        #add and norm
        x_after_attention = x + output_with_pos
        x = self.rms_norm2(x)

        #SwiGLU
        x = self.feed_forward(x)

        #add
        x = x + x_after_attention
        return x

class Llama2(nn.Module):
    def __init__(self, decoder_layers_num, num_hidden, num_ffn_hidden, num_heads, num_kv_heads, seq_len, vocab_size, weights, logger, args) -> None:
        super().__init__()
        self.model = LlamaModel(decoder_layers_num, num_heads, num_kv_heads, seq_len, num_hidden, num_ffn_hidden, weights, logger, args)
        self.embedding = nn.Embedding(vocab_size, num_hidden) #XXX QLinear, no bias
        self.linear = self.linear = QLinear(num_hidden, vocab_size, logger=logger,
                    wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                    wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                    subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target)
        self.softmax = nn.Softmax(dim=-1)
        self.rms_norm = RMSNorm(num_hidden, weights, "model.norm.weight")

        with torch.no_grad():
            self.linear.weight.copy_(weights["lm_head.weight"])
            self.embedding.weight.copy_(weights["model.embed_tokens.weight"])

    def forward(self, x):
        #embeddings
        x = self.embedding(x)

        #forward pass -> each llama decoder layer
        output = self.model(x)
        
        #rms norm and final linear layer
        output = self.rms_norm(output)
        output = self.linear(output)
        output = self.softmax(output)

        #softmax to return the probability distribution
        return output
