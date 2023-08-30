from torch import nn
import math
import time
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np

class InferenceModule(torch.nn.Module):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()

class InferenceModuleList(torch.nn.ModuleList):
    def inference(self):
        for mod in self.modules():
            if mod != self:
                mod.inference()

class PositionalEncoding(InferenceModule):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if self.training:
            return super(LayerNorm, self).forward(x)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight.data, self.bias.data, self.eps)
    def inference(self):
        self.training = False

class Value(InferenceModule):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = Linear(dim_input, dim_val, bias=False)
    def forward(self, x):
        return self.fc1(x)

class Key(InferenceModule):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = Linear(dim_input, dim_attn, bias=False)
    def forward(self, x):
        return self.fc1(x)

class Query(InferenceModule):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = Linear(dim_input, dim_attn, bias=False)
    def forward(self, x):
        return self.fc1(x)

class QuerySelector(nn.Module):
    def __init__(self, fraction=0.33):
        super(QuerySelector, self).__init__()
        self.fraction = fraction
    def forward(self, queries, keys, values):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape
        l_Q = int((1.0 - self.fraction) * L_Q)
        K_reduce = torch.mean(keys.topk(l_Q, dim=1).values, dim=1).unsqueeze(1)
        sqk = torch.matmul(K_reduce, queries.transpose(1, 2))
        indices = sqk.topk(l_Q, dim=-1).indices.squeeze(1)
        Q_sample = queries[torch.arange(B)[:, None], indices, :]
        Q_K = torch.matmul(Q_sample, keys.transpose(-2, -1))
        attn = torch.softmax(Q_K / math.sqrt(D), dim=-1)
        mean_values = values.mean(dim=-2)
        result = mean_values.unsqueeze(-2).expand(B, L_Q, mean_values.shape[-1]).clone()
        result[torch.arange(B)[:, None], indices, :] = torch.matmul(attn, values).type_as(result)
        return result, None
    def inference(self):
        pass  # no parameters

# Add Adaptive Winow-aware Mask to self-attention
def deformableAttn(attn, win_mask, dx):
    dx = dx.expand_as(attn)
    return attn.masked_fill(torch.gt(win_mask, dx), -1e18) + attn

# Adaptive Window-aware Offset
class learnedvector(nn.Module):
    def __init__(self, embed_dim, offset_embed1):
        super().__init__()
        self.offset_predictor = nn.Linear(embed_dim, offset_embed1, bias=False)
        self._linear2 = nn.Linear(offset_embed1, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        L = x.shape[1]
        wh_bias = torch.tensor(5. / 3.).sqrt().log()
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(1) + wh_bias)
        pred_offset = self.offset_predictor(self.act(x))
        self.pred_offset = torch.sigmoid(self._linear2(pred_offset)) * L
        return self.pred_offset

class MultiHeadAttention(nn.Module):
    def __init__(self,window_size,learneddim, hidden_size):
        """Initialize the Multi Head Block."""
        super().__init__()
        self.learnedvector = learnedvector(hidden_size, learneddim)
        self._scores = None
        self.window_size = window_size
        self.window_size = list(self.window_size)
        qkv_bias = True
        self.qkv = nn.Linear(hidden_size, hidden_size* 3, bias=qkv_bias)

    def get_mask(self,in_seq_length):
        win_mask = torch.zeros([in_seq_length, in_seq_length])#.cuda()
        for i in range(len(win_mask)):
            for j in range(len(win_mask)):
                win_mask[i, j] = abs(i - j)
                if j < i:
                    win_mask[i, j] = 1e18
        return win_mask

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                ) -> torch.Tensor:
        K = query.shape[1]
        qkv = self.qkv(query).reshape(query.shape[0], query.shape[1], 3, query.shape[2]).permute(2, 0, 1, 3)
        qkv_groups = qkv.chunk(len(self.window_size), -1)
        x_groups = []
        win_mask = self.get_mask(query.shape[1])
        for i, qkv_group in enumerate(qkv_groups):
            dx = self.learnedvector(query)
            window_s = self.window_size[i]
            dx += window_s
            [q, k, v] = [x for x in qkv_group]
            self._scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(K)
            self._scores = deformableAttn(self._scores,win_mask, dx)#.cuda()
            self._scores = F.softmax(self._scores, dim=-1)
            attention = torch.bmm(self._scores, v)
            x_groups.append(attention)
        x = torch.cat(x_groups, -1)
        return x
    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores

class AttentionBlock(InferenceModule):
    def __init__(self, dim_val, dim_attn, window_size,learneddim, hidden_size, debug=False, attn_type='full'):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
        MHA = MultiHeadAttention(window_size,learneddim,hidden_size)
        self._selfAttention = MHA
        self.debug = debug
        self.qk_record = None
        self.qkv_record = None
        self.n = 0
        if attn_type == "full":
            self.attentionLayer = None
        elif attn_type.startswith("query_selector"):
            args = {}
            if len(attn_type.split('_')) == 3:
                args['fraction'] = float(attn_type.split('_')[-1])
            self.attentionLayer = QuerySelector(**args)
        else:
            raise Exception

    def forward(self, x, kv=None):
        if kv is None:
            if self.attentionLayer:
                qkv = self.attentionLayer(self.query(x), self.key(x), self.value(x))[0]
            else:
                x = self._selfAttention(self.query(x), self.key(x), self.value(x))
        return x

class MultiHeadAttentionBlock(InferenceModule):
    def __init__(self, dim_val, dim_attn, window_size,learneddim, hidden_size,embedding_size, attn_type):
        super(MultiHeadAttentionBlock, self).__init__()
        self.h = AttentionBlock(dim_val, dim_attn,window_size,learneddim, hidden_size,attn_type=attn_type)
        self.fc = Linear(hidden_size, embedding_size , bias=False)

    def forward(self, x, kv=None):
        a = self.h(x, kv=kv)
        a = self.fc(a)
        return a

class EncoderLayer(InferenceModule):
    def __init__(self, dim_val, dim_attn, window_size,learneddim, hidden_size, embedding_size, attn_type='full'):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, window_size,learneddim,hidden_size,embedding_size,attn_type='full')
        self.fc1 = Linear(dim_val, dim_val)
        self.fc2 = Linear(dim_val, dim_val)
        self.norm1 = LayerNorm(dim_val)
        self.norm2 = LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)
        a = self.fc1(F.relu(self.fc2(x)))
        x = self.norm2(x + a)
        return x

    def record(self):
        self.attn.record()

class Dropout(nn.Dropout):

    def forward(self, x=False):
        if self.training:
            return super(Dropout, self).forward(x)
        else:
            return x

    def inference(self):
        self.training = False

class Linear(nn.Linear):
    def forward(self, x=False):
        if self.training:
            return super(Linear, self).forward(x)
        else:
            return F.linear(x, self.weight.data, self.bias.data if self.bias is not None else None)
    def inference(self):
        self.training = False

class Transformer(InferenceModule):
    def __init__(self, dim_val, dim_attn, input_dim, out_seq_len, n_encoder_layers, window_size,learneddim, hidden_size, embedding_size,
                 enc_attn_type='full', dec_attn_type='full', dropout=0.1,  debug=False, output_len=1):
        super(Transformer, self).__init__()
        self.output_len = output_len
        self._linear = nn.Linear(embedding_size, 1)
        # Initiate encoder and Decoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, window_size,learneddim, hidden_size, embedding_size,attn_type='full'))
        self.encs = InferenceModuleList(self.encs)
        self.pos = PositionalEncoding(dim_val)
        self.enc_dropout = Dropout(dropout)
        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = Linear(input_dim, dim_val)
        # print("dim_val")
        # print(dim_val)
        self.dec_input_fc = Linear(input_dim, dim_val)
        # self.out_fc = Linear(dec_seq_len * dim_val, out_seq_len * output_len)
        self.debug = debug

    def forward(self, x):
        # encoder
        x = x.to(torch.float32)
        a = self.enc_input_fc(x)
        b = self.enc_dropout(a)
        c = self.pos(b)
        e = self.encs[0](c)
        for enc in self.encs[1:]:
            e = enc(e)
        if self.debug:
            print('Encoder output size: {}'.format(e.shape))
        return e

    def record(self):
        self.debug = True
        for enc in self.encs:
            enc.record()
        for dec in self.decs:
            dec.record()

# 一个hidden module 一次预测多个值
class TVarchitecture(nn.Module):
    def __init__(self, input_dim, output_dim, in_seq_length, out_seq_length, n_encoder_layers, window_size,learneddim, hidden_size, embedding_size,device):
        super(TVarchitecture, self).__init__()
        self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device
        self.pred_len = 6
        self.window_size = window_size
        self.learneddim = learneddim
        self.hidden_size =hidden_size
        self.embedding_size= embedding_size
        # self_d_model = self.hidden_dim
        hidden_layer2 = [Transformer(embedding_size, hidden_size, input_dim,
                                     out_seq_length,n_encoder_layers, window_size,learneddim, hidden_size,embedding_size,
                                     enc_attn_type='full',
                                     dec_attn_type='full', dropout=0.1)]
        for i in range(out_seq_length - 1):  # 12
            hidden_layer2.append(Transformer(embedding_size, hidden_size, input_dim,
                                             out_seq_length,
                                             n_encoder_layers, window_size,learneddim, hidden_size,embedding_size,
                                             enc_attn_type='full',
                                             dec_attn_type='full', dropout=0.1))

        self.hidden_layer2 = nn.ModuleList(hidden_layer2)
        self.output_layer = nn.ModuleList([nn.Linear(in_seq_length, 1) for i in range(out_seq_length)])
        output_layer = [nn.Linear(in_seq_length, self.pred_len)]
        for i in range(out_seq_length - 1):
            output_layer.append(nn.Linear(self.in_seq_length + self.pred_len + self.pred_len, self.pred_len))
        self.output_layer = nn.ModuleList(output_layer)
        self.output_layer1 = nn.ModuleList([nn.Linear(in_seq_length, 1) for i in range(out_seq_length)])
        output_layer1 = [nn.Linear(embedding_size, 7)]
        for i in range(out_seq_length - 1):
            output_layer1.append(nn.Linear(embedding_size, 7))
        self.output_layer1 = nn.ModuleList(output_layer1)

    def forward(self, input, input_emb, target, target_emb, is_training=False):
        target_emb = target_emb.transpose(0, 1)
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(
            self.device)
        next_cell_input = input
        next_cell_input = next_cell_input#.cuda()
        # next_cell_input = next_cell_input
        for i in range(4):
            hidden = self.hidden_layer2[i](next_cell_input)
            hidden = hidden.transpose(1, 2)
            hidden = self.output_layer[i](hidden)
            hidden = hidden.transpose(1, 2)
            hidden_self_attention = self.output_layer1[i](hidden)
            if i==0:
                outputs[i*self.pred_len :(i+1)*self.pred_len , :, :] = hidden_self_attention.transpose(0,1)
            else:
                outputs[i*self.pred_len :(i+1)*self.pred_len , :, :] = hidden_self_attention.transpose(0,1)
            hidden = hidden#.cuda()
            input = input#.cuda()
            target = target#.cuda()
            hidden_self_attention = hidden_self_attention#.cuda()
            hidden = hidden.to(torch.float32)
            input = input.to(torch.float32)
            target = target.to(torch.float32)
            hidden_self_attention = hidden_self_attention.to(torch.float32)
            if is_training:
                if i == 0:
                    next_cell_input = torch.cat((input, hidden_self_attention, target[i * self.pred_len :(i + 1) * self.pred_len , :, :].transpose(0, 1)), dim=1)
                else:
                    next_cell_input = torch.cat((input, hidden_self_attention, target[i * self.pred_len :(i + 1) * self.pred_len , :, :].transpose(0, 1)), dim=1)
            else:
                if i == 0:
                    next_cell_input = torch.cat(
                        (input, hidden_self_attention, target[i * self.pred_len :(i + 1) * self.pred_len , :, :].transpose(0, 1)),
                        dim=1)
                else:
                    next_cell_input = torch.cat(
                        (input, hidden_self_attention, target[i * self.pred_len :(i + 1) * self.pred_len , :, :].transpose(0, 1)),
                        dim=1)
        return outputs


# 一个hidden module 一次预测一个值
# class TVarchitecture(nn.Module):
#     def __init__(self, input_dim, output_dim, in_seq_length, out_seq_length, device):
#         super(TVarchitecture, self).__init__()
#         self.input_dim = input_dim
#         # self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.in_seq_length = in_seq_length
#         self.out_seq_length = out_seq_length
#         self.device = device
#         # self_d_model = self.hidden_dim
#         hidden_layer2 = [Transformer(embedding_size, hidden_size, input_size, pred_len,
#                                      output_len=output_len,
#                                      n_heads=n_heads, n_encoder_layers=n_encoder_layers,
#                                      n_decoder_layers=n_decoder_layers, enc_attn_type=encoder_attention,
#                                      dec_attn_type=decoder_attention, dropout=dropout)]
#         for i in range(out_seq_length - 1):  # 12
#             hidden_layer2.append(Transformer(embedding_size, hidden_size, input_size,  pred_len,
#                                              output_len=output_len,
#                                              n_heads=n_heads, n_encoder_layers=n_encoder_layers,
#                                              n_decoder_layers=n_decoder_layers, enc_attn_type=encoder_attention,
#                                              dec_attn_type=decoder_attention, dropout=dropout))
#         self.hidden_layer2 = nn.ModuleList(hidden_layer2)
#         self.output_layer = nn.ModuleList([nn.Linear(in_seq_length, 1) for i in range(out_seq_length)])
#         output_layer = [nn.Linear(seq_len, pred_len)]
#         for i in range(out_seq_length - 1):  # 12
#             output_layer.append(nn.Linear(seq_len + pred_len + 1, pred_len))
#         self.output_layer = nn.ModuleList(output_layer)
#
#         self.output_layer1 = nn.ModuleList([nn.Linear(in_seq_length, 1) for i in range(out_seq_length)])
#         output_layer1 = [nn.Linear(embedding_size, 7)]
#         for i in range(out_seq_length - 1):  # 12
#             output_layer1.append(nn.Linear(embedding_size, 7))
#         self.output_layer1 = nn.ModuleList(output_layer1)
#
#         self.output_layer3 = nn.ModuleList([nn.Linear(in_seq_length, 1) for i in range(out_seq_length)])
#         output_layer3 = [nn.Linear(pred_len, 1)]
#         for i in range(out_seq_length - 1):  # 12
#             output_layer3.append(nn.Linear(pred_len, 1))
#         self.output_layer3 = nn.ModuleList(output_layer3)
#
#     def forward(self, input, input_emb, target, target_emb, is_training=False):
#         target_emb = target_emb.transpose(0, 1)  # ([24, 32, 4])
#         outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(
#             self.device)  # 12 16 1    torch.Size([12, 133, 1])
#         next_cell_input = input
#         next_cell_input = next_cell_input#.cuda()
#         for i in range(pred_len):
#             hidden = self.hidden_layer2[i](next_cell_input)
#             hidden = hidden.transpose(1, 2)
#             hidden = self.output_layer[i](hidden)  # （16，32，723）
#             hidden = hidden.transpose(1, 2)
#             hidden_self_attention = self.output_layer1[i](hidden)
#             output = hidden_self_attention.transpose(1, 2)
#             output = self.output_layer3[i](output)
#             outputs[i, :, :] = output.transpose(1, 2).squeeze(1)
#             hidden = hidden#.cuda()
#             input = input#.cuda()
#             target = target#.cuda()
#             hidden = hidden.to(torch.float32)
#             input = input.to(torch.float32)
#             target = target.to(torch.float32)
#             if is_training:
#                 next_cell_input = torch.cat((input, hidden_self_attention, target[i, :, :].unsqueeze(1)), dim=1)
#             else:
#                 next_cell_input = torch.cat((input, hidden_self_attention, outputs[i, :, :].unsqueeze(1)), dim=1)
#         return outputs
