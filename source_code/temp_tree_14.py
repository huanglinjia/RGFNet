import random
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
from transformers import BertModel, BertConfig, AutoModel

USE_CUDA = torch.cuda.is_available()
MAX_OUTPUT_LENGTH = 46


class TreeNode:
    def __init__(self, r_node_embedding):
        self.r_embedding = r_node_embedding  # 兄弟节点


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, feed_hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.feed_hidden = feed_hidden
        self.all_output = all_output


# class Encoder(nn.Module):
#     def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.emb_dim = emb_dim
#         self.hid_dim = hid_dim
#         self.n_layers = n_layers
#
#         self.embedding = nn.Embedding(input_dim, emb_dim)
#         self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=True)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, src, input_length):
#         embedded = self.embedding(src)
#         embedded = self.dropout(embedded)
#
#         pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
#         outputs, hidden, = self.rnn(pack)
#         output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
#         problem_output = output[-1, :, :self.hid_dim] + output[0, :, self.hid_dim:]
#         output = output[:, :, self.hid_dim:] + output[:, :, :self.hid_dim]
#         hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
#         hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
#         return output, hidden, problem_output

def max_pooling(model_output):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    return torch.mean(token_embeddings, 1)[0]

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, model_name, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, input_length):
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([1 for _ in range(i)] + [0 for _ in range(i, max_len)])
        seq_mask = torch.FloatTensor(seq_mask)
        if USE_CUDA:
            seq_mask = seq_mask.cuda()
        embedded = self.embedding(src.transpose(0, 1), attention_mask=seq_mask, return_dict=True)

        output = embedded[0].transpose(0, 1)
        hidden = output[0] + output[-1]

        problem_output = output[-1, :, :] + output[0, :, :]
        # print(hidden.size(), problem_output.size(), "--------------")

        return output, hidden, problem_output

# class Encoder(nn.Module):
#     def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.emb_dim = emb_dim
#         self.hid_dim = hid_dim
#         self.n_layers = n_layers
#
#         self.embedding = nn.Embedding(input_dim, emb_dim)
#         self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=True)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, src, input_length):
#         embedded = self.embedding(src)
#         embedded = self.dropout(embedded)
#
#         pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
#         outputs, hidden, = self.rnn(pack)
#         output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
#         problem_output = output[-1, :, :self.hid_dim] + output[0, :, self.hid_dim:]
#         output = output[:, :, self.hid_dim:] + output[:, :, :self.hid_dim]
#         hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
#         hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
#         return output, hidden, problem_output

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None, dropout=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        if dropout is not None:
            attn_energies = dropout(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class DotAttn(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttn, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None, dropout=None):
        attn_energies = hidden.transpose(0, 1).bmm(encoder_outputs.permute(1, 2, 0)).squeeze(1)
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        if dropout is not None:
            attn_energies = dropout(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


# class MinusGate(nn.Module):
#     def __init__(self, hidden_size, n_layers=4):
#         super(MinusGate, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#
#         self.i_gates = nn.Linear(hidden_size * 2, hidden_size)
#         self.c_gates_d = []
#         self.z_gates_d = []
#         self.c_gates_s = []
#         self.z_gates_s = []
#
#         for _ in range(n_layers):
#             self.c_gates_d.append(nn.Linear(hidden_size * 2, hidden_size // 2))
#             self.z_gates_d.append(nn.Linear(hidden_size, hidden_size // 2))
#             self.c_gates_s.append(nn.Linear(hidden_size * 2, hidden_size // 2))
#             self.z_gates_s.append(nn.Linear(hidden_size, hidden_size // 2))
#
#         self.c_gates_d = nn.ModuleList(self.c_gates_d)
#         self.z_gates_d = nn.ModuleList(self.z_gates_d)
#         self.c_gates_s = nn.ModuleList(self.c_gates_s)
#         self.z_gates_s = nn.ModuleList(self.z_gates_s)
#
#     def forward(self, pinput, context):
#         finput = torch.relu(self.i_gates(torch.cat((pinput, context), -1)))  # information merge
#         finput_s, finput_d = torch.split(finput, self.hidden_size // 2, -1)
#         finput_s_c = finput_s
#         del finput
#
#         for layer in range(self.n_layers):
#             c_d = torch.relu(self.c_gates_d[layer](torch.cat((pinput, context), -1)))
#             c_s = torch.relu(self.c_gates_s[layer](torch.cat((pinput, context), -1)))
#             z_d = torch.sigmoid(self.z_gates_d[layer](torch.cat((finput_d, c_d), -1)))
#             z_s = torch.sigmoid(self.z_gates_s[layer](torch.cat((finput_s_c, c_s), -1)))
#             finput_s_c = finput_s - c_s * z_s
#             finput_d = finput_d - c_d * z_d
#
#         finput = torch.cat((finput_s_c, finput_d), -1)
#
#         return finput

# class MinusGate(nn.Module):
#     def __init__(self, hidden_size):
#         super(MinusGate, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.i_gates = nn.Linear(hidden_size * 2, hidden_size)
#         self.c_gates = []
#         self.z_gates = []
#         self.norms = []
#
#         for _ in range(4):
#             self.c_gates.append(nn.Linear(hidden_size * 2, hidden_size))
#             self.z_gates.append(nn.Linear(hidden_size * 2, hidden_size))
#             self.norms.append(nn.LayerNorm(hidden_size))
#
#         self.z_gates.append(nn.Linear(hidden_size * 2, hidden_size))
#         self.norms.append(nn.LayerNorm(hidden_size))
#
#         self.c_gates = nn.ModuleList(self.c_gates)
#         self.z_gates = nn.ModuleList(self.z_gates)
#         self.norms = nn.ModuleList(self.norms)
#
#     def forward(self, pinput, context):
#         finput = torch.relu(self.i_gates(torch.cat((pinput, context), -1)))  # information merge
#
#         c_z = 0
#         c = torch.relu(self.c_gates[0](torch.cat((pinput, context), -1)))
#         z = torch.sigmoid(self.z_gates[0](torch.cat((finput, c), -1)))
#
#         fu = self.norms[0](c * z)
#         f1 = finput - fu
#         c_z += fu
#
#         c = torch.relu(self.c_gates[1](torch.cat((pinput, context), -1)))
#         z = torch.sigmoid(self.z_gates[1](torch.cat((f1, c), -1)))
#         fu = self.norms[1](c * z)
#         c_z += 2 * fu
#         f2 = finput - fu
#         c = torch.relu(self.c_gates[2](torch.cat((pinput, context), -1)))
#         z = torch.sigmoid(self.z_gates[2](torch.cat((f2, c), -1)))
#         fu = self.norms[2](c * z)
#         c_z += 2 * fu
#         f3 = finput - fu
#
#         c = torch.relu(self.c_gates[3](torch.cat((pinput, context), -1)))
#         z = torch.sigmoid(self.z_gates[3](torch.cat((f3, c), -1)))
#         fu = self.norms[3](c * z)
#         f4 = finput - fu
#         z = torch.sigmoid(self.z_gates[4](torch.cat((f4, c), -1)))
#
#         c_z += self.norms[4](c * z)
#
#         return finput - c_z / 6.0

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class MinusGate(nn.Module):
    def __init__(self, hidden_size, use_pos=False):
        super(MinusGate, self).__init__()

        self.hidden_size = hidden_size
        self.use_pos = use_pos

        pe = torch.zeros(MAX_OUTPUT_LENGTH, self.hidden_size)
        position = torch.arange(0, MAX_OUTPUT_LENGTH).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, self.hidden_size, 2, dtype=torch.float) * -(math.log(10000.0) / self.hidden_size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

        self.i_gates = nn.Linear(hidden_size * 2, hidden_size)
        self.c_gates = []
        self.z_gates = []
        # self.g_gates = []
        self.norms = []
        self.bn_pos_1 = nn.LayerNorm(hidden_size)
        self.bn_pos_2 = nn.LayerNorm(hidden_size)

        for _ in range(1):
            self.c_gates.append(nn.Linear(hidden_size * 2, hidden_size))
            self.z_gates.append(nn.Linear(hidden_size * 2, hidden_size))

        for _ in range(2):
            self.norms.append(nn.LayerNorm(hidden_size))

        # for _ in range(1):
        #     self.g_gates.append(nn.Linear(hidden_size * 2, hidden_size))

        self.c_gates = nn.ModuleList(self.c_gates)
        self.z_gates = nn.ModuleList(self.z_gates)
        # self.g_gates = nn.ModuleList(self.g_gates)
        self.norms = nn.ModuleList(self.norms)

    # def forward(self, pinput, context): #context 作为消融物
    #     # = torch.relu(self.i_gates(torch.cat((pinput, context), -1)))  # information merge
    #     finput = self.norms[0]((1 - torch.sigmoid(self.i_gates(torch.cat((pinput, context), -1)))) * pinput)  #信息映射
    #
    #     c = torch.relu(self.c_gates[0](torch.cat((finput, context), -1)))
    #     z = torch.sigmoid(self.z_gates[0](torch.cat((c, context), -1)))
    #     fu1 = self.norms[1](c * z)
    #     f1 = finput - fu1
    #
    #     return f1

    def forward(self, pinput, context): #context 作为消融物
        # = torch.relu(self.i_gates(torch.cat((pinput, context), -1)))  # information merge

        if self.use_pos:
            pinput = pinput * math.sqrt(self.hidden_size)
            pinput = pinput + self.pe[:pinput.size(0)]
            context = context * math.sqrt(self.hidden_size)
            context = context + self.pe[pinput.size(0)-1]

        finput = self.norms[0]((1 - torch.sigmoid(self.i_gates(torch.cat((pinput, context), -1)))) * pinput) #信息映射

        c = torch.relu(self.c_gates[0](torch.cat((finput, context), -1)))
        z = torch.sigmoid(self.z_gates[0](torch.cat((c, context), -1)))
        fu1 = self.norms[1](c * z)
        f1 = finput - fu1

        return f1

    # class Decoder(nn.Module):


#     def __init__(self, hidden_size, n_layers=2, dropout=0.5):
#         super(Decoder, self).__init__()

#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = dropout

#         self.dropout = nn.Dropout(dropout)
#         self.attn = Attn(hidden_size)
#         self.remain = nn.Linear(hidden_size * 2, hidden_size)
#         self.gf = MinusGate(hidden_size, 1)
#         self.g1 = MinusGate(hidden_size, 1)
#         self.g2 = MinusGate(hidden_size, 1)
#         self.g3 = MinusGate(hidden_size, 1)
#         self.g4 = MinusGate(hidden_size, 1)
#         self.attn_feed = Attn(hidden_size)
#         self.norm = nn.InstanceNorm1d(hidden_size)

#     def forward(self, last_embedded, last_context, feed_hidden, word_dict_vec, encoder_outputs, seq_mask):
#         last_embedded = self.dropout(last_embedded)

#         remain = torch.relu(
#             self.norm(self.remain(torch.cat((feed_hidden[-1].unsqueeze(0), last_embedded.unsqueeze(0)), -1))))

#         temp = remain.repeat(feed_hidden.size(0), 1, 1)
#         feed_hidden = self.gf(feed_hidden, temp)

#         query = self.g1(feed_hidden[-1].unsqueeze(0), last_embedded.unsqueeze(0))

#         query = self.g2(query, last_context.transpose(0, 1))

#         attn_weights = self.attn_feed(query, feed_hidden)
#         context_f = attn_weights.bmm(feed_hidden.transpose(0, 1))
#         new_hidden = self.g3(query, context_f.transpose(0, 1))

#         attn_weights = self.attn(new_hidden, encoder_outputs, seq_mask)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
#         output = self.g4(new_hidden, context.transpose(0, 1))

#         return output.squeeze(0), context, torch.cat((feed_hidden, new_hidden), dim=0)

class Compine(nn.Module):
    def __init__(self, hidden_size, num_size):
        super(Compine, self).__init__()

        self.hidden_size = hidden_size

        self.com_g = nn.Linear(hidden_size * num_size, hidden_size)
        self.com_c = nn.Linear(hidden_size * num_size, hidden_size)

    def forward(self, input):
        return torch.sigmoid(self.com_g(input)) * torch.relu(self.com_c(input))



class Decoder(nn.Module):
    def __init__(self, hidden_size, operator_index, n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.fuse = nn.Linear(hidden_size * 2, hidden_size)
        #self.dropout = nn.Dropout(dropout)
        self.attn = Attn(hidden_size)
        self.remain = MinusGate(hidden_size)
        # self.remain1 = MinusGate(hidden_size)
        self.gf = MinusGate(hidden_size, use_pos=False)
        self.g1 = MinusGate(hidden_size)
        # self.g2 = MinusGate(hidden_size)
        # self.g3 = MinusGate(hidden_size)
        self.out_l = MinusGate(hidden_size)
        self.out_r = MinusGate(hidden_size)
        # self.combine_tpl = Compine(hidden_size)
        # self.combine_tpr = Compine(hidden_size)
        # self.combine_pl = Compine(hidden_size)
        # self.combine_pr = Compine(hidden_size)
        # self.combine = Compine(hidden_size)
        # self.combine_p = Compine(hidden_size)
        self.combine_p = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_l = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_r = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_plr = nn.Linear(hidden_size * 3, hidden_size)
        self.combine_lr = nn.Linear(hidden_size * 2, hidden_size)
        #self.combine_fe = nn.Linear(hidden_size * 2, hidden_size)
        # self.combine_p = nn.Linear(hidden_size * 2, hidden_size)
        # self.combine_r = nn.Linear(hidden_size * 2, hidden_size)


        self.attn_feed = Attn(hidden_size)
        self.operator_index = operator_index

    def forward(self, input_ids, nodes_word_dict_vector, nodes_parent_index, parent_index, bool_index, bool_parent, \
                last_context, feed_hidden,
                word_dict_vec, encoder_outputs, seq_mask, parent_r_node, first_node=False):

        trg_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
        last_embedded = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).squeeze(1)  # 真实标签的向量

        # last_embedded = self.dropout(last_embedded)

        last_context = self.g1(last_context.transpose(0, 1), last_embedded.unsqueeze(0))
        remain = self.remain(feed_hidden[-1].unsqueeze(0), last_embedded.unsqueeze(0))

        remain = torch.relu(self.fuse(torch.cat((remain, last_context),dim=-1)))

        # remain = self.remain(feed_hidden[-1].unsqueeze(0), last_embedded.unsqueeze(0))
        # remain = self.g1(remain, last_context.transpose(0, 1))


        temp = remain.repeat(feed_hidden.size(0), 1, 1)
        feed_hidden = self.gf(feed_hidden, temp)  # 问题消融

        if first_node:
            nodes_word_dict_vector = last_embedded.unsqueeze(0)
            if USE_CUDA:
                nodes_word_dict_vector = nodes_word_dict_vector.cuda()
        else:
            nodes_word_dict_vector = torch.cat((nodes_word_dict_vector, last_embedded.unsqueeze(0)), 0)

        batch_size = feed_hidden.size(1)

        parent_embedded = []
        parent_feed_hidden = []
        length = feed_hidden.size(0) - 1

        nodes_word_dict_vector = nodes_word_dict_vector.transpose(0, 1)
        feed_hidden = feed_hidden.transpose(0, 1)

        for i in range(batch_size):
            nodes_word_dict_vector_idx = nodes_word_dict_vector[i]
            feed_hidden_idx = feed_hidden[i]
            nodes_parent_idx = nodes_parent_index[i]
            bool_p_idx = bool_index[i]
            truth = input_ids[i]
            parent_r_idx = parent_r_node[i]

            if len(nodes_parent_idx) >= 3:
                l_index = nodes_parent_idx[-2][0]
                r_index = nodes_parent_idx[-1][0]
                p_index = nodes_parent_idx[-3][0]
                l_node = nodes_parent_idx[-2][1]
                r_node = nodes_parent_idx[-1][1]
                p_node = nodes_parent_idx[-3][1]

                if l_index == r_index and bool_p_idx[-1] == 1 and bool_p_idx[-2] == 1:
                    while l_index == r_index and bool_p_idx[-1] == 1 and bool_p_idx[-2] == 1:
                        # c = self.combine_tpl(torch.cat((nodes_word_dict_vector_idx[p_node].unsqueeze(0),
                        #                                            nodes_word_dict_vector_idx[l_node].unsqueeze(0)),
                        #                                           -1))
                        # #c = nodes_word_dict_vector_idx[p_node].unsqueeze(0)
                        # c = self.combine_tpr(torch.cat((c,
                        #                                            nodes_word_dict_vector_idx[r_node].unsqueeze(0)),
                        #                                           -1))
                        left = torch.relu(self.combine_l(torch.cat((nodes_word_dict_vector_idx[l_node].unsqueeze(0), feed_hidden_idx[l_node].unsqueeze(0)),-1)))
                        right = torch.relu(self.combine_r(torch.cat(
                            (nodes_word_dict_vector_idx[r_node].unsqueeze(0), feed_hidden_idx[r_node].unsqueeze(0)),
                            -1)))
                        parent = torch.relu(self.combine_p(torch.cat(
                            (nodes_word_dict_vector_idx[p_node].unsqueeze(0), feed_hidden_idx[p_node].unsqueeze(0)),
                            -1)))

                        c = torch.relu(self.combine_plr(torch.cat((left, right, parent), -1)))

                        #nodes_word_dict_vector_idx[p_node] = c
                        nodes_parent_idx.pop()
                        nodes_parent_idx.pop()
                        bool_p_idx.pop()
                        bool_p_idx.pop()
                        bool_p_idx[-1] = 1

                        # c = torch.relu(self.combine_plr(torch.cat((feed_hidden_idx[l_node].unsqueeze(0), feed_hidden_idx[r_node].unsqueeze(0), \
                        #                             feed_hidden_idx[p_node].unsqueeze(0)), -1)))
                        #
                        # c = torch.relu(self.combine_fe(torch.cat((c, c1),-1)))
                        feed_hidden_idx[p_node] = c
                        del c
                        # c = self.combine_pl(torch.cat((feed_hidden_idx[p_node].unsqueeze(0),
                        #                                           feed_hidden_idx[l_node].unsqueeze(0)), -1))
                        # c = self.combine_pr(torch.cat((c,
                        #                                           feed_hidden_idx[r_node].unsqueeze(0)), -1))
                        # feed_hidden_idx[p_node] = c
                        # del c

                        if len(nodes_parent_idx) >= 3:
                            l_index = nodes_parent_idx[-2][0]
                            r_index = nodes_parent_idx[-1][0]
                            p_index = nodes_parent_idx[-3][0]
                            l_node = nodes_parent_idx[-2][1]
                            r_node = nodes_parent_idx[-1][1]
                            p_node = nodes_parent_idx[-3][1]
                        else:
                            r_index = nodes_parent_idx[-1][0]
                            r_node = nodes_parent_idx[-1][1]
                            break

                    parent_index[i] = r_index
                    bool_parent[i] = 0
                    # c = torch.relu(self.combine_r(torch.cat((nodes_word_dict_vector_idx[r_index].unsqueeze(0),
                    #                                        nodes_word_dict_vector_idx[r_node].unsqueeze(0)), -1)))
                    #
                    # nodes_word_dict_vector_idx[r_index] = c
                    parent_embedded.append(nodes_word_dict_vector_idx[r_index])

                    c = parent_r_idx[r_node].r_embedding
                    # c = torch.relu(self.combine_p(torch.cat((c.unsqueeze(0),
                    #                                          feed_hidden_idx[r_node].unsqueeze(0)), -1)))

                    parent_feed_hidden.append(c)
                    del c

                elif truth <= self.operator_index:
                    parent_index[i] = length
                    bool_parent[i] = 1
                    parent_embedded.append(nodes_word_dict_vector_idx[-1])
                    parent_feed_hidden.append(feed_hidden_idx[-1])
                else:
                    parent_index[i] = nodes_parent_idx[-1][0]
                    bool_parent[i] = 1
                    c = parent_r_idx[nodes_parent_idx[-1][1]].r_embedding
                    c = torch.relu(self.combine_lr(torch.cat((c.unsqueeze(0),
                                                             feed_hidden_idx[-1].unsqueeze(0)), -1)))
                    parent_feed_hidden.append(c.squeeze(0))
                    del c
                    # c = self.combine_tpl(torch.cat((nodes_word_dict_vector_idx[nodes_parent_idx[-1][0]].unsqueeze(0),
                    #                                 nodes_word_dict_vector_idx[-1].unsqueeze(0)), -1))
                    # nodes_word_dict_vector_idx[nodes_parent_idx[-1][0]] = c
                    parent_embedded.append(nodes_word_dict_vector_idx[nodes_parent_idx[-1][0]])

            elif truth <= self.operator_index:
                parent_index[i] = length
                bool_parent[i] = 1
                parent_embedded.append(nodes_word_dict_vector_idx[-1])
                parent_feed_hidden.append(feed_hidden_idx[-1])
            else:
                parent_index[i] = nodes_parent_idx[-1][0]
                bool_parent[i] = 1
                c = parent_r_idx[nodes_parent_idx[-1][1]].r_embedding
                c = torch.relu(self.combine_lr(torch.cat((c.unsqueeze(0),
                                                         feed_hidden_idx[-1].unsqueeze(0)), -1)))
                parent_feed_hidden.append(c.squeeze(0))
                del c
                # c = self.combine_tpl(torch.cat((nodes_word_dict_vector_idx[nodes_parent_idx[-1][0]].unsqueeze(0),
                #                                 nodes_word_dict_vector_idx[-1].unsqueeze(0)), -1))
                # nodes_word_dict_vector_idx[nodes_parent_idx[-1][0]] = c
                parent_embedded.append(nodes_word_dict_vector_idx[nodes_parent_idx[-1][0]])

        # last_embedded = torch.stack(parent_embedded, 0) #父节点embedding
        parent_feed_hidden = torch.stack(parent_feed_hidden, 0)  # 当前树表达式所有节点的剩余信息
        feed_hidden = feed_hidden.transpose(0, 1)

        # remain = self.g1(parent_feed_hidden.unsqueeze(0), remain) #问题聚焦
        # remain = self.g2(remain, last_context.transpose(0, 1))

        # temp = remain.repeat(feed_hidden.size(0), 1, 1)
        # feed_hidden = self.gf(feed_hidden, temp) #问题消融

        # query = self.g1(parent_feed_hidden.unsqueeze(0), last_embedded.unsqueeze(0)) #父节点的问题消融

        attn_weights = self.attn_feed(parent_feed_hidden.unsqueeze(0), feed_hidden)

        new_hidden = attn_weights.bmm(feed_hidden.transpose(0, 1)).transpose(0, 1)  # 问题选择

        # new_hidden = self.g3(remain, context_f.transpose(0, 1)) #问题聚焦

        attn_weights = self.attn(new_hidden, encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # 消融好的新上下文信息
        output_l = self.out_l(new_hidden, context.transpose(0, 1))
        output_r = self.out_r(new_hidden, context.transpose(0, 1))

        nodes_word_dict_vector = nodes_word_dict_vector.transpose(0, 1)

        return output_l.squeeze(0), output_r.squeeze(0), context, torch.cat((feed_hidden, new_hidden),
                                                                            dim=0), nodes_word_dict_vector, parent_index, \
               bool_parent


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, stuff_size, hidden_size, output_lang, beam_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.stuff_size = stuff_size
        self.output_lang = output_lang
        self.beam_size = beam_size

        self.embedding_weight = torch.nn.Parameter(torch.randn((stuff_size, hidden_size)))
        self.score = Score(hidden_size, hidden_size)
        self.encoder = encoder
        self.decoder = decoder

        #self.operator_index = output_lang.word2index['^']
        self.operator_index = output_lang.word2index['+']

    def forward(self, src, input_length, trg, num_pos, output_lang, teacher_forcing, num_stack=None):
        if num_stack is not None:
            num_stack = deepcopy(num_stack)

        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask)

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        num_mask = []
        for pos in num_pos:
            num_mask.append(
                [0] * self.stuff_size + [0 for _ in range(len(pos))] + [1 for _ in range(len(pos), num_size)])
        num_mask = torch.BoolTensor(num_mask)

        if USE_CUDA:
            seq_mask = seq_mask.cuda()
            num_mask = num_mask.cuda()
        assert len(num_pos) == src.size(1)
        batch_size = src.size(1)
        if trg is not None:
            max_len = trg.shape[0]
        else:
            max_len = MAX_OUTPUT_LENGTH

        encoder_out, hidden, problem = self.encoder.forward(src, input_length)

        # make output word dict
        word_dict_vec = self.embedding_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        num_embedded = get_all_number_encoder_outputs(encoder_out, num_pos, batch_size, num_size, self.hidden_size)
        word_dict_vec = torch.cat((word_dict_vec, num_embedded), dim=1)

        out_list = []
        decoder_hidden = problem.unsqueeze(0).transpose(0, 1)
        feed_hidden = torch.relu(problem.unsqueeze(0))
        input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

        nodes_word_dict_vector = None
        parent_index = torch.LongTensor([0 for _ in range(batch_size)])  # 保存当前节点的父节点
        bool_index = [[] for _ in range(batch_size)]  # 保存每一个节点是否可以融合
        bool_parent = [1 for _ in range(batch_size)]  # 保存每一个节点是否加入栈
        nodes_parent_index = [[] for _ in range(batch_size)]  # 保存每一个节点的父节点
        parent_r_node = [[TreeNode(None)] for _ in range(batch_size)]  # 保存每一个节点的右子树向量

        if USE_CUDA:
            input = input.cuda()
            parent_index = parent_index.cuda()
            # nodes_word_dict_vector = nodes_word_dict_vector.cuda()
        if teacher_forcing > random.random():
            first_node = True

            for t in range(0, max_len - 1):

                output_l, output_r, decoder_hidden, feed_hidden, nodes_word_dict_vector, parent_index, bool_parent = \
                    self.decoder.forward(input, nodes_word_dict_vector, nodes_parent_index, \
                                         parent_index, bool_index, bool_parent,
                                         decoder_hidden, feed_hidden,
                                         word_dict_vec, encoder_out, seq_mask, parent_r_node, first_node)

                first_node = False
                score_list = self.score(output_l.unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input = generate_decoder_input(trg[t + 1], score_list, num_stack, self.stuff_size,
                                               output_lang.word2index["UNK"])
                trg[t + 1] = input

                for i in range(batch_size):
                    p = parent_index[i].cpu()
                    nodes_parent_index[i].append((p, t + 1))
                    parent_r_node[i].append(TreeNode(output_r[i]))
                    if input[i] > self.operator_index:  # 表示trg是数字
                        bool_index[i].append(1)
                    else:
                        bool_index[i].append(0)

            decoder_out = torch.stack(out_list, dim=0)
        else:
            first_node = True

            for t in range(0, max_len - 1):

                output_l, output_r, decoder_hidden, feed_hidden, nodes_word_dict_vector, parent_index, bool_parent = \
                    self.decoder.forward(input, nodes_word_dict_vector, nodes_parent_index, \
                                         parent_index, bool_index, bool_parent,
                                         decoder_hidden, feed_hidden,
                                         word_dict_vec, encoder_out, seq_mask, parent_r_node, first_node)

                first_node = False
                score_list = self.score(output_l.unsqueeze(1), word_dict_vec, num_mask)

                out_list.append(score_list)

                input = torch.argmax(score_list, dim=-1)

                for i in range(batch_size):
                    p = parent_index[i].cpu()
                    nodes_parent_index[i].append((p, t + 1))
                    parent_r_node[i].append(TreeNode(output_r[i]))
                    if input[i] > self.operator_index:  # 表示trg是数字
                        bool_index[i].append(1)
                    else:
                        bool_index[i].append(0)

            decoder_out = torch.stack(out_list, dim=0)

        return decoder_out
