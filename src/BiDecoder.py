import random

import torch
from torch import nn

USE_CUDA = torch.cuda.is_available()
MAX_OUTPUT_LENGTH = 46


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, feed_hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.feed_hidden = feed_hidden
        self.all_output = all_output


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, input_length):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        pack = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        outputs, hidden, = self.rnn(pack)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        problem_output = output[-1, :, :self.hid_dim] + output[0, :, self.hid_dim:]
        output = output[:, :, self.hid_dim:] + output[:, :, :self.hid_dim]
        hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        return output, hidden, problem_output


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
        return torch.nn.functional.log_softmax(score, dim=-1)


class Decoder(nn.Module):
    def __init__(self, hidden_size, n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size * 3, hidden_size, n_layers, dropout=dropout)
        self.hidden_cat = nn.Linear(hidden_size * 2, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.attn = Attn(hidden_size)
        self.attn2 = Attn(hidden_size)

        self.w_g = nn.Linear(hidden_size * 2, hidden_size)
        self.w_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_ids, last_hidden, feed_hidden, word_dict_vec, encoder_outputs, seq_mask, other_feed):
        trg_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=word_dict_vec.size(1)).float()
        last_embedded = trg_one_hot.unsqueeze(1).bmm(word_dict_vec).squeeze(1)
        last_embedded = self.dropout(last_embedded)

        g = torch.sigmoid(self.w_g(torch.cat((last_embedded, feed_hidden), -1)))
        c = torch.tanh(self.w_c(torch.cat((last_embedded, feed_hidden), -1)))
        new_hidden = g * c

        attn_weights = self.attn(new_hidden.unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        attn_weights = self.attn2(new_hidden.unsqueeze(0), other_feed)
        context2 = attn_weights.bmm(other_feed.transpose(0, 1))

        rnn_output, hidden = self.gru(
            torch.cat((new_hidden.unsqueeze(0), context.transpose(0, 1), context2.transpose(0, 1)), -1), last_hidden)

        output = torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), -1)))

        return output, hidden, new_hidden


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
        self.l2r = decoder[0]
        self.r2l = decoder[1]

    def forward(self, src, input_length, trg, trg2, num_pos, output_lang, teacher_forcing):

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
        if trg != None:
            max_len = trg.shape[0]
        else:
            max_len = MAX_OUTPUT_LENGTH

        encoder_out, hidden, feed_hidden = self.encoder.forward(src, input_length)

        # make output word dict
        word_dict_vec = self.embedding_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        num_embedded = get_all_number_encoder_outputs(encoder_out, num_pos, batch_size, num_size, self.hidden_size)
        word_dict_vec = torch.cat((word_dict_vec, num_embedded), dim=1)

        l2r_out = []
        r2l_out = []
        l2r_feed = torch.zeros((1, batch_size, self.hidden_size))
        r2l_feed = torch.zeros((1, batch_size, self.hidden_size))
        l2r_feed_hidden = feed_hidden
        r2l_feed_hidden = feed_hidden
        if USE_CUDA:
            l2r_feed = l2r_feed.cuda()
            r2l_feed = r2l_feed.cuda()
        if teacher_forcing > random.random():
            for t in range(0, max_len - 1):
                output, hidden, l2r_feed_hidden = self.l2r(trg[t], hidden, l2r_feed_hidden, word_dict_vec,
                                                           encoder_out, seq_mask, r2l_feed)
                l2r_feed = torch.cat((l2r_feed, output.unsqueeze(0)))
                l2r_score = self.score(output.unsqueeze(1), word_dict_vec, num_mask)
                l2r_out.append(l2r_score)

                output, hidden, r2l_feed_hidden = self.r2l(trg2[t], hidden, r2l_feed_hidden, word_dict_vec,
                                                           encoder_out, seq_mask, l2r_feed)
                r2l_feed = torch.cat((r2l_feed, output.unsqueeze(0)))
                r2l_score = self.score(output.unsqueeze(1), word_dict_vec, num_mask)
                r2l_out.append(r2l_score)
        else:
            l2r_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)
            r2l_input = torch.LongTensor([output_lang.word2index["EOS"]] * batch_size)
            if USE_CUDA:
                l2r_input = l2r_input.cuda()
                r2l_input = r2l_input.cuda()
            for t in range(0, max_len - 1):
                output, hidden, l2r_feed_hidden = self.l2r(l2r_input, hidden, l2r_feed_hidden, word_dict_vec,
                                                           encoder_out, seq_mask, r2l_feed)
                l2r_feed = torch.cat((l2r_feed, output.unsqueeze(0)))
                l2r_score = self.score(output.unsqueeze(1), word_dict_vec, num_mask)
                l2r_input = torch.argmax(l2r_score, dim=-1)
                l2r_out.append(l2r_score)

                output, hidden, r2l_feed_hidden = self.r2l(r2l_input, hidden, r2l_feed_hidden, word_dict_vec,
                                                           encoder_out, seq_mask, l2r_feed)
                r2l_feed = torch.cat((r2l_feed, output.unsqueeze(0)))
                r2l_score = self.score(output.unsqueeze(1), word_dict_vec, num_mask)
                r2l_input = torch.argmax(r2l_score, dim=-1)
                r2l_out.append(r2l_score)

        l2r_out = torch.stack(l2r_out, dim=0)
        r2l_out = torch.stack(r2l_out, dim=0)
        return l2r_out, r2l_out
