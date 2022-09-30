import copy
import random

import numpy
import torch

from src.expressions_transfer import *

USE_CUDA = torch.cuda.is_available()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def train_masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    length = length - 1
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = torch.sum(losses, 1) / length.float().sum()
    loss_score = sorted(loss.detach().cpu().numpy())
    loss_score_1 = copy.deepcopy(loss_score)

    def func1(t):
      l = len(loss_score) // 4
      a,b,c,d,e = 0, l, l * 2, l * 3, len(loss_score)-1
      if t >= loss_score[a] and t < loss_score[b]:
        return 0
      if t >= loss_score[b] and t < loss_score[c]:
        return 1
      if t >= loss_score[c] and t < loss_score[d]:
        return 2
      if t >= loss_score[d] and t <= loss_score[e]:
        return 3

    loss_score = 0.97 ** (torch.cuda.FloatTensor(numpy.array(list(map(func1,loss_score_1)), dtype=numpy.float64)))
    loss = (loss * loss_score).sum()
    #loss = losses.sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

def masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    length = length - 1
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    #log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
    log_probs_flat = torch.log2(torch.nn.functional.softmax(logits_flat, dim=1) + 1e-8)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum()  / length.float().sum()
    #loss = losses.sum() / losses.size(0)
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

def train(input_batch, input_length, output_batch, output_length, num_stack_batch, output_lang, num_pos,
          model, optimizer, teacher_forcing):
    model.train()

    optimizer.zero_grad()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()

    output = model.forward(input_var, input_length, trg, num_pos, output_lang, teacher_forcing, num_stack_batch)
    trg = trg[1:]
    loss = masked_cross_entropy(output.transpose(0, 1).contiguous(), trg.transpose(0, 1).contiguous(), output_length)

    loss.backward()
    # print([x.grad for x in optimizer.param_groups[0]['params']])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.999)
    optimizer.step()

    return loss.item()

def train_beam_search(input_batch, input_length, output_batch, output_length, num_stack_batch, output_lang, num_pos,
          model, optimizer, teacher_forcing):
    model.train()

    optimizer.zero_grad()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()

    _, output = model.forward(input_var, input_length, trg, num_pos, output_lang, teacher_forcing, num_stack_batch)
    trg = trg[1:]
    loss = masked_cross_entropy(output.transpose(0, 1).contiguous(), trg.transpose(0, 1).contiguous(), output_length)

    loss.backward()
    # print([x.grad for x in optimizer.param_groups[0]['params']])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.999)
    optimizer.step()

    return loss.item()

'''
def evaluate(input_batch, input_length, output_batch, output_length, output_lang, num_pos, model):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()
    output = model.forward(input_var, input_length, trg, num_pos, output_lang, -1)
    trg = trg[1:]
    loss = masked_cross_entropy(output.transpose(0, 1).contiguous(), trg.transpose(0, 1).contiguous(), output_length)

    output = torch.argmax(output.transpose(0, 1), dim=-1)
    if USE_CUDA:
        output = output.cpu()
    return output.detach().numpy(), loss.item()
'''

def evaluate(input_batch, input_length, output_batch, output_length, output_lang, num_pos, model):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()
    output = model.forward(input_var, input_length, trg, num_pos, output_lang, -1)

    trg = trg[1:]
    loss = masked_cross_entropy(output.transpose(0, 1).contiguous(), trg.transpose(0, 1).contiguous(), output_length)

    output = torch.argmax(output.transpose(0, 1), dim=-1)

    if USE_CUDA:
        output = output.cpu()
    return output.detach().numpy(), loss.item()

def evaluate_beam_search(input_batch, input_length, output_batch, output_length, output_lang, num_pos, model, beam_size=5):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()

    output, _ = model.forward(input_var, input_length, trg, num_pos, output_lang, -1, beam_size=beam_size)

    # loss = masked_cross_entropy(output.transpose(0, 1).contiguous(), trg.transpose(0, 1).contiguous(), output_length)
    #
    # output = torch.argmax(output.transpose(0, 1), dim=-1)

    if USE_CUDA:
        output = output.cpu()
    return output.detach().numpy()

def train_bert(input_batch, input_length, output_batch, num_stack_batch, output_lang, num_pos,
               model, optimizer, criterion, teacher_forcing):
    model.train()

    optimizer.zero_grad()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()

    output = model.forward(input_var, input_length, trg)
    trg = trg[1:]
    loss = criterion(output.permute(1, 2, 0), trg.transpose(0, 1))

    loss.backward()

    optimizer.step()

    return loss.item()


def evaluate_bert(input_batch, input_length, output_batch, output_lang, num_pos, model, criterion):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = torch.LongTensor(output_batch).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()
    output = model.forward(input_var, input_length, trg)
    trg = trg[1:]
    loss = criterion(output.permute(1, 2, 0), trg.transpose(0, 1))

    output = torch.argmax(output.transpose(0, 1), dim=-1)
    if USE_CUDA:
        output = output.cpu()
    return output.detach().numpy(), loss.item()


def bitrain(input_batch, input_length, output_batch, num_stack_batch, output_lang, num_pos,
            model, optimizer, criterion, teacher_forcing):
    model.train()

    optimizer.zero_grad()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    trg = torch.LongTensor(output_batch[0]).transpose(0, 1)
    trg2 = torch.LongTensor(output_batch[1]).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()
        trg2 = trg2.cuda()

    output, output2 = model.forward(input_var, input_length, trg, trg2, num_pos, output_lang, teacher_forcing)

    trg = trg[1:]
    trg2 = trg2[1:]
    loss1 = criterion(output.permute(1, 2, 0), trg.transpose(0, 1))
    loss2 = criterion(output2.permute(1, 2, 0), trg2.transpose(0, 1))

    loss = loss1 + loss2
    loss.backward()

    optimizer.step()

    return loss.item()


def bievaluate(input_batch, input_length, output_batch, output_lang, num_pos, model, criterion):
    model.eval()

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    trg = torch.LongTensor(output_batch[0]).transpose(0, 1)
    trg2 = torch.LongTensor(output_batch[1]).transpose(0, 1)
    if USE_CUDA:
        input_var = input_var.cuda()
        trg = trg.cuda()
        trg2 = trg2.cuda()

    output, output2 = model.forward(input_var, input_length, trg, trg2, num_pos, output_lang, -1)
    trg = trg[1:]
    trg2 = trg2[1:]
    loss1 = criterion(output.permute(1, 2, 0), trg.transpose(0, 1))
    loss2 = criterion(output2.permute(1, 2, 0), trg2.transpose(0, 1))
    loss = loss1 + loss2

    output = torch.argmax(output.transpose(0, 1), dim=-1)
    output2 = torch.argmax(output2.transpose(0, 1), dim=-1)
    if USE_CUDA:
        output = output.cpu()
        output2 = output2.cpu()
    return output.detach().numpy(), output2.detach().numpy(), loss.item()


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack=None):
    # print(test_res, test_tar)
    temp = []
    for i in test_res:
        word = output_lang.index2word[i]
        if word == 'EOS':
            break
        if word == 'SOS':
            temp.reverse()
            break
        temp.append(i)
    test_res = temp


    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar[1:], output_lang, num_list, num_stack)

    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar
