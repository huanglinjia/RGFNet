import math
import pickle
import time

import torch
from sklearn.utils import shuffle
from tqdm import tqdm

import torch.nn
from temp_tree_14 import *   #temp_tree_2 74.1 temp_tree_4 (74.46\75.51\74/48) temp_tree_6 73.8  #test_4layer 75.43 test_2layer(temp_tree5) 74.91
from src.pre_data import *
from src.train_and_evaluate import *
import warnings
from schdule import WarmupLR
import numpy as np

transformer_name = "hfl/chinese-electra-180g-base-discriminator"

warnings.filterwarnings("ignore")
torch.cuda.set_device(1)
USE_CUDA = torch.cuda.is_available()

size_split = 0.8
PAD_token = 0

transformer_name = "hfl/chinese-electra-180g-base-discriminator"
dropout = 0.5
hidden_size = 768
n_layer = 4
n_epochs = 100
batch_size = 64
path = "models/electra2seq/"
seed = 22
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data = load_ape_data("data/train.ape.json")
pairs_trained, generate_nums, copy_nums = transfer_ape_num(data, id_file="data/ape_simple_id.txt")
temp_pairs = []
for p in pairs_trained:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs

data = load_ape_data("data/test.ape.json")
pairs_tested, _, _ = transfer_ape_num(data, id_file="data/ape_simple_test_id.txt")
temp_pairs = []
for p in pairs_tested:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_tested = temp_pairs

input_lang, output_lang, train_pairs, test_pairs = prepare_data_bert(pairs_trained, pairs_tested, 5, generate_nums,
                                                                     copy_nums, '/home/dj/hlj_MWP/weight/electra/vocab.txt', tree=True)

print(output_lang)
stuff_size = output_lang.num_start + len(generate_nums) + 4
print(output_lang.index2word, " stuff size:", stuff_size)

input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(
    train_pairs, batch_size)

input_batches_test, input_lengths_test, output_batches_test, output_lengths_test, nums_batches_test, num_stack_batches_test, num_pos_batches_test, _ = prepare_test_batch(
    test_pairs, batch_size)

# enc = Encoder(input_dim=input_lang.n_words, emb_dim=hidden_size, hid_dim=hidden_size, n_layers=n_layer, dropout=dropout)
enc = Encoder(input_dim=input_lang.n_words, emb_dim=hidden_size, hid_dim=hidden_size, n_layers=n_layer,
              model_name=transformer_name, dropout=dropout)
dec = Decoder(hidden_size=hidden_size, operator_index = output_lang.word2index['+'], n_layers=n_layer, dropout=dropout)

model = Seq2Seq(encoder=enc, decoder=dec, stuff_size=stuff_size, hidden_size=hidden_size,
                output_lang=output_lang)

if USE_CUDA:
    model.cuda()

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
#                                                        verbose=True, threshold=0.05, threshold_mode='rel')
enc_params = list(map(id, model.encoder.parameters()))
base_params = filter(lambda p: id(p) not in enc_params, model.parameters())

optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": base_params},
], lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                       verbose=True, threshold=0.1, threshold_mode='rel')

#scheduler = WarmupLR(op_scheduler, learning_rate / 5, num_warmup=5, warmup_strategy='linear')
# best_valid_loss = float('inf')
# best_vacc = -1
# best_equacc = -1
# tt = -1
# for epoch in range(1, n_epochs + 1):
#
#     # if epoch <= 5:
#     #     scheduler.step(epoch)
#
#     teacher_forcing = 1
#
#     print("******************************************")
#     print('Epoch ', epoch)
#     print('teacher forcing: ', round(teacher_forcing, 3),
#           'learning rate:', optimizer.param_groups[0]['lr'])
#     start_time = time.time()
#
#     tloss = 0
#     t = tqdm(range(len(input_lengths)))
#     for idx in t:
#         loss = train(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
#                      num_stack_batches[idx], output_lang, num_pos_batches[idx],
#                      model, optimizer, teacher_forcing)
#         t.set_postfix(loss=loss)
#         tloss += loss
#
#     tloss /= len(input_lengths)
#     print('Train Loss', tloss)
#
#     # if epoch > 5:
#     #     scheduler.step(tloss)
#     scheduler.step(tloss)
best_valid_loss = float('inf')
best_vacc = -1
best_equacc = -1
tt = -1
loss_his = []
for epoch in range(n_epochs):
    teacher_forcing = 1
    print("******************************************")
    print('Epoch ', epoch + 1)
    print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'], " || ",
          optimizer.state_dict()['param_groups'][1]['lr'])
    # print('learning rate:', optimizer.param_groups[0]['lr'])
    start_time = time.time()

    tloss = 0
    t = tqdm(range(len(input_lengths)))
    for idx in t:
        loss = train(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                     num_stack_batches[idx], output_lang, num_pos_batches[idx],
                     model, optimizer, teacher_forcing)
        loss_his.append(loss)
        t.set_postfix(loss=loss)
        tloss += loss

    # if epoch == 40:
    #     with open(path + 'loss_mlp.pkl', 'wb') as f:
    #         pickle.dump(loss_his, f)

    tloss /= len(input_lengths)
    print('Train Loss', tloss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    scheduler.step(tloss)
    with torch.no_grad():
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        tloss = 0
        for idx in tqdm(range(len((input_lengths_test)))):

            out, loss = evaluate(input_batches_test[idx], input_lengths_test[idx], output_batches_test[idx],
                                 output_lengths_test[idx], output_lang, num_pos_batches_test[idx], model)
            tloss += loss

            for i in range(out.shape[0]):
                eval_total += 1

                val_ac, equ_ac, _, _ = compute_prefix_tree_result(out[i].tolist(), output_batches_test[idx][i],
                                                                  output_lang, nums_batches_test[idx][i],
                                                                  num_stack_batches_test[idx][i])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1

        print('test: ', equation_ac, value_ac, eval_total, value_ac / eval_total)
        print('test loss:', tloss / len(input_lengths_test))

    if best_vacc < value_ac:
        best_vacc = value_ac
        best_equacc = equation_ac
        tt = eval_total
        # if epoch > 50:
        #     torch.save(model.state_dict(), path + 'model_rk4.pkl')

print("******************************Fold: ", fold + 1, "********************************")
print('best: ', best_equacc, best_vacc, tt, best_vacc / tt)
print("***********************************************************************************")
