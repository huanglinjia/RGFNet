import math
import pickle
import time

import torch
from sklearn.utils import shuffle
from tqdm import tqdm

import torch.nn
from temp_graph import *
from src.pre_data_graph import *
from src.train_and_evaluate_graph import *
import warnings

warnings.filterwarnings("ignore")

USE_CUDA = torch.cuda.is_available()
torch.cuda.set_device(3)
fold = 0

size_split = 0.8
PAD_token = 0

dropout = 0.5
embedding_size = 128
hidden_size = 512
n_layer = 2
n_epochs = 150
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
swap_times = 4

data = load_raw_data("data/Math_23K.json")
graph_info = load_raw_graph("./data/graph2.json")

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p, g in zip(pairs, graph_info):
    temp_pairs.append(
        (p[0], from_infix_to_prefix(p[1]), p[2], p[3], g['character_group']))
pairs = temp_pairs

fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []

pairs_tested = []
pairs_trained = []
for fold_t in range(5):
    if fold_t == fold:
        pairs_tested += fold_pairs[fold_t]
    else:
        pairs_trained += fold_pairs[fold_t]

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)

# with open('models/greedy_s2s/input_lang.pkl', 'wb') as f:
#     pickle.dump(input_lang, f)
#
# with open('models/greedy_s2s/output_lang.pkl', 'wb') as f:
#     pickle.dump(output_lang, f)

stuff_size = output_lang.num_start + len(generate_nums) + 4
print(output_lang.index2word, " stuff size:", stuff_size)

input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, \
        num_size_batches, graph_info_batches, unk_pos_batches = prepare_train_batch(
        train_pairs, batch_size)

input_batches_test, input_lengths_test, output_batches_test, output_lengths_test, nums_batches_test, num_stack_batches_test, \
        num_pos_batches_test, num_size_batches_test, graph_info_batches_test, unk_pos_batches_test = prepare_test_batch(
        test_pairs, batch_size)

enc = Encoder(input_dim=input_lang.n_words, emb_dim=hidden_size, hid_dim=hidden_size, n_layers=n_layer, dropout=dropout)
dec = Decoder(hidden_size=hidden_size, n_layers=n_layer, dropout=dropout)

model = Seq2Seq(encoder=enc, decoder=dec, stuff_size=stuff_size, hidden_size=hidden_size,
                output_lang=output_lang)

if USE_CUDA:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                       verbose=True, threshold=0.05, threshold_mode='rel')

best_valid_loss = float('inf')
best_vacc = -1
best_equacc = -1
tt = -1
for epoch in range(n_epochs):
    teacher_forcing = 1

    print("******************************************")
    print('Epoch ', epoch + 1)
    print('teacher forcing: ', round(teacher_forcing, 3),
          'learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
    start_time = time.time()

    tloss = 0
    t = tqdm(range(len(input_lengths)))
    for idx in t:
        graph_batch = get_single_graph_batch(graph_info_batches[idx], input_batches[idx], num_pos_batches[idx], unk_pos_batches[idx])
        loss = train(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                     num_stack_batches[idx], output_lang, num_pos_batches[idx], graph_batch,
                     model, optimizer, teacher_forcing)
        t.set_postfix(loss=loss)
        tloss += loss

    tloss /= len(input_lengths)
    print('Train Loss', tloss)
    scheduler.step(tloss)

    with torch.no_grad():
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        tloss = 0
        for idx in tqdm(range(len((input_lengths_test)))):

            graph_batch = get_single_graph_batch(graph_info_batches_test[idx], input_batches_test[idx], num_pos_batches_test[idx],
                                                 unk_pos_batches_test[idx])

            out, loss = evaluate(input_batches_test[idx], input_lengths_test[idx], output_batches_test[idx],
                                 output_lengths_test[idx], output_lang, num_pos_batches_test[idx], graph_batch, model)
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
        if epoch + 1 > 20:
            torch.save(model.state_dict(), "models/greedy_s2s/greedy_s2s")
    print("******************************************")

print("******************************Fold: ", fold + 1, "********************************")
print('best: ', best_equacc, best_vacc, tt, best_vacc / tt)
print("***********************************************************************************")
