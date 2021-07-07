import io
import pickle
import os
import re
import math
import datetime
from collections import defaultdict

import torch
import numpy as np
from nltk.parse import DependencyGraph

from constants import *

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def save_obj(obj, dir, name):
    with open(os.path.join(dir, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(obj_path):
    with open(obj_path, 'rb') as f:
        return pickle.load(f)


def normalize(word):
    return NUM_SYMBOL if numberRegex.match(word) else word.lower()


def vocab(data_path):
    words_count = defaultdict(int)
    unique_rels = {ROOT_SYMBOL, }
    unique_pos = {ROOT_SYMBOL, }

    with open(data_path, 'r') as fh:

        for line in fh:
            if line.startswith('#'):
                continue
            if line != '\n':

                parsed_line = line.strip().split('\t')

                idx, word, pos_tag, rel = parsed_line[0], normalize(parsed_line[1]), parsed_line[3], parsed_line[7]

                if len(word.split(' ')) > 1:
                    word = '_'.join(word.split(' '))
                if idx == '1':
                    words_count[ROOT_SYMBOL] += 1

                words_count[word] += 1
                unique_rels.add(rel)
                unique_pos.add(pos_tag)

    w2i = {word: i for i, word in enumerate(words_count, 1)}
    r2i = {rel: i for i, rel in enumerate(sorted(list(unique_rels)))}
    p2i = {pos_tag: i for i, pos_tag in enumerate(sorted(list(unique_pos)), 1)}

    w2i[UNK_SYMBOL], p2i[UNK_SYMBOL] = 0, 0

    return words_count, w2i, r2i, p2i


def load_pretrained_word_embed(ext_emb, vocab, dtype=np.float32):

    ex_w2i = {UNK_SYMBOL: 0, ROOT_SYMBOL: 1}
    word_vectors = []
    vec_idx = 2

    with io.open(ext_emb, 'r', encoding='utf-8', newline='\n', errors='ignore') as fh:
        for line in fh:
            parsed_line = line.rstrip().split(' ')
            word, word_vector = parsed_line[0], np.array([float(val) for val in parsed_line[1:]], dtype=dtype)

            if word in ex_w2i:
                continue
            if normalize(word) not in vocab:
                continue

            ex_w2i[word] = vec_idx
            word_vectors.append(word_vector)
            vec_idx += 1

    # add random word vectors for the special symbols and initialize them using Xavier uniform
    vec_dim = word_vectors[0].shape[-1]
    bound = math.sqrt(3. / vec_dim)
    special_word_vec = np.random.uniform(-bound, bound, (2, vec_dim)).astype(dtype)
    return ex_w2i, torch.from_numpy(np.concatenate((special_word_vec, np.stack(word_vectors))))


def to_dependency_graph(sentence, pos_tag, tree, rels):
    dep_graph_string = ''
    n = len(sentence)
    for i in range(n):
        dep_graph_string += sentence[i] + "\t" + pos_tag[i] + "\t" + str(tree[i]) + "\t" + str.upper(rels[i]) + "\n"
    return DependencyGraph(dep_graph_string)


def log_results(eval_dict, logger, data_split):
    logger.info('-----------+-----------+-----------+-----------+-----------')
    logger.info(data_split + ' results:')
    logger.info('{:20}|{:6}|'.format('Metric', 'Score'))
    logger.info('-----------+-----------+-----------+-----------+-----------')
    for metric_name, value in eval_dict.items():
        if 'loss' in metric_name:
            logger.info('{:20}|{:.4f}|'.format(metric_name, value))
        else:
            logger.info('{:20}|{:6.2f}|'.format(metric_name, 100 * value))


def write_results_file(eval_dict, output_file):
    with open(output_file, 'w') as f:
        f.write('{:20}|{:6}|\n'.format('Metric', 'Score'))
        f.write('-----------+-----------+-----------+-----------+-----------')
        for metric_name, value in eval_dict.items():
            f.write('\n')
            if 'loss' in metric_name:
                f.write('{:20}|{:.4f}|'.format(metric_name, value))
            else:
                f.write('{:20}|{:6.2f}|'.format(metric_name, 100 * value))


def write_predicted_labels(args, dep_graph_pred, data_split):
    if data_split == 'dev':
        data_path = args.dev_path
    else:
        data_path = args.test_path

    lines = []
    sentence_idx = 0
    with open(data_path, 'r') as f:
        for line in f:
            if line == '\n':
                lines.append(line)
                sentence_idx += 1
                continue
            if line.startswith('#'):
                lines.append(line)
                continue
            parsed_line = line.split('\t')
            if '-' in parsed_line[0] or '.' in parsed_line[0]:
                lines.append(line)
            else:
                word_idx = int(parsed_line[0])
                parsed_line[6] = str(dep_graph_pred[sentence_idx].nodes[word_idx]['head'])
                parsed_line[7] = dep_graph_pred[sentence_idx].nodes[word_idx]['rel'].lower()
                parsed_line = '\t'.join(parsed_line)
                lines.append(parsed_line)

    with open(os.path.join(args.experiment_dir, data_split + '_pred.conll'), 'w') as f:
        for item in lines:
            f.write("%s" % item)


def generate_exp_name(args):
    return f'./results/' \
           f'ds={args.ds_name}_' \
           f'epochs={args.epochs}_' \
           f'lr={args.lr}_' \
           f'seed={args.seed}_' \
           f'extEmb={args.ext_emb is not None}_' \
           f'wDim={args.w_emb_dim}_' \
           f'pDim={args.pos_emb_dim}_' \
           f'lstmDim={args.lstm_hid_dim}_' \
           f'mlpDim={args.mlp_hid_dim}_' \
           f'lstmN={args.n_lstm_layers}_' \
           f'date={datetime.date.today().strftime("%m_%d_%Y")}'
