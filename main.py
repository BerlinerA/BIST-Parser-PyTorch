import argparse
import logging
import os
import sys
from operator import itemgetter
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from nltk.parse import DependencyEvaluator

from utils import *
from constants import *
from data import *
from model import BISTParser


def main():
    parser = argparse.ArgumentParser('description=Graph based dependency parser using BiLSTM feature extractors')
    parser.add_argument('--train_path', type=str, default='../data/ptb/train.conll',
                        help='path to an annotated CONLL train file')
    parser.add_argument('--dev_path', type=str, default='../data/ptb/dev.conll',
                        help='path to an annotated CONLL development file')
    parser.add_argument('--test_path', type=str, default='../data/ptb/test.conll',
                        help='path to an annotated CONLL test file')
    parser.add_argument('--ds_name', type=str, default='ptb',
                        help='dataset name')
    parser.add_argument('--model_path',
                        help='path to a pretrained model', type=str, default=None)
    parser.add_argument('--ext_emb',
                        help='path to an external word embeddings file', default=None)
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=2.5e-1,
                        help='word dropout parameter (default: 2.5e-1)')
    parser.add_argument('--w_emb_dim', type=int, default=100,
                        help='word embedding dimension (default: 100)')
    parser.add_argument('--pos_emb_dim', type=int, default=25,
                        help='POS tag embedding dimension (default: 25)')
    parser.add_argument('--lstm_hid_dim', type=int, default=125,
                        help='LSTM hidden dimension (default: 125)')
    parser.add_argument('--mlp_hid_dim', type=int, default=100,
                        help='MLP hidden dimension (default: 100)')
    parser.add_argument('--n_lstm_layers', type=int, default=2,
                        help='number of LSTM layers (default: 2)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=2000,
                        help='how many samples to wait before logging training status (default: 2000)')
    parser.add_argument('--do_eval', action='store_true', default=False,
                        help='evaluate a given model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # define experiment path
    if args.do_eval:
        assert 'in evaluation mode, a model must be provided.', \
            args.model_path is not None and os.path.isfile(args.model_path)
        args.experiment_dir = os.path.dirname(args.model_path)
    else:
        args.experiment_dir = generate_exp_name(args)

        if not os.path.exists(args.experiment_dir):
            os.makedirs(args.experiment_dir)

    # initialize logging object
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(args.experiment_dir, 'eval.log' if args.do_eval else 'train.log'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # save the experiment parameters
    save_obj(vars(args), args.experiment_dir, 'config')

    logger.info(f'Experiment Parameters - \n{vars(args)}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    # create vocabulary
    words_count, w2i, r2i, p2i = vocab(args.train_path)
    save_obj([words_count, w2i, r2i, p2i], args.experiment_dir, 'vocab')
    logger.info(f'Vocab statistics: words - {len(w2i)} | relations - {len(r2i)} | POS tags - {len(p2i)}')

    # load external word embeddings
    if args.ext_emb is not None:
        ex_w2i, ex_word_vectors = load_pretrained_word_embed(args.ext_emb, w2i)
    else:
        ex_w2i, ex_word_vectors = None, None

    # set up training, validation and test data
    train_set = DependencyDataSet(words_count, w2i, r2i, p2i, args.train_path)
    dev_set = DependencyDataSet(words_count, w2i, r2i, p2i, args.dev_path)
    test_set = DependencyDataSet(words_count, w2i, r2i, p2i, args.test_path)

    train_gen = DataLoader(train_set, shuffle=True)
    dev_gen = DataLoader(dev_set, shuffle=False)
    test_gen = DataLoader(test_set, shuffle=False)

    # parser initialization
    parser = BISTParser(w_emb_dim=args.w_emb_dim,
                        pos_emb_dim=args.pos_emb_dim,
                        lstm_hid_dim=args.lstm_hid_dim,
                        mlp_hid_dim=args.mlp_hid_dim,
                        n_lstm_l=args.n_lstm_layers,
                        n_arc_relations=train_set.n_unique_rel,
                        w_i_counter=train_set.w_i_counts,
                        w2i=train_set.w2i,
                        p2i=train_set.p2i,
                        alpha=args.alpha,
                        device=args.device,
                        ext_emb_w2i=ex_w2i,
                        ex_w_vec=ex_word_vectors).to(args.device)

    if args.model_path is not None:
        # load pretrained model weights
        best_parser_path = args.model_path
        parser.load_state_dict(torch.load(best_parser_path, map_location=args.device))
        parser.to(args.device)
    else:
        best_parser_path = None

    optimizer = Adam(parser.parameters(), lr=args.lr)

    if not args.do_eval:

        train_stats = defaultdict(list)
        best_uas = 0.
        for epoch in range(1, args.epochs + 1):

            logger.info('-----------+-----------+-----------+-----------+-----------')
            logger.info(f'Train epoch: {epoch}')

            train(args, parser, train_gen, optimizer, logger)
            dev_eval_dict = evaluate(args, parser, dev_gen, logger, data_split='dev')

            train_stats['dev_uas'].append(dev_eval_dict['uas'])
            train_stats['dev_las'].append(dev_eval_dict['las'])

            if dev_eval_dict['uas'] > best_uas:
                best_uas = dev_eval_dict['uas']
                best_parser_path = os.path.join(args.experiment_dir, 'parser.pt')
                torch.save(parser.state_dict(), best_parser_path)
                logger.info("---UAS was improved. The current parser was saved.---")

        save_obj(train_stats, args.experiment_dir, 'train_stats')

    # load & test the best model
    parser.load_state_dict(torch.load(best_parser_path, map_location=args.device))
    evaluate(args, parser, test_gen, logger, data_split='test')


def train(args, parser, train_gen, optimizer, logger):
    parser.train()

    ds_size = len(train_gen)
    for sample_idx, (sentence, pos_tags, gold_tree, gold_rels) in enumerate(train_gen):

        gold_tree = gold_tree.squeeze().numpy()
        gold_rels = gold_rels.squeeze().numpy()

        # parse
        arc_scores, rel_scores, aug_pred_tree = parser(sentence,
                                                       pos_tags,
                                                       gold_tree=gold_tree,
                                                       word_dropout=True)

        # loss augmented inference
        loss_ls = [(arc_scores[p][i] - arc_scores[g][i])
                   for i, (p, g) in enumerate(zip(aug_pred_tree, gold_tree)) if p != g]

        for idx, gold_rel in enumerate(gold_rels[1:]):
            wrong_rel = max(((rel, score)
                             for rel, score in enumerate(rel_scores[idx]) if rel != gold_rel), key=itemgetter(1))[0]
            if rel_scores[idx][gold_rel] < rel_scores[idx][wrong_rel] + 1:
                loss_ls.append(rel_scores[idx][wrong_rel] - rel_scores[idx][gold_rel])

        if len(loss_ls) > 0:
            loss = sum(loss_ls)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        if sample_idx % args.log_interval == 0:
            logger.info('[{}/{} ({:.0f}%)]'.format(sample_idx, ds_size, 100. * sample_idx / ds_size))


def evaluate(args, parser, test_gen, logger, data_split):

    parser.eval()

    dep_graph_pred = []
    with torch.no_grad():
        for sample_idx, (sentence, pos_tags, gold_tree, gold_rel) in enumerate(test_gen):

            # parse
            pred_tree, pred_rels = parser(sentence, pos_tags)

            dep_graph_pred.append(to_dependency_graph(test_gen.dataset.sentences[sample_idx][1:],
                                                      [p[0] for p in pos_tags[1:]],
                                                      pred_tree[1:],
                                                      [test_gen.dataset.i2r[rel_idx] for rel_idx in pred_rels]))

    de = DependencyEvaluator(dep_graph_pred, test_gen.dataset.dep_graph_gold)
    las, uas = de.eval()

    eval_dict = {'las': las,
                 'uas': uas}

    log_results(eval_dict, logger, data_split)
    output_file = os.path.join(args.experiment_dir, data_split + '_results.txt')
    write_results_file(eval_dict, output_file)
    write_predicted_labels(args, dep_graph_pred, data_split)

    return eval_dict


if __name__ == '__main__':
    main()