import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from utils import normalize
from decoder_eisner import parse_proj


class BISTParser(nn.Module):
    def __init__(self, w_emb_dim, pos_emb_dim, lstm_hid_dim, mlp_hid_dim, n_lstm_l,
                 n_arc_relations, w_i_counter, w2i, p2i, alpha, device,
                 ex_w_vec=None, ext_emb_w2i=None):
        super(BISTParser, self).__init__()

        self.w_i_counter = w_i_counter
        self.w2i = w2i
        self.p2i = p2i

        self.alpha = alpha
        self.device = device

        # embedding layers initialization
        self.word_embedding = nn.Embedding(len(w2i), w_emb_dim)
        self.pos_embedding = nn.Embedding(len(p2i), pos_emb_dim)

        self.ex_emb_flag = False
        if ex_w_vec is not None and ext_emb_w2i is not None:
            # load external word embeddings
            self.ex_emb_flag = True
            self.ex_emb_w2i = ext_emb_w2i
            self.ex_word_emb = nn.Embedding.from_pretrained(ex_w_vec, freeze=False)

        # LSTM dimensions
        input_dim = w_emb_dim + pos_emb_dim
        if self.ex_emb_flag:
            input_dim += ex_w_vec.size(-1)

        # bidirectional LSTM initialization
        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=lstm_hid_dim,
                               num_layers=n_lstm_l,
                               bidirectional=True,
                               batch_first=True)

        # arc scorer initialization
        self.hid_arc_h = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_arc_m = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_arc_bias = nn.Parameter(torch.empty((1, mlp_hid_dim)))
        BISTParser.param_init(self.hid_arc_bias)

        self.slp_out_arc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(mlp_hid_dim, 1, bias=False)
        )

        # arc relations MLP initialization
        self.hid_rel_h = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_rel_m = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_rel_bias = nn.Parameter(torch.empty((1, mlp_hid_dim)))
        BISTParser.param_init(self.hid_rel_bias)

        self.slp_out_rel = nn.Sequential(
            nn.Tanh(),
            nn.Linear(mlp_hid_dim, n_arc_relations)
        )

        # initialize model weights
        for name, module in self.named_children():
            if name == 'ex_word_emb':
                continue
            else:
                BISTParser.modules_init(module)

    def forward(self, sentence, pos_tags, gold_tree=None, word_dropout=False):

        n_words = len(sentence)

        s_w_i = torch.tensor([self.w2i.get(normalize(w[0]), 0) for w in sentence]).to(self.device).unsqueeze(0)
        s_pos_i = torch.tensor([self.p2i.get(pos_tag[0], 0) for pos_tag in pos_tags]).to(self.device).unsqueeze(0)

        # word dropout
        if word_dropout:
            unk_probs = torch.tensor([1 - (self.alpha / (self.w_i_counter[w_i.item()] + self.alpha))
                                      for w_i in s_w_i.squeeze()]).to(self.device)
            do_w_ber_sample = Bernoulli(probs=unk_probs).sample().int()
            do_s_w_i = s_w_i * do_w_ber_sample
            w_emb_tensor = self.word_embedding(do_s_w_i)

            do_ext_emb_probs = torch.abs(do_w_ber_sample - 1) * 0.5
            do_ext_emb_ber_sample = Bernoulli(probs=do_ext_emb_probs).sample().int()
        else:
            w_emb_tensor = self.word_embedding(s_w_i)

        pos_emb_tensor = self.pos_embedding(s_pos_i)

        # embeddings concatenation
        if self.ex_emb_flag:
            ex_s_w_i = torch.tensor(
                [self.ex_emb_w2i.get(w[0], self.ex_emb_w2i.get(normalize(w[0]), 0)) for w in sentence]).to(
                self.device).unsqueeze(0)
            if word_dropout:
                ex_do_s_w_i = ex_s_w_i * (do_ext_emb_ber_sample + do_w_ber_sample)
                ex_word_em_tensor = self.ex_word_emb(ex_do_s_w_i)
            else:
                ex_word_em_tensor = self.ex_word_emb(ex_s_w_i)
            input_vectors = torch.cat((w_emb_tensor, pos_emb_tensor, ex_word_em_tensor), dim=-1)
        else:
            input_vectors = torch.cat((w_emb_tensor, pos_emb_tensor), dim=-1)

        hidden_vectors, _ = self.encoder(input_vectors)

        # score all possible arcs
        arc_h_scores = self.hid_arc_h(hidden_vectors)
        arc_m_scores = self.hid_arc_m(hidden_vectors)

        idx_ls = [idx for idx in range(n_words)]
        arc_scores = self.slp_out_arc(arc_h_scores[0, np.repeat(idx_ls, n_words), :]
                                      + arc_m_scores[0, idx_ls * n_words, :]
                                      + self.hid_arc_bias)
        arc_scores = arc_scores.view(n_words, n_words)

        # get the highest scoring dependency tree
        pred_tree = parse_proj(arc_scores.detach().cpu().numpy(), gold_tree)

        # score all possible relations
        heads = gold_tree[1:] if gold_tree is not None else pred_tree[1:]
        rel_h_scores = self.hid_rel_h(hidden_vectors)
        rel_m_scores = self.hid_rel_m(hidden_vectors)
        rel_scores = self.slp_out_rel(rel_h_scores[0, heads, :]
                                      + rel_m_scores[0, idx_ls[1:], :]
                                      + self.hid_rel_bias)

        if gold_tree is not None:  # during training
            return arc_scores, rel_scores, pred_tree

        return pred_tree, torch.argmax(rel_scores, dim=-1)  # during inference

    @staticmethod
    def modules_init(m):
        if isinstance(m, nn.Embedding):
            emb_bound = math.sqrt(3. / m.embedding_dim)
            nn.init.uniform_(m.weight, -emb_bound, emb_bound)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if 'bias' in name:
                    h_dim = p.shape[-1] // 4
                    nn.init.constant_(p[: h_dim], 0.)
                    nn.init.constant_(p[h_dim: 2 * h_dim], 0.5)  # forget gate bias initialization
                    nn.init.constant_(p[2 * h_dim:], 0.)
                else:
                    nn.init.xavier_uniform_(p)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Sequential):
            nn.init.xavier_uniform_(m[1].weight)
            if m[1].bias is not None:
                BISTParser.param_init(m[1].bias)

    @staticmethod
    def param_init(p):
        bound = math.sqrt(3. / p.shape[-1])
        nn.init.uniform_(p, -bound, bound)