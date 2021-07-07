import numpy as np
from torch.utils.data.dataset import Dataset
from nltk.parse import DependencyGraph

from constants import *


class DependencyDataSet(Dataset):
    def __init__(self, w2i, r2i, p2i, data_path):
        super().__init__()

        self.data_path = data_path
        self.w2i = w2i
        self.r2i = r2i
        self.p2i = p2i

        self.i2r = list(self.r2i.keys())

        # read the data
        self.sentences, self.pos_tags, self.gold_trees, self.relations, self.dep_graph_gold = self._data_reader()

        # convert to dataset format
        self.sentences_dataset = self._convert_to_dataset()

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        sentence, pos_tags, gold_tree, relations = self.sentences_dataset[index]
        return sentence, pos_tags, gold_tree, relations

    def _data_reader(self):
        with open(self.data_path, 'r', encoding="utf8") as fh:

            sentences = []
            pos_tags = []
            gold_trees = []
            relations = []
            dep_graph_gold = []

            sentence, sentence_pos, gold_tree, sentence_rels, dep_graph_string = [ROOT_SYMBOL], \
                                                                                 [ROOT_SYMBOL], \
                                                                                 [-1], \
                                                                                 [self.r2i[ROOT_SYMBOL]], \
                                                                                 ''

            for line in fh:

                if line.startswith('#'):
                    continue

                if line == '\n' and len(sentence) > 1:

                    sentences.append(sentence)
                    pos_tags.append(sentence_pos)
                    gold_trees.append(np.array(gold_tree))
                    relations.append(np.array(sentence_rels))
                    dep_graph_gold.append(DependencyGraph(dep_graph_string))

                    sentence = [ROOT_SYMBOL]
                    sentence_pos = [ROOT_SYMBOL]
                    gold_tree = [-1]
                    sentence_rels = [self.r2i[ROOT_SYMBOL]]
                    dep_graph_string = ''

                else:
                    parsed_line = line.strip().split('\t')

                    if '-' in parsed_line[0] or '.' in parsed_line[0]:
                        continue

                    word, pos_tag, head, arc_rel = parsed_line[1], parsed_line[3], int(parsed_line[6]), parsed_line[7]
                    if len(word.split(' ')) > 1:
                        word = '_'.join(word.split(' '))

                    sentence.append(word)
                    sentence_pos.append(pos_tag)
                    gold_tree.append(head)
                    sentence_rels.append(self.r2i.get(arc_rel, 0))

                    dep_graph_string += word + "\t" + pos_tag + "\t" + str(head) + "\t" + str.upper(arc_rel) + "\n"

        return sentences, pos_tags, gold_trees, relations, dep_graph_gold

    def _convert_to_dataset(self):
        return {i: sample_tuple for i, sample_tuple in
                enumerate(zip(self.sentences, self.pos_tags, self.gold_trees, self.relations))}