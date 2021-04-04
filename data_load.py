import random, os
import numpy as np
import torch
from torch.utils import data
from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

class PretrainingDataset(data.Dataset):
    def __init__(self, path):
        self.sents = []
        for file in os.listdir(path):
            if file[-4:] == ".txt":
                with open(path+file, 'r') as f:
                    for line in f:
                        self.sents.append(tokenizer.tokenize(line.strip()))

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        tokens = self.sents[idx]  # words, tags: string list

        # Reused as attention mask for generator
        mask = [0 if random.randint(0, 100) < 15 else 1 for _ in tokens]

        tokens_with_mask = []
        for m, t in zip(mask, tokens):
            if not m:
                tokens_with_mask.append('[MASK]')
            else:
                tokens_with_mask.append(t)
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_with_mask = tokenizer.convert_tokens_to_ids(tokens_with_mask)
        assert len(tokens) == len(tokens_with_mask) == len(mask)
        return tokens, tokens_with_mask, mask, len(tokens)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # pads each sequence with 0 values
    tokens = f(0, maxlen)
    tokens_with_mask = f(1, maxlen)
    generator_masks = f(2, maxlen)

    f = lambda x, seqlen: [[1] * len(sample[x]) + [0] * (seqlen - len(sample[x])) for sample in batch]  # pads each sequence with 0 values
    discriminator_masks = f(0, maxlen)

    f = torch.LongTensor
    return f(tokens), f(tokens_with_mask), f(generator_masks), f(discriminator_masks)

#
# class Document:
#     def __init__(self, document_name, document_dir, bidirectional=True, e2e=False):
#         self.document_name = document_name
#         self.bidirectional = bidirectional
#         self.e2e = e2e
#         self.document_dir = document_dir
#         self.document = None
#         self.lines = []
#         self.assertions = []
#         self.concepts = []
#         self.relations = []
#         self.tokens = []
#         self.get_document()
#         self.document_to_tokens()
#         self.get_concept()
#         self.get_relations()
#
#     def get_document(self):
#         file = open(self.document_dir + f'/txt/{self.document_name}.txt', 'r')
#         for line in file:
#             self.lines.append(line.strip())
#
#     def __str__(self):
#         string = ''
#         for token_i in range(len(self.tokens)):
#             if str(self.tokens[token_i]) != '':
#                 string = string + str(self.tokens[token_i]) + ' '
#         return string
#
#     def document_to_tokens(self):
#         for i, line in enumerate(self.lines):
#             words = []
#             line = re.sub(r'[ ]{2,10}', ' ', line)
#             for word in line.split(' '):
#                 words.append(
#                     Token(word)
#                 )
#             self.lines[i] = words
#
#
# class Token:
#     def __init__(self, original, tag=None):
#         self.original_text = original
#         if not tag:
#             self.concept = 'O'
#         else:
#             self.concept = tag
#         self.relations = []
#         self.prediction_text = ''
#         self.prediction_result = None
#         tokens = tokenizer.tokenize(self.original_text) if self.original_text not in ("[CLS]", "[SEP]") else [self.original_text]
#         self.tokens = tokenizer.convert_tokens_to_ids(tokens)
#
#     def __repr__(self):
#         return '(%s, %s, %s)' % (
#             repr(self.original_text), self.prediction_text, self.prediction_result)
#
#     def __str__(self):
#         return self.original_text
