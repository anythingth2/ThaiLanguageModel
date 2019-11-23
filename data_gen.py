# %%
import numpy as np
from keras.utils import to_categorical
import ujson
from pathlib import Path
from pythainlp.tokenize import word_tokenize
import re
import pandas as pd
# %%
with open('character_copus.txt', 'r', encoding='utf-8') as f:
    CHAR_CORPUS = f.read().replace('\n', '')
    CHAR_CORPUS = ''.join(sorted(list(set(CHAR_CORPUS))))
CHAR_DICT = {char: i for i, char in enumerate(CHAR_CORPUS)}
CHAR_DICT['$'] = len(CHAR_DICT)
NUM_VOCAB = len(CHAR_DICT)


def tokenize(text):
    return [CHAR_DICT[c] for c in text]


# %%
def random_blank(p=0.01):

    def func(text):
        idxs = np.random.choice(np.arange(len(text)),
                                int(p*len(text)), replace=False)
        text = list(text)
        for i in idxs:
            text[i] = '$'
        text = ''.join(text)
        return text
    return func


def sentence_generator(input_dir, length, batch_size, adversarial=None):
    input_paths = list(input_dir.glob('**/wiki_*.json'))

    scrap_sentences = []
    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            wiki_jsons = ujson.load(f)
        texts = [blog['text'] for blog in wiki_jsons]
        full_text = ''.join(texts)
        full_text = full_text.replace('\n', ' ').replace('  ', ' ')
        full_text = re.sub(f'[^{CHAR_CORPUS}]', '', full_text)
        full_text = full_text.replace(' ', '')

        words = word_tokenize(full_text, engine='mm')

        sentences = [] + scrap_sentences

        text = ''
        while len(words) > 1:
            if len(text + words[0]) < length-2:
                text += words.pop(0)
            else:
                sentences.append(text)
                text = ''

        for i in range(0, len(sentences), batch_size):
            batch_x = sentences[i:i+batch_size]
            batch_y = [adversarial(s) for s in batch_x]

            batch_x = np.array([tokenize(s+' '*(length - len(s)))
                                for s in batch_x])
            batch_y = np.array([tokenize(s+' '*(length - len(s)))
                                for s in batch_y])

            yield np.array([to_categorical(s, NUM_VOCAB) for s in batch_x]), np.array([to_categorical(s, NUM_VOCAB) for s in batch_y])
        scrap_sentences = sentences[i:]


# %%
# length = 160
# batch_size = 32
# input_dir = Path('dataset/thwiki')
# gen = sentence_generator(input_dir, length, batch_size,
#                          adversarial=random_space(0.02))
# # %%
# X,Y = next(gen)
