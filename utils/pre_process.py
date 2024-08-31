import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
# from pytorch_transformers import *
import pandas as pd
from functools import reduce
import numpy as np



def generate_data_loader(data_path, batch_size, debug=False, debug_lines=10):
    df = pd.read_csv(data_path, delimiter='\t', header=None, names=['source', 'sentence', 'label'])
    df = df[1:]
    sentences = df.sentence.values
    sentences = ['[CLS] ' + sentence + ' [SEP]' for sentence in sentences]
    # 去掉label结尾的空格
    df['label'] = df['label'].apply(lambda x: x[:-1] if x[-1]==' ' else x)
    df['label'] = df['label'].apply(lambda x: x.split(' '))
    labels = df.label.values
    label_set = [list(set(i)) for i in labels]
    label_set = list(set(reduce(lambda x, y: x + y, label_set)))
    df['label_ids'] = df['label'].apply(lambda x: [label_set.index(i) for i in x])
    labels = df.label_ids.values

    # 减少数据，调试专用
    if debug:
        sentences = sentences[:debug_lines]
        labels = labels[:debug_lines]

    tokenizer = BertTokenizer.from_pretrained('pretrained_weights', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # print('Tokenize the first sentence:')
    # print(tokenized_texts[0])

    MAX_LEN = 128
    # pad input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

    tags = pad_sequences([t for t in labels], maxlen=MAX_LEN, padding='post', dtype='long', truncating='post')
    attention_masks = [[float(i>0) for i in j] for j in input_ids]

    input_ids = torch.tensor(input_ids)
    tags = torch.tensor(tags)
    attention_masks = torch.tensor(attention_masks)

    dataset = TensorDataset(input_ids, attention_masks, tags)
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size), label_set


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)