import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
# from pytorch_transformers import *
import pandas as pd
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
from tqdm import tqdm


def read_label_set(path):
    with open(path, mode='r', encoding='utf-8') as f:
        label = f.read()
    label_set = label.split('\n')
    label_set = [i for i in label_set if i]
    return label_set


if __name__ == '__main__':
    label_set = read_label_set(path='../script_config/label_set')
else:
    label_set = read_label_set(path='./script_config/label_set')


def split_dataset(dataset_dir, max_len, dataset_save_dir, test_size=0.3, truncation='post'):
    """

    :param dataset_dir:
    :param max_len:
    :param dataset_save_dir:
    :param test_size:
    :param truncation: 'pre'，若超过max_len，则从开头起舍弃，'post'则从结尾起舍弃
    :return:
    """
    store_x, store_y = [], []
    for fp in tqdm(os.listdir(dataset_dir)):
        filename = os.path.join(dataset_dir, fp)
        if os.path.exists(filename):
            with open(os.path.join(dataset_dir, fp), mode='r', encoding='utf-8') as f:
                data = f.read().split('\n')
            data = [i.split(' ') for i in data if i != '\n' and i != '']
            sentence, tags = [], []
            for i in data:
                sentence.append(i[0])
                tags.append(i[1])
            if len(sentence) > max_len - 2:
                if truncation == 'post':
                    sentence = sentence[:max_len-2]
                    tags = tags[:max_len-2]
                elif truncation == 'pre':
                    sentence = sentence[-max_len+2:]
                    tags = tags[-max_len+2:]
                else:
                    raise Exception('truncation is "pre" or "post"!')
            store_x.append(['[CLS]'] + sentence + ['[SEP]'])
            store_y.append(tags)
    print('===>> File being written.............')
    x_train, x_test, y_train, y_test = train_test_split(store_x, store_y, test_size=test_size, random_state=0)
    x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=0.3, random_state=0)
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)
    save_dataset(x_train, y_train, os.path.join(dataset_save_dir, 'train.tsv'))
    save_dataset(x_test, y_test, os.path.join(dataset_save_dir, 'test.tsv'))
    save_dataset(x_dev, y_dev, os.path.join(dataset_save_dir, 'dev.tsv'))
    print('===========DataSet Split Successfully!============')


def save_dataset(data_x, data_y, save_path):
    with open(save_path, mode='w', encoding='utf-8') as writer:
        print('>>>> %s being written........' % save_path)
        for i in tqdm(zip(data_x, data_y)):
            tmp = i[0] + ['\t'] + i[1]
            tmp = reduce(lambda x, y: x + ' ' + y, tmp)
            writer.write(tmp + '\n')


def sentence2ids(tokenizer, sentence, device, max_len=200, truncation='post'):
    """
    将输入的文本序列转换成id list, 为mask list，以用于喂入模型进行推理
    :param tokenizer: BertTokenizer
    :param sentence: 输入文本序列（无需用空格分开）
    :param device: cpu or gpu
    :param max_len: 序列最大长度
    :param truncation: 'post' or 'pre', 截断序列的方向，post从结尾向前，pre从开头向后
    :return: torch.tensor(ids), torch.tensor(mask)，非batch_size，是单条数据
    """
    sentence = tokenizer.tokenize(sentence)
    if len(sentence) > max_len - 2:
        if truncation == 'post':
            sentence = sentence[:max_len - 2]
        elif truncation == 'pre':
            sentence = sentence[-max_len + 2:]
        else:
            raise Exception('truncation is "pre" or "post"!')
    # sentence是单条数据，tokenized_sentence是单条数据，因此后续要加一个维度
    tokenized_sentence = ['[CLS]'] + sentence + ['[SEP]']
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in [tokenized_sentence]],
                              maxlen=max_len, dtype='long', truncating='post', padding='post')
    attention_masks = [[float(i>0) for i in j] for j in input_ids]

    input_ids = torch.tensor(input_ids).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)
    return input_ids, attention_masks


def generate_data_loader(tokenizer, data_path, label_set, batch_size, debug=False, debug_lines=10, max_len=200):
    if not os.path.exists(data_path):
        raise Exception('%s NotFound!!!!!' % data_path)
    df = pd.read_csv(data_path, delimiter='\t', header=None, names=['sentence', 'label'])
    # 去掉sentence结尾的空格
    df['sentence'] = df['sentence'].apply(lambda x: x[:-1])
    sentences = df.sentence.values
    sentences = [sentence for sentence in sentences]
    # 去掉label开头的空格
    df['label'] = df['label'].apply(lambda x: x[1:])
    df['label'] = df['label'].apply(lambda x: x.split(' '))
    # labels = df.label.values
    # label_set = [list(set(i)) for i in labels]
    # label_set = list(set(reduce(lambda x, y: x + y, label_set)))
    df['label_ids'] = df['label'].apply(lambda x: [label_set.index(i) for i in x])
    labels = df.label_ids.values

    # 减少数据，调试专用
    if debug:
        sentences = sentences[:debug_lines]
        labels = labels[:debug_lines]

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # pad input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype='long', truncating='post', padding='post')

    tags = pad_sequences([t for t in labels], maxlen=max_len, padding='post', dtype='long', truncating='post')
    attention_masks = [[float(i>0) for i in j] for j in input_ids]

    input_ids = torch.tensor(input_ids)
    tags = torch.tensor(tags)
    attention_masks = torch.tensor(attention_masks)

    dataset = TensorDataset(input_ids, attention_masks, tags)
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='dataset_dir')
    parser.add_argument('-max_len', help='max_len, default=256', default=200)
    parser.add_argument('-t', help='dataset_save_dir, default="./', default='./')
    parser.add_argument('-size', help='test_size, default=0.3', default=0.3)
    parser.add_argument('-truncation', help='"post" or "pre", default="post"', default='post')
    arg = parser.parse_args()
    split_dataset(arg.s, int(arg.max_len), arg.t, float(arg.size), arg.truncation)

    # dataset_dir = '../../../dataset/理赔单证/dataset/hospital_name'
    # split_dataset(dataset_dir, max_len=200, dataset_save_dir='../../../dataset/hospital_name_dataset', test_size=0.3, truncation='post')

    # data_path = '../../../dataset/hospital_name_dataset/train.tsv'
    # tokenizer = BertTokenizer.from_pretrained('../pretrained_weights', do_lower_case=True)
    # generate_data_loader(tokenizer, data_path, batch_size=16, debug=True)
