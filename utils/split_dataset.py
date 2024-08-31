import numpy as np
from sklearn.model_selection import train_test_split
import os
from functools import reduce
from tqdm import tqdm
import argparse


def split_dataset(dataset_dir, max_len, dataset_save_dir, test_size=0.3, truncation='post'):
    """
    将带标签(竖排列)的多文件数据分成train test dev集，数据集中一行一样本，文本在前，标签在后，中间为Tab位隔开，[CLS], [SEP]已加入

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='dataset_dir')
    parser.add_argument('-max_len', help='max_len, default=256', default=200)
    parser.add_argument('-t', help='dataset_save_dir, default="./', default='./')
    parser.add_argument('-size', help='test_size, default=0.3', default=0.3)
    parser.add_argument('-truncation', help='"post" or "pre", default="post"', default='post')
    arg = parser.parse_args()
    split_dataset(arg.s, int(arg.max_len), arg.t, float(arg.size), arg.truncation)