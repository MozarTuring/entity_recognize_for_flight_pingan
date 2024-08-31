import numpy as np
from sklearn.model_selection import train_test_split
import os
from functools import reduce
from tqdm import tqdm
import argparse


def merge_split_dataset(dataset_dir, dataset_save_dir, test_size=0.1, dev_size=0.1):
    """
    将带标签(竖排列)的多文件数据分成train test dev集，数据集中每行一个字及其标签，中间为空格隔开。所有样本均在第一列，所有标签在第二列
    样本与样本间用空行隔开。

    :param dataset_dir:
    :param dataset_save_dir:
    :param test_size:
    :param dev_size:
    :return:
    """
    total_filepath = []
    total_filename = []
    for fp in os.listdir(dataset_dir):
        total_filename.append(fp)
        total_filepath.append(os.path.join(dataset_dir, fp))
    train_filepath, test_filepath, train_filename, test_filename = train_test_split(total_filepath,
                                                                                    total_filename,
                                                                                    test_size=test_size+dev_size)
    test_filepath, dev_filepath, test_filename, dev_filename = train_test_split(test_filepath,
                                                                                test_filename,
                                                                                test_size=dev_size/(test_size+dev_size))
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    train_path = os.path.join(dataset_save_dir, 'train.txt')
    test_path = os.path.join(dataset_save_dir, 'test.txt')
    dev_path = os.path.join(dataset_save_dir, 'dev.txt')

    with open(train_path, mode='w', encoding='utf-8') as writer:
        for fp in tqdm(train_filepath, desc='Writing train set...'):
            with open(fp, mode='r', encoding='utf-8') as f:
                writer.write(f.read() + '\n')

    with open(test_path, mode='w', encoding='utf-8') as writer:
        for fp in tqdm(test_filepath, desc='Writing test set...'):
            with open(fp, mode='r', encoding='utf-8') as f:
                writer.write(f.read() + '\n')

    with open(dev_path, mode='w', encoding='utf-8') as writer:
        for fp in tqdm(dev_filepath, desc='Writing dev set...'):
            with open(fp, mode='r', encoding='utf-8') as f:
                writer.write(f.read() + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', help='dataset_dir')

    parser.add_argument('--target_path', help='dataset_save_dir, default="./', default='./')

    parser.add_argument('--test_size', help='test_size, default=0.1', default=0.1, type=float)
    parser.add_argument('--dev_size', help='dev_size, default=0.1', default=0.1, type=float)


    arg = parser.parse_args()
    merge_split_dataset(arg.source_path, arg.target_path, arg.test_size, arg.dev_size)