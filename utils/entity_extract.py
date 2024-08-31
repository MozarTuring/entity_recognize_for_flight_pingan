import os
import logging, sys, argparse
from tqdm import tqdm


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def entity_extract(result_path, label_set, example_len=None, location=False):
    """
    从类似train.txt格式的文本中抽取实体。
    :param result_path:
    :param label_set: 要取实体的标签集，如{'Admission_Date', 'Admission_Number', 'Department',
                                         'Discharge_Date', 'Hospital_Name'}
    :param example_len: 取一个样本的长度。
    :return:
    """
    result = []
    with open(os.path.join(result_path), "r", encoding='utf-8') as f:
        tmp = []
        for line in tqdm(f, desc='Processing origin...'):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tmp:
                    result.append(entity_extract_from_one_example(tmp, label_set, example_len))
                    tmp = []
            else:
                tmp.append(line)
    if location:
        return result
    else:
        no_location_result = []
        for each_dict in tqdm(result, desc='Removing locations...'):
            tmp = {}
            for key in each_dict:
                tmp[key] = list(set(each_dict[key][0]))
            no_location_result.append(tmp)
        return no_location_result

def entity_extract_from_one_example(one_example, label_set, example_len=None):
    """

    :param one_example: 一个完整样本，按竖排列的（example_len x 3）的矩阵。第一列为单字，第二列为空格，第三列为标签
    :param label_set: 要取实体的标签集，如{'Admission_Date', 'Admission_Number', 'Department',
                                         'Discharge_Date', 'Hospital_Name'}
    :param example_len: 取一个样本的长度。
    :return:
    """
    tag_seq, original_seq = [], []
    for line in one_example:
        s = line.strip().split()  # 每行结尾有换行符
        tag_seq.append(s[1])
        original_seq.append(s[0] if s[0] else ' ')   # 处理原文中为空格的情况
    assert len(tag_seq) == len(original_seq), '标签长度不等于原文长度！'
    if not example_len:
        assert isinstance(example_len, int), 'example_len 不为 int !'
        if example_len < len(tag_seq):
            tag_seq = tag_seq[:example_len]
            original_seq = original_seq[:example_len]
    entity_dict = {}
    for tag_name in label_set:
        entity = get_tag_entity(tag_seq, original_seq, tag_name)
        entity_dict[tag_name] = entity
    return entity_dict


def get_entity(tag_id_seq, original_seq, label_set, tag_set):
    """
    获得tag_set中提及的实体
    :param tag_id_seq: id_list, 如[1,2,3,0,0,0,0]
    :param original_seq: 原始序列
    :param label_set: 如{'Admission_Date', 'Admission_Number', 'Department', 'Discharge_Date', 'Hospital_Name'}
    :param tag_set: list, 标签集合
    :return: {'Department': ([], []),
              'Admission_Number': ([], []),
              'Hospital_Name': (['东山区第二医院'], ['25,32']),
              'Discharge_Date': ([':'], ['132,133']),
              'Admission_Date': ([], []),
              'Sentence': ""}
    """
    entity_dict = {}
    tag_id_seq = [tag_set[i] for i in tag_id_seq]
    for tag_name in label_set:
        entity = get_tag_entity(tag_id_seq, original_seq, tag_name)
        entity_dict[tag_name] = entity
        entity_dict['Sentence'] = original_seq
    return entity_dict


def get_tag_entity(tag_seq, char_seq, tag_name):
    """
    input tag sequence ,tag2label and char sequence ,output the entity and the index in the char sequence.
    :param tag_seq:
    :param char_seq:
    :param tag2label:
    :return:
    """
    # for tag in tag2label.keys():
    #     if tag != "O":
    #         tag_name = tag[2:]
    length = len(char_seq)
    entity = []
    entity_idx = []
    start_idx = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-' + tag_name:
            if 'per' in locals().keys():
                entity.append(per)
                entity_idx.append(str(start_idx) + ',' + str(start_idx + len(per)))
                del per
            per = char
            start_idx = i  # the start idx of one entity
            if i + 1 == length:
                entity.append(per)
                entity_idx.append(str(start_idx) + ',' + str(length))
        if tag == 'I-' + tag_name:
            if 'per' not in locals().keys():
                continue
            per += char
            if i + 1 == length:
                entity.append(per)
                entity_idx.append(str(start_idx) + ',' + str(length))
        if tag not in ['I-' + tag_name, 'B-' + tag_name]:
            if 'per' in locals().keys():
                entity.append(per)
                entity_idx.append(str(start_idx) + ',' + str(start_idx + len(per)))
                del per
            continue
    return entity, entity_idx


# def get_tag_entity(tag_seq, char_seq, tag_name):
#     # for tag in tag2label.keys():
#     #     if tag != "O":
#     #         tag_name = tag[2:]
#     length = len(char_seq)
#     entity = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-'+tag_name:
#             if 'per' in locals().keys():
#                 entity.append(per)
#                 del per
#             per = char
#             if i+1 == length:
#                 entity.append(per)
#         if tag == 'I-'+tag_name:
#             try:
#                 per += char
#                 if i+1 == length:
#                     entity.append(per)
#             except:
#                 print(char, tag)
#         if tag not in ['I-'+tag_name, 'B-'+tag_name]:
#             if 'per' in locals().keys():
#                 entity.append(per)
#                 del per
#             continue
#     return entity
'''
def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG
'''

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


if __name__ == '__main__':

    label_set = ['Admission_Date', 'Admission_Number', 'Department', 'Discharge_Date', 'Hospital_Name']
    output = entity_extract(result_path='../model_hospital_400/test_predictions.txt', label_set=label_set, example_len=400)
    print(output)