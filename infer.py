# coding=utf-8

from __future__ import absolute_import, division, print_function

import logging
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# from tqdm import tqdm, trange
from utils_ner import get_labels
from utils.entity_extract import get_entity

from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from config import *
from IE_evaluation_utils import extract_entity, entity_assign2catego, ATTACH_SYMBOL
from constants_jw import PAT_DICT, ENTITY_C2E_DICT, MRC_QUESTIONS
import ipdb

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )),
    ())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
}


class ArgumentParser:
    def __init__(self, model_type, model_name_or_path, labels, config_name, tokenizer_name, max_seq_length,
                 per_gpu_infer_batch_size, local_rank=-1, do_lower_case=True, no_cuda=False, fp16=False):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.labels = labels
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.per_gpu_infer_batch_size = per_gpu_infer_batch_size
        self.do_lower_case = do_lower_case
        self.no_cuda = no_cuda
        self.fp16 = fp16
        self.local_rank = local_rank


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_len):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len


class NerInfer:
    def __init__(self, model_type=model_type, model_name_or_path=None, labels=None,
                   config_name=None,
                   tokenizer_name=None, max_seq_length=max_seq_length,
                   per_gpu_infer_batch_size=per_gpu_infer_batch_size):

        args = ArgumentParser(model_type, model_name_or_path, labels, config_name, tokenizer_name, max_seq_length,
                 per_gpu_infer_batch_size, do_lower_case=True, no_cuda=False, fp16=False)

        self.args, self.model, self.tokenizer, self.labels = self.initialize(args)
        self.catego_ls = []
        for ele in self.labels:
            if 'B-' in ele:
                self.catego_ls.append(ele.replace('B-', ''))

        self.names_of_extracted_entity = self.get_names_of_extracted_entity(self.labels)

        logging.info('**** Initialized! ****')

    @staticmethod
    def get_names_of_extracted_entity(labels):
        labels = [i for i in labels if i != 'O']
        names = [i.replace('B-', '').replace('I-', '').replace('O-', '').replace('S-', '').replace('E-', '') for i in labels]
        return set(names)

    def initialize(self, args):

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            device = 0
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1
        args.device = device

        # Setup logging
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

        labels = get_labels(args.labels)
        num_labels = len(labels)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        # pad_token_label_id = CrossEntropyLoss().ignore_index

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels)
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        if torch.cuda.is_available():
            model.to(args.device)

        logger.info("Prediction parameters %s", args)

        return args, model, tokenizer, labels

    @staticmethod
    def convert_input_to_features(input_list,
                                  # label_list,  # 预测时没有label
                                  max_seq_length,
                                  tokenizer,
                                  cls_token_at_end=False,
                                  cls_token="[CLS]",
                                  cls_token_segment_id=1,
                                  sep_token="[SEP]",
                                  sep_token_extra=False,
                                  pad_on_left=False,
                                  pad_token=0,
                                  pad_token_segment_id=0,
                                  # pad_token_label_id=-1,
                                  sequence_a_segment_id=0,
                                  mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        # label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(input_list):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(input_list))

            tokens = []
            # label_ids = []
            # for word, label in zip(example.words, example.labels):
            for word in example:
                # word_tokens = tokenizer.tokenize(word) if word != '' else tokenizer.tokenize('卍')  # 中文中对空格的处理
                word_tokens = tokenizer.tokenize(word) if word != '' else ['[unused1]']  # 中文中对空格的处理
                # word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
                # label_ids = label_ids[:(max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            # label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                # label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                # label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                # label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                # label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += ([pad_token] * padding_length)
                input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids += ([pad_token_segment_id] * padding_length)
                # label_ids += ([pad_token_label_id] * padding_length)



            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert len(label_ids) == max_seq_length

            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     # logger.info("guid: %s", example.guid)
            #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                # logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              input_len=len(example)
                              ))
        return features

    def generate_infer_dataset(self, args, input_list, tokenizer):
        logger.info("Creating features to predict")
        features = self.convert_input_to_features(input_list, args.max_seq_length, tokenizer,
                                                  cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                  # xlnet has a cls token at the end
                                                  cls_token=tokenizer.cls_token,
                                                  cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                  sep_token=tokenizer.sep_token,
                                                  sep_token_extra=bool(args.model_type in ["roberta"]),
                                                  # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                  pad_on_left=bool(args.model_type in ["xlnet"]),
                                                  # pad on the left for xlnet
                                                  pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                  pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0
                                                  )
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_input_len)
        return dataset

    def infer(self, input_list, MRC_QUESTIONS, mrc_ret=None):
        label_map = {i: label for i, label in enumerate(self.labels)}

        input_list_len = len(input_list)

        pred_label_lls = [[] for _ in range(input_list_len)]
        for i in range(input_list_len):
            pred_label_lls[i] = list(map(lambda x:label_map[x], preds[i][1:all_inputs_len[i]+1]))
            # for j in range(1, all_inputs_len[i] + 1):
            #     pred_label_lls[i].append(label_map[preds[i][j]])

        # logger.info('label_lls:{}'.format(pred_label_lls))
        


        # entity_list = []
        # for i in range(input_list_len):
        #     tag_id = preds[i][1: all_inputs_len[i] + 1]
        #     entity_list.append(get_entity(tag_id_seq=tag_id, original_seq=input_list[i],
        #                                   label_set=self.names_of_extracted_entity, tag_set=self.labels))

        return entity_list


if __name__ == "__main__":
    ner_infer = NerInfer()

    input = ['毛泽东在天安门广场发表讲话     ', '邓小平在深圳参观    ']

    preds_list = ner_infer.infer(input)

     # print(preds_list)



