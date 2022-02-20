from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from nltk.tokenize import word_tokenize

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
# RobertaForSequenceClassification
from transformers.modeling_roberta import RobertaModel
from transformers import AutoTokenizer, AutoModel

import os
import json


bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large'


def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir += flag_str
    print('starting model storing....')
    torch.save(model.state_dict(), output_dir)
    print('store succeed')


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        # self.roberta_single = RobertaModel.from_pretrained(pretrain_model_dir)
        self.roberta_single = RobertaModel.from_pretrained(
            "roberta-large", cache_dir='./cache')
        self.w_q = nn.Linear(bert_hidden_dim*2, bert_hidden_dim)
        self.w_c = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.v = nn.Linear(bert_hidden_dim, 1)
        self.single_hidden2tag = RobertaClassificationHead(
            bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask, span_a_mask, span_b_mask):
        # input_ids_a, input_mask_a, span_a_mask_a, input_ids_b, input_mask_b, span_b_mask_b):
        # single_train_input_ids, single_train_input_mask, single_train_segment_ids, single_train_label_ids = batch_single
        pair_output = self.roberta_single(input_ids, input_mask, None)[0]
        # (batch_size, sequence_length, hidden_size)`)
        span_a_reps = torch.sum(
            pair_output * span_a_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        span_b_reps = torch.sum(
            pair_output * span_b_mask.unsqueeze(2), dim=1)  # (batch, hidden)

        combined_rep = torch.cat(
            [span_a_reps, span_b_reps], dim=1)  # (batch, 2*hidden)

        context_reps = torch.cat(
            [span_a_reps, span_b_reps], dim=0)  # (2*batch, hidden)
        context_reps = context_reps.unsqueeze(0).repeat(
            span_a_reps.size(0), 1, 1)  # (batch, 2*batch, hidden)

        e_c = self.v(torch.tanh(self.w_q(combined_rep).unsqueeze(
            1).expand_as(context_reps)+self.w_c(context_reps))).view(context_reps.size(0), context_reps.size(1))  # (batch, 2*batch)
        attn_mask = torch.diag(torch.ones(context_reps.size(0))).repeat(
            1, 2).cuda()  # (batch, 2*batch)
        e_c = e_c.masked_fill(attn_mask == 1., -10000.0)

        attn_scores = F.softmax(e_c, dim=1)  # (batch, 2*batch)
        selected_mask = (torch.max(attn_scores, dim=1)[
                         0] > 0.047).float().unsqueeze(1)  # (batch, 1)
        selected_context_index = torch.argmax(attn_scores, dim=1).unsqueeze(
            1).unsqueeze(2).expand(context_reps.size(0), 1, bert_hidden_dim)
        selected_context_rep = context_reps.gather(
            1, selected_context_index).squeeze()  # (batch, hidden)
        selected_context_rep = selected_context_rep*selected_mask
        combined_context_rep = torch.cat(
            (combined_rep, selected_context_rep), 1)
#        print(attn_scores.cpu().detach().numpy().tolist())
        score_single = self.single_hidden2tag(
            combined_context_rep)  # (batch, tag_set)

        # score_single = self.single_hidden2tag(combined_rep)  # (batch, tag_set)
        return score_single


class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.out_proj = nn.Linear(bert_hidden_dim*3, num_labels)

    def forward(self, features):
        x = features  # [:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.out_proj(x)
        return x


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, span_a_left=None, span_a_right=None, text_b=None, span_b_left=None,
                 span_b_right=None, label=None, pair_id=None):
        """Constructs a InputExample.
        Args:
                guid: Unique id for the example.
                text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
                text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
                label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.span_a_left = span_a_left
        self.span_a_right = span_a_right

        self.text_b = text_b
        self.span_b_left = span_b_left
        self.span_b_right = span_b_right
        self.label = label
        self.pair_id = pair_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_a_mask, span_b_mask,
                 # input_ids_a, input_mask_a, segment_ids_a, span_a_mask_a,
                 # input_ids_b, input_mask_b, segment_ids_b, span_b_mask_b,
                 label_id, pair_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_a_mask = span_a_mask
        self.span_b_mask = span_b_mask

        self.label_id = label_id
        self.pair_id = pair_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class CEProcessor(DataProcessor):
    """Processor for the CE data set."""

    def get_data(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples = []
        readfile = open(filename, 'r')
        line_co = 0
        pos_size = 0
        for row in readfile:

            line = row.strip().split('\t')
            if len(line) == 9:
                guid = "test-" + str(line_co - 1)
                event_id_1 = line[0].strip()
                event_id_2 = line[1].strip()
                text_a = line[2].strip()
                span_a_left = int(line[3].strip())
                span_a_right = int(line[4].strip())
                text_b = line[5].strip()
                span_b_left = int(line[6].strip())
                span_b_right = int(line[7].strip())
                label = int(line[8].strip())
                if label == 1:
                    pos_size += 1
                examples.append(
                    InputExample(guid=guid, text_a=text_a, span_a_left=span_a_left, span_a_right=span_a_right,
                                 text_b=text_b, span_b_left=span_b_left, span_b_right=span_b_right, label=label,
                                 pair_id=event_id_1 + '&&' + event_id_2))
            # else:
            #       print(line)
            line_co += 1
        readfile.close()
        print('data line: ', line_co)
        print('loaded  size:', len(examples), ' pos_size:', pos_size)
        return examples


def wordpairID_2_tokenpairID(sentence, wordindex_left, wordindex_right, full_token_id_list, tokenizer, sent_1=True,
                             only_1_sen=False):
    '''pls note that the input indices pair include the b in (a,b), but the output doesn't'''
    '''first find the position of [sep, sep]'''

    position_two_two = 0
    for i in range(len(full_token_id_list)):
        if full_token_id_list[i] == 2 and full_token_id_list[i + 1] == 2:
            position_two_two = i
            break

    span = ' '.join(word_tokenize(sentence)[
                    wordindex_left: wordindex_right + 1])

    if wordindex_left != 0:
        '''this span is the begining of the sent'''
        span = ' ' + span

    span_token_list = tokenizer.tokenize(span)
    span_id_list = tokenizer.convert_tokens_to_ids(span_token_list)

    # print('span:', span, 'span_id_list:', span_id_list)
    if sent_1:
        # for i in range(wordindex_left, len(full_token_id_list)-len(span_id_list)):
        if only_1_sen:
            position_two_two = len(full_token_id_list)
        for i in range(wordindex_left, position_two_two):
            if full_token_id_list[i:i + len(span_id_list)] == span_id_list:
                return i, i + len(span_id_list), span_token_list

        return None, None, span_token_list
    else:
        for i in range(position_two_two + 2, len(full_token_id_list)):
            if full_token_id_list[i:i + len(span_id_list)] == span_id_list:
                return i, i + len(span_id_list), span_token_list

        return None, None, span_token_list


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                    - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                    - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    give_up = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        # print('tokens_a:', tokens_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # print('tokens_b:', tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        tokens = tokens_a + [sep_token]
        # tokens_a += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]

        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids

        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        span_a_left, span_a_right, span_a_token_list = wordpairID_2_tokenpairID(example.text_a, example.span_a_left,
                                                                                example.span_a_right, input_ids,
                                                                                tokenizer, sent_1=True,
                                                                                only_1_sen=False)
        span_b_left, span_b_right, span_b_token_list = wordpairID_2_tokenpairID(example.text_b, example.span_b_left,
                                                                                example.span_b_right, input_ids,
                                                                                tokenizer, sent_1=False,
                                                                                only_1_sen=False)

        if span_a_left is None or span_b_left is None:
            '''give up this pair'''
            give_up += 1
            continue
        else:

            span_a_mask = [0] * len(input_ids)
            for i in range(span_a_left, span_a_right):
                span_a_mask[i] = 1
            span_b_mask = [0] * len(input_ids)
            for i in range(span_b_left, span_b_right):
                span_b_mask[i] = 1

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              span_a_mask=span_a_mask,
                              span_b_mask=span_b_mask,
                              label_id=label_id,
                              pair_id=example.pair_id))
    print('input example size:', len(examples),
          ' give_up:', give_up, ' remain:', len(features))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def feature2vector(feature, batch_size):
    all_input_ids = torch.tensor(
        [f.input_ids for f in feature], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in feature], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in feature], dtype=torch.long)
    all_span_a_mask = torch.tensor(
        [f.span_a_mask for f in feature], dtype=torch.long)
    all_span_b_mask = torch.tensor(
        [f.span_b_mask for f in feature], dtype=torch.long)

    all_label_ids = torch.tensor(
        [f.label_id for f in feature], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_a_mask, all_span_b_mask,
                         all_label_ids)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return data, sampler, dataloader


def init(args, num_labels, device):
    model = RobertaForSequenceClassification(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-large", do_lower_case=args.do_lower_case, cache_dir='./cache')
    if args.model_path:
        model.load_state_dict(torch.load(
            args.model_path, map_location=torch.device(device)))
    model.to(device)

    return model, tokenizer


def finetune(args, model, tokenizer, label_list, num_labels, device, n_gpu, train_path, dev_path, test_path):

    processors = {
        "ce": CEProcessor
    }

    output_modes = {
        "ce": "classification"
    }

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    train_examples = processor.get_data(
        train_path) 
    dev_examples = processor.get_data(dev_path)
    test_examples = processor.get_data(test_path)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        # bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,  # 2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,
        # bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        # bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)  # 4 if args.model_type in ['xlnet'] else 0,)

    '''load dev set'''
    dev_features = convert_examples_to_features(
        dev_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        # bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,  # 2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,
        # bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        # bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)  # 4 if args.model_type in ['xlnet'] else 0,)

    dev_data, dev_sampler, dev_dataloader = feature2vector(
        dev_features, args.eval_batch_size)

    '''load test set'''
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        # bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,  # 2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,
        # bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        # bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)  # 4 if args.model_type in ['xlnet'] else 0,)

    eval_all_pair_ids = [f.pair_id for f in test_features]

    eval_data, eval_sampler, test_dataloader = feature2vector(
        test_features, args.eval_batch_size)

    print("***** Running training *****")
    print("  Num examples = %d", len(train_features))
    print("  Batch size = %d", args.train_batch_size)
    print("  Num steps = %d", num_train_optimization_steps)

    train_data, train_sampler, train_dataloader = feature2vector(
        train_features, args.train_batch_size)

    iter_co = 0
    final_test_performance = 0.0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids = batch

            logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            iter_co += 1
            # if iter_co %20==0:
            if iter_co % len(train_dataloader) == 0:
                # if iter_co % (len(train_dataloader)//2)==0:
                '''
                start evaluate on dev set after this epoch
                '''
                model.eval()

                for idd, dev_or_test_dataloader in enumerate([dev_dataloader, test_dataloader]):
                    # for idd, dev_or_test_dataloader in enumerate([dev_dataloader]):

                    if idd == 0:
                        print("***** Running dev *****")
                        print("  Num examples = %d", len(dev_features))
                    else:
                        print("***** Running test *****")
                        print("  Num examples = %d", len(test_features))
                    # print("  Batch size = %d", args.eval_batch_size)

                    eval_loss = 0
                    nb_eval_steps = 0
                    preds = []
                    gold_label_ids = []
                    # print('Evaluating...')
                    for input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids in dev_or_test_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        span_a_mask = span_a_mask.to(device)
                        span_b_mask = span_b_mask.to(device)

                        label_ids = label_ids.to(device)
                        gold_label_ids += list(label_ids.detach().cpu().numpy())

                        with torch.no_grad():

                            logits = model(input_ids, input_mask,
                                           span_a_mask, span_b_mask)
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(
                                preds[0], logits.detach().cpu().numpy(), axis=0)

                    preds = preds[0]

                    pred_probs = softmax(preds, axis=1)
                    if idd == 1:
                        '''which label is positiv?'''
                        score_for_print = list(pred_probs[:, 0])
                        assert len(eval_all_pair_ids) == len(score_for_print)
                    pred_label_ids = list(np.argmax(pred_probs, axis=1))

                    gold_label_ids = gold_label_ids
                    assert len(pred_label_ids) == len(gold_label_ids)
                    hit_co = 0
                    for k in range(len(pred_label_ids)):
                        if pred_label_ids[k] == gold_label_ids[k]:
                            hit_co += 1
                    test_acc = hit_co / len(gold_label_ids)
                    if len(label_list) == 2:
                        overlap = 0
                        for k in range(len(pred_label_ids)):
                            if pred_label_ids[k] == gold_label_ids[k] and gold_label_ids[k] == 1:
                                overlap += 1
                        recall = overlap / (1e-6 + sum(gold_label_ids))
                        precision = overlap / (1e-6 + sum(pred_label_ids))
                        f1 = 2 * recall * precision / \
                            (1e-6 + recall + precision)
                        print(precision, recall, f1)

                    if idd == 0:  # this is dev

                        if f1 > max_dev_acc:
                            max_dev_acc = f1
                            model_to_save = (
                                model.module if hasattr(
                                    model, "module") else model
                            )  # Take care of distributed/parallel training
                            store_transformers_models(model_to_save, tokenizer,
                                                      args.model_dir,
                                                      'attn_' + str(f1) + '.pt')
                            print('\ndev:', [test_acc, f1],
                                  ' max_dev_f1:', max_dev_acc, '\n')
                            '''store the model, because we can test after a max_dev acc reached'''
                        else:
                            print('\ndev:', [test_acc, f1],
                                  ' max_dev_f1:', max_dev_acc, '\n')
                            break
                    else:  # this is test
                        if f1 > max_test_acc:
                            max_test_acc = f1

                        '''write new scores to test file'''
                        writescore = codecs.open(
                            args.output_file, 'w', 'utf-8')
                        for id, score in enumerate(score_for_print):
                            pair_idd = eval_all_pair_ids[id].split('&&')
                            writescore.write(
                                pair_idd[0] + '\t' + pair_idd[1] + '\t' + str(score) + '\n')
                        print('test score written over')
                        writescore.close()
                        final_test_performance = f1
                        print('\ntest:', [test_acc, f1],
                              ' max_test_f1:', max_test_acc, '\n')
    print('final_test_f1:', final_test_performance)


def predict(args, model, tokenizer, label_list, device, test_path):
    processors = {
        "ce": CEProcessor
    }

    output_modes = {
        "ce": "classification"
    }

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    model.eval()

    test_examples = processor.get_data(test_path)

    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
        # bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,  # 2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,
        # bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        # bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)  # 4 if args.model_type in ['xlnet'] else 0,)

    eval_all_pair_ids = [f.pair_id for f in test_features]
    print(args.eval_batch_size)
    eval_data, eval_sampler, test_dataloader = feature2vector(
        test_features, args.eval_batch_size)

    print("***** Running test *****")
    print("  Num examples = %d", len(test_features))

    max_test_acc = 0.0

    preds = []
    gold_label_ids = []
    for input_ids, input_mask, segment_ids, span_a_mask, span_b_mask, label_ids in tqdm(test_dataloader):

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        span_a_mask = span_a_mask.to(device)
        span_b_mask = span_b_mask.to(device)

        label_ids = label_ids.to(device)
        gold_label_ids += list(label_ids.detach().cpu().numpy())

        with torch.no_grad():
            logits = model(input_ids, input_mask, span_a_mask, span_b_mask)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]

    pred_probs = softmax(preds, axis=1)
    score_for_print = list(pred_probs[:, 0])
    assert len(eval_all_pair_ids) == len(score_for_print)
    pred_label_ids = list(np.argmax(pred_probs, axis=1))

    gold_label_ids = gold_label_ids
    assert len(pred_label_ids) == len(gold_label_ids)
    hit_co = 0
    for k in range(len(pred_label_ids)):
        if pred_label_ids[k] == gold_label_ids[k]:
            hit_co += 1
    test_acc = hit_co / len(gold_label_ids)
    if len(label_list) == 2:
        overlap = 0
        for k in range(len(pred_label_ids)):
            if pred_label_ids[k] == gold_label_ids[k] and gold_label_ids[k] == 1:
                overlap += 1
        recall = overlap / (1e-6 + sum(gold_label_ids))
        precision = overlap / (1e-6 + sum(pred_label_ids))
        f1 = 2 * recall * precision / \
            (1e-6 + recall + precision)
        print(precision, recall, f1)

    '''write new scores to test file'''
    writescore = codecs.open(
        args.output_file, 'w', 'utf-8')
    for id, flag in enumerate(score_for_print):
        pair_idd = eval_all_pair_ids[id].split('&&')
        writescore.write(pair_idd[0] + '\t' +
                         pair_idd[1] + '\t' + str(flag) + '\n')
    print('complex event predictions written over')
    writescore.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input model path. ")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model predictions will be written.")
    parser.add_argument("--model_dir",
                        default='./models/',
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    # Other parameters
    parser.add_argument("--cache_dir",
                        default="./cache",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--kshot',
                        type=int,
                        default=5,
                        help="random seed for initialization")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--learning_rate",
                        default=1e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--use_mixup",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--beta_sampling_times',
                        type=int,
                        default=10,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument("--per_gpu_train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The training pair file. ")

    parser.add_argument("--dev_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The dev pair file. ")

    parser.add_argument("--test_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The test pair file. ")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    label_list = [0, 1]
    num_labels = len(label_list)

    model, tokenizer = init(args, num_labels, device)

    train_path = args.train_file
    dev_path = args.dev_file
    test_path = args.test_file

    if args.do_train:
        finetune(args, model, tokenizer, label_list, num_labels,
                 device, n_gpu, train_path, dev_path, test_path)
    else:
        predict(args, model, tokenizer, label_list,
                device, test_path)


if __name__ == "__main__":
    main()
