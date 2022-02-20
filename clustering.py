from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import gc
import sys
import jsonlines
import json
import subprocess
from collections import Counter

import _pickle as cPickle
import logging
import argparse

from operator import itemgetter
import collections

import operator


def get_preds(file, output_file, threshold, ce_input_file, ce_context_file, ce_cluster_file):
    clustering = AgglomerativeClustering(
        n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=threshold)
    documents = {}
    doc_clusters = {}
    pairs = {}
    with open(file, 'r') as f_r:
        for line in f_r:
            eles = line.strip().split('\t')
            pairs[(eles[0], eles[1])] = float(eles[2])
            doc_id = eles[0].split('-')[1]
            event_a = int(eles[0].split('-')[3])
            event_b = int(eles[1].split('-')[3])
            if doc_id not in documents:
                documents[doc_id] = max(event_a, event_b)
            else:
                documents[doc_id] = max(event_a, event_b, documents[doc_id])

    all_outputs = 0
    for doc_id, events_num in documents.items():
        outputs = []
        for i in range(events_num):
            output = []
            for j in range(events_num):
                if j < i:
                    if ('-'.join(['article', doc_id, 'event', str(j+1)]),
                            '-'.join(['article', doc_id, 'event', str(i+1)])) not in pairs:
                        output.append(1.)
                    else:
                        output.append(pairs[('-'.join(['article', doc_id, 'event', str(j+1)]),
                                             '-'.join(['article', doc_id, 'event', str(i+1)]))])
                elif i == j:
                    output.append(0.)
                else:
                    if ('-'.join(['article', doc_id, 'event', str(i+1)]),
                            '-'.join(['article', doc_id, 'event', str(j+1)])) not in pairs:
                        output.append(1.)
                    else:
                        output.append(pairs[('-'.join(['article', doc_id, 'event', str(i+1)]),
                                             '-'.join(['article', doc_id, 'event', str(j+1)]))])
            outputs.append(output)

        predicted = clustering.fit(np.array(outputs))
        doc_clusters[doc_id] = predicted.labels_
        all_outputs += events_num
    #     print(predicted.labels_)
    # print(len(documents))

    docs = list(documents.keys())
    docs.sort(key=lambda x: int(x))
    # print(len(docs), ' documents')
    # print(docs, 'docs')
    with open(output_file, 'w', encoding='utf-8') as f_w:
        for document in docs:
            f_w.write('#begin document ('+document+'); part 000\n')
            event_labels = doc_clusters[document]
            for event in event_labels:
                f_w.write(document+'('+str(event)+')\n')
            f_w.write('#end document\n')

    if ce_context_file:
        event2sent = dict()  # (doc_id,(event_id,sent))
        with open(ce_input_file, 'r') as f_r:
            for line in f_r:
                eles = line.strip().split('\t')
                assert len(eles) == 9
                event_a, event_b = eles[0], eles[1]
                doc = event_a.split('-')[1]
                if doc not in event2sent:
                    event2sent[doc] = dict()
                event2sent[doc][event_a.split('-')[-1]] = eles[2]
                event2sent[doc][event_b.split('-')[-1]] = eles[5]

        with open(ce_context_file, 'w') as f_context:
            with open(ce_cluster_file, 'w') as f_cluster:
                for doc_id in docs:
                    f_context.write('article-'+str(doc_id)+'\n')
                    f_cluster.write('article-'+str(doc_id)+'\n')
                    labels = list(doc_clusters[doc_id])
                    counter = Counter(labels).most_common()
                    for num, freq in counter:
                        if freq == 1:
                            break
                        start = 0
                        cluster = []
                        sents = []
                        for i in range(freq):
                            start = labels.index(num, start)+1
                            cluster.append(start)
                            if event2sent[doc_id][str(start)] not in sents:
                                sents.append(event2sent[doc_id][str(start)])
                        for c in cluster:
                            f_cluster.write('article-'+doc_id +
                                            '-event-'+str(c)+'\n')
                        f_cluster.write('\n')
                        f_context.write(' '.join(sents)+'\n')
                    f_context.write('\n')

                # cluster = sorted(cluster, key=lambda x: int(x))
                # text = ' '.join(sentences[document][x] for x in cluster)
                # f_w.write(
                #     {'id': '-'.join(['article', document, 'complex_event', str(i)]), 'text': text})
    return doc_clusters


def read_conll_f1(filename):
    '''
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :return: the CoNLL F1
    '''
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    muc_f1 = f1_list[1]
    bcued_f1 = f1_list[3]
    ceafe_f1 = f1_list[7]

    return (muc_f1 + bcued_f1 + ceafe_f1)/float(3)


def run_conll_scorer(key_file, response_file, event_conll_file, conll_output):

    event_scorer_command = ('perl reference-coreference-scorers/scorer.pl all {} {} none > {} \n'.format
                            (key_file, response_file, event_conll_file))

    # entity_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
    #         (config_dict["entity_gold_file_path"], entity_response_filename, entity_conll_file))

    processes = []
    # print('Run scorer command for cross-document event coreference: {} \n'.format(event_scorer_command))
    processes.append(subprocess.Popen(event_scorer_command, shell=True))

    # print('Run scorer command for cross-document entity coreference')
    # processes.append(subprocess.Popen(entity_scorer_command, shell=True))

    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    scores_file = open(conll_output, 'w')

    event_f1 = read_conll_f1(event_conll_file)
    # entity_f1 = read_conll_f1(entity_conll_file)
    scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
    # scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))

    scores_file.close()
    return event_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_predictions",
                        default='outputs/dev_complex_event_predictions.txt',
                        type=str,
                        help="dev complex event predictions")
    parser.add_argument("--test_predictions",
                        default='outputs/test_complex_event_predictions.txt',
                        type=str,
                        help="test complex event predictions")
    parser.add_argument("--dev_response",
                        default='outputs/dev.response',
                        type=str,
                        help="convert dev complex event predictions to conll f1 format")
    parser.add_argument("--test_response",
                        default='outputs/test.response',
                        type=str,
                        help="convert test complex event predictions to conll f1 format")
    parser.add_argument("--dev_key",
                        default='data/complex/gold_dev.key',
                        type=str,
                        help="dev conll f1 format key")
    parser.add_argument("--test_key",
                        default='data/complex/gold_test.key',
                        type=str,
                        help="test conll f1 format key")
    parser.add_argument("--conll_results",
                        default='outputs/event_conll_result.txt',
                        type=str,
                        help="conll results")
    parser.add_argument("--conll_score",
                        default='outputs/conll_f1_score.txt',
                        type=str,
                        help="conll f1 score")
    parser.add_argument("--ce_input_file",
                        default='data/complex/test_data.txt',
                        type=str,
                        help="The pairwise complex event identification input file")
    parser.add_argument("--ce_context_file",
                        default='outputs/ce_context.txt',
                        type=str,
                        help="The context output file")
    parser.add_argument("--ce_cluster_file",
                        default='outputs/ce_events.txt',
                        type=str,
                        help="The complex event output file")
    args = parser.parse_args()
    # build_clusters_for_sub(args.predictions, args.response,
    #                        args.complex_events, args.srl_inputs)
    max_f1 = 0
    max_f1_threshold = 0
    cur_threshold = 0.1
    thresholds = [0.01]
    for i in range(90):
        cur_threshold += 0.01
        thresholds.append(cur_threshold)

    for i in thresholds:
        all_clusters = get_preds(
            args.dev_predictions, args.dev_response, i, None, None, None)

        f1 = run_conll_scorer(args.dev_key, args.dev_response,
                              args.conll_results, args.conll_score)

        if f1 > max_f1:
            max_f1 = f1
            max_f1_threshold = i
            print(max_f1, max_f1_threshold)

    print('dev performance, f1 : {}, threshold: {}'.format(
        round(max_f1, 4), round(max_f1_threshold, 2)))
    all_clusters = get_preds(
        args.test_predictions, args.test_response, max_f1_threshold, args.ce_input_file, args.ce_context_file, args.ce_cluster_file)

    f1 = run_conll_scorer(args.test_key, args.test_response,
                          args.conll_results, args.conll_score)
    print('test performance, f1: {}, threshold: {}'.format(
        round(f1, 4), round(max_f1_threshold, 2)))

