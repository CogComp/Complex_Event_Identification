import os
from os import listdir
from os.path import isfile, join
import random
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import jsonlines


def merge(sts):
    i = 0
    while i < len(sts):
        j = i+1
        while j < len(sts):
            if len(sts[i].intersection(sts[j])) > 0:
                sts[i] = sts[i].union(sts[j])
                sts.pop(j)
            else:
                j += 1
        i += 1
    lst = [list(s) for s in sts]
    return lst


def read_file(mode):
    positive_pairs = []
    negative_pairs = []
    all_pairs = []
    random.seed(10)
    complex_event_num = 0
    files = ['./data/HiEve/processed/'+f for f in listdir('./data/HiEve/processed')]
    files.sort()
    # print(files)
    # print(len(files))
    if mode == 'train':
        files = files[:len(files)//10*6]
    elif mode == 'dev':
        files = files[len(files)//10*6:len(files)//10*8]
    else:
        files = files[len(files)//10*8:]
    # print(len(files),'number of files')
    # print(files)
    num_clusters = []
    id2event = {}
    event_num_for_files = {}

    for file in files:
        total_event_count = 0
        with open(file, 'r', encoding='utf-8') as f:
            text = f.readline().strip().split('\t')[1]
            sentences = sent_tokenize(text)
            # start = 0
            sentence_span = {}
            for sent in sentences:
                start = text.index(sent)
                index = start+len(sent)-1

                text_span = word_tokenize(sent)
                if text_span:
                    sentence_span[(start, index)] = text_span
                # start = index+2
            clusters = []
            events = {}
            for line in f:
                line = line.strip()
                elements = line.split('\t')
                if elements[0] == 'Event':
                    event_index = int(elements[1])
                    trigger = elements[2]
                    start = int(elements[4])
                    sent = None

                    for span in sentence_span:
                        if start <= span[1]:
                            sent = sentence_span[(span[0], span[1])]
                            break
                    # print(sent,'e')
                    # print(start,sentence_span)
                    if file == './data/HiEve/processed/article-12028.tsvx' and trigger == 'sitting':
                        trigger = 'baby-sitting'
                    if file == './data/HiEve/processed/article-17016.tsvx' and trigger == '11':
                        trigger = '9/11'
                    elif file == './data/HiEve/processed/article-15808.tsvx' and trigger == 'take':
                        trigger = 'take-off'
                    elif file == './data/HiEve/processed/article-1526.tsvx' and trigger == 'tip':
                        trigger = 'tip-off'
                    elif file == './data/HiEve/processed/article-3736.tsvx' and trigger == 'elections' and start == 2822:
                        trigger = 'by-elections'
                    elif file == './data/HiEve/processed/article-14969.tsvx' and start == 437:
                        trigger = 're-light'
                    elif file == './data/HiEve/processed/article-1857.tsvx' and start == 1488:
                        trigger = 'run-offs'
                    elif file == './data/HiEve/processed/article-1857.tsvx' and start == 2430:
                        trigger = 'run-offs'
                    elif file == './data/HiEve/processed/article-1857.tsvx' and start == 2720:
                        trigger = 'run-offs'
                    elif file == './data/HiEve/processed/article-17803.tsvx' and trigger == 'run':
                        trigger = 'run-in'
                    # print(file,trigger,start,sent)
                    # print(file,sent,'here',start)
                    # print(sent)
                    left_span = sent.index(trigger.split()[0])
                    # if len(trigger.split()) > 1:
                    #     print(trigger, 'ss')
                    right_span = left_span+len(trigger.split())-1
                    # print(sent_end_idx +
                    #       len(' '.join(sent[:left_span]))-1, start)
                    # assert sent_end_idx + \
                    #     len(' '.join(sent[:left_span]))-1 == start
                    events[event_index] = [
                        trigger, left_span, right_span, ' '.join(sent)]
                else:
                    if elements[3] == 'NoRel' or elements[3] == 'Coref':
                        continue
                    event_a_index = elements[1]
                    event_b_index = elements[2]
                    if not clusters:
                        new_set = set()
                        new_set.add(event_a_index)
                        new_set.add(event_b_index)
                        clusters.append(new_set)
                    else:
                        flag = 1
                        for cluster in clusters:
                            if event_a_index in cluster:
                                cluster.add(event_b_index)
                                flag = 0
                            elif event_b_index in cluster:
                                cluster.add(event_a_index)
                                flag = 0
                        if flag:
                            new_set = set()
                            new_set.add(event_a_index)
                            new_set.add(event_b_index)
                            clusters.append(new_set)

            clusters = merge(clusters)
            complex_event_pairs = []
            # num_clusters.append(len(clusters))

            # number of granular events in each cluster
            num_clusters.extend([len(cluster) for cluster in clusters])

            for cluster in clusters:
                for i in range(len(cluster)-1):
                    for j in range(i+1, len(cluster)):
                        complex_event_pairs.append((int(cluster[i]), int(cluster[j]))) if int(cluster[i]) < int(
                            cluster[j]) else complex_event_pairs.append((int(cluster[j]), int(cluster[i])))

            complex_event_num += len(complex_event_pairs)
            complex_event_pairs.sort(key=lambda x: (x[0], x[1]))

            prefix = file.split('/')[-1][:-5]+'-event-'
            event_num_for_files[file.split('/')[-1][:-5]] = len(events)
            neg_count = 0
            # events index start from 1
            cur_positive_pairs, cur_negative_pairs = [], []
            for i in range(1, len(events)):
                for j in range(i+1, len(events)+1):
                    # flag means i and j belong to the same complex event

                    flag = 1 if (i, j) in complex_event_pairs else 0
                    event_a, event_b = [], []
                    event_a_span_left, event_a_span_right, event_b_span_left, event_b_span_right = -1, -1, -1, -1
                    event_a.append(events[i][3])
                    event_a_span_left = events[i][1]
                    event_a_span_right = events[i][2]
                    event_b.append(events[j][3])
                    event_b_span_left = events[j][1]
                    event_b_span_right = events[j][2]
                    assert event_b_span_left != -1
                    assert event_a_span_left != -1
                    if flag == 1:
                        cur_positive_pairs.append([prefix+str(i), prefix+str(j), ' '.join(event_a), str(event_a_span_left), str(
                            event_a_span_right), ' '.join(event_b), str(event_b_span_left), str(event_b_span_right), str(flag)])
                    else:
                        cur_negative_pairs.append([prefix+str(i), prefix+str(j), ' '.join(event_a), str(event_a_span_left), str(
                            event_a_span_right), ' '.join(event_b), str(event_b_span_left), str(event_b_span_right), str(flag)])

                    if prefix+str(i) not in id2event:
                        id2event[prefix+str(i)] = [' '.join(event_a),
                                                   str(event_a_span_left), str(event_a_span_right)]
                    if prefix+str(j) not in id2event:
                        id2event[prefix+str(j)] = [' '.join(event_b),
                                                   str(event_b_span_left), str(event_b_span_right)]

                    if flag == 0:
                        neg_count += 1

            cur_pairs = cur_positive_pairs+cur_negative_pairs
            #if mode == 'train':
            random.shuffle(cur_pairs)

            total_event_count += len(events)
        all_pairs.extend(cur_pairs)

    # print(len(negative_pairs)/len(positive_pairs), 'ratio')
    # print(len(positive_pairs)+len(negative_pairs))
#    if mode == 'train':
#        print('here')
#        negative_pairs = random.sample(negative_pairs,len(positive_pairs)*3)

    # all_pairs = positive_pairs+negative_pairs
    print(len(all_pairs), 'all_pairs')
    # random.shuffle(all_pairs)
    # context_per_pair = 0

    # complex_dict = {}
    # for num in num_clusters:
    #     if str(num) not in complex_dict:
    #         complex_dict[str(num)] = 1
    #     else:
    #         complex_dict[str(num)] += 1
    # for key, value in complex_dict.items():
    #     print(key, ' ', value)
    with open('data/complex/'+mode+'_data.txt', 'w', encoding='utf-8') as f_w:
        count = 0
        for i in range(len(all_pairs)):
            f_w.write('\t'.join(all_pairs[i])+'\n')
            count += 1
        print(count)


if __name__ == '__main__':
    read_file('train')
    read_file('dev')
    read_file('test')
