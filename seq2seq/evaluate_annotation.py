# -*- coding:utf-8 -*-
"""
アノテータデータの評価用スクリプト
check_kappa: fleiss's kappa のスコアを計算する関数
evaluate1: task1のアノテートデータの評価
evaluate2: task2のアノテートデータの評価

"""

import nltk
import pickle
import numpy as np
import pandas as pd
import matplotlib
from test_data import parse_ja_text
matplotlib.use('agg')

def check_kappa1():
    """
    fleiss's kappa を計算する
    :return:
    """
    # load some data
    books1 = [
        # test
        # './annotation_files/_test/annotation1_tanaka.xlsx',
        # './annotation_files/_test/annotation1_isogawa.xlsx',
        # './annotation_files/_test/annotation1_miura.xlsx',
        # './annotation_files/_test/annotation1_takayama.xlsx',

        # production
        './annotation_files/_production/annotation1_maekawa.xlsx',
        './annotation_files/_production/annotation1_wakuta.xlsx',
        './annotation_files/_production/annotation1_sasaki.xlsx',
        './annotation_files/_production/annotation1_nagata.xlsx',
        './annotation_files/_production/annotation1_takebayashi.xlsx',
    ]
    with open('annotation_files/swap_keys.pkl', 'rb') as f:
        swap_keys = pickle.load(f)

    # making data frames
    sheet_name = 'annotation1'
    data_frames = []
    for book in books1:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))

    # make a matrix
    mat = [[] for _ in range(6)]
    for coder_index, data_frame in enumerate(data_frames):
        for item_index, line in data_frame.iterrows():
            if swap_keys[item_index]:
                # swap している場合（A: proposal, B: existing）
                row_order = [str(line['fluency_B']), str(line['fluency_A']),
                             str(line['consistency_B']), str(line['consistency_A']),
                             str(line['domain_consistency']), str(line['emotion'])]
                for metric_index in range(len(row_order)):
                    mat[metric_index].append((coder_index, str(item_index), row_order[metric_index]))
            else:
                # swap していない場合（A: existing, B: proposal）
                row_order = [str(line['fluency_A']), str(line['fluency_B']),
                             str(line['consistency_A']), str(line['consistency_B']),
                             str(line['domain_consistency']), str(line['emotion'])]
                for metric_index in range(len(row_order)):
                    mat[metric_index].append((coder_index, str(item_index), row_order[metric_index]))
            # TODO: 本番は除去
            # if item_index == 49:
            #     break

    # show fleiss's kappa for each evaluation metrics
    for m in mat:
        task = nltk.AnnotationTask(data=m)
        print('fleiss kappa:', task.multi_kappa())


def check_kappa2():
    books2 = [
        # test
        # './annotation_files/_test/annotation2_tanaka.xlsx',
        # './annotation_files/_test/annotation2_isogawa.xlsx',
        # './annotation_files/_test/annotation2_miura.xlsx',
        # './annotation_files/_test/annotation2_takayama.xlsx',

        # production
        './annotation_files/_production/annotation2_maekawa.xlsx',
        './annotation_files/_production/annotation2_wakuta.xlsx',
        './annotation_files/_production/annotation2_sasaki.xlsx',
        './annotation_files/_production/annotation2_nagata.xlsx',
        './annotation_files/_production/annotation2_takebayashi.xlsx',
    ]

    # making data frames
    sheet_name = 'annotation2'
    data_frames = []
    for book in books2:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))

    # make a matrix
    mat = [[] for _ in range(1)]
    for coder_index, data_frame in enumerate(data_frames):
        for item_index, line in data_frame.iterrows():
            row_order = [str(line['emotion_tag'])]
            for metric_index in range(len(row_order)):
                mat[metric_index].append((coder_index, str(item_index), row_order[metric_index]))

    # show fleiss's kappa for each evaluation metrics
    for m in mat:
        task = nltk.AnnotationTask(data=m)
        print('fleiss kappa:', task.multi_kappa())


def evaluate_task1():
    """
    アノテートされたデータを評価する関数
    :return:
    """
    books1 = [
        # test
        # './annotation_files/_test/annotation1_tanaka.xlsx',
        # './annotation_files/_test/annotation1_isogawa.xlsx',
        # './annotation_files/_test/annotation1_miura.xlsx',
        # './annotation_files/_test/annotation1_takayama.xlsx',

        # production
        './annotation_files/_production/annotation1_maekawa.xlsx',
        './annotation_files/_production/annotation1_wakuta.xlsx',
        './annotation_files/_production/annotation1_sasaki.xlsx',
        './annotation_files/_production/annotation1_nagata.xlsx',
        './annotation_files/_production/annotation1_takebayashi.xlsx',
    ]

    with open('annotation_files/swap_keys.pkl', 'rb') as f:
        swap_keys = pickle.load(f)

    # making data frames
    sheet_name = 'annotation1'
    data_frames = []
    for book in books1:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))

    # counting
    graph_data = []
    all_fluency = [0, 0]            # fluency[0]: 既存手法の成功数, fluency[1]: 提案手法の成功数
    all_consistency = [0, 0]        # consistency[0]: 既存手法の成功数, consistency[1]: 提案手法の成功数
    all_domain_consistency = 0      # 会話ドメイン整合性の観点で提案手法が選択された数
    all_emotion = 0                 # 感情の豊かさで提案手法が選択された数
    user_num = len(data_frames)     # アノテータ数
    for data_frame in data_frames:

        text_num = 0                # テストデータ数
        fluency = [0, 0]            # fluency[0]: 既存手法の成功数, fluency[1]: 提案手法の成功数
        consistency = [0, 0]        # consistency[0]: 既存手法の成功数, consistency[1]: 提案手法の成功数
        domain_consistency = 0      # 会話ドメイン整合性の観点で提案手法が選択された数
        emotion = 0                 # 感情の豊かさで提案手法が選択された数
        for index, line in data_frame.iterrows():
            # data が入っている場合のみカウント
            if not np.isnan(line['fluency_A']):
                text_num += 1
                if swap_keys[index]:
                    # swap している場合（A: proposal, B: existing）
                    fluency[0] += line['fluency_B']
                    fluency[1] += line['fluency_A']
                    consistency[0] += line['consistency_B']
                    consistency[1] += line['consistency_A']
                    if line['domain_consistency'] == 'a':
                        domain_consistency += 1
                    else:
                        # none case
                        pass
                    if line['emotion'] == 'a':
                        emotion += 1
                else:
                    # swap していない場合（A: existing, B: proposal）
                    fluency[0] += line['fluency_A']
                    fluency[1] += line['fluency_B']
                    consistency[0] += line['consistency_A']
                    consistency[1] += line['consistency_B']
                    if line['domain_consistency'] == 'b':
                        domain_consistency += 1
                    else:
                        # none case
                        pass
                    if line['emotion'] == 'b':
                        emotion += 1
        print('text num:', text_num)
        graph_data.append([float(fluency[0] / text_num), float(fluency[1] / text_num), float(consistency[0] / text_num),
                           float(consistency[1] / text_num), float(domain_consistency / text_num),
                           float(emotion / text_num)])
        all_fluency[0] += float(fluency[0] / text_num)
        all_fluency[1] += float(fluency[1] / text_num)
        all_consistency[0] += float(consistency[0] / text_num)
        all_consistency[1] += float(consistency[1] / text_num)
        all_domain_consistency += float(domain_consistency / text_num)
        all_emotion += float(emotion / text_num)

    # making graph
    # import matplotlib
    # import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    # font = {'family': 'Osaka'}
    # matplotlib.rc('font', **font)
    # plt.rcParams.update({'font.size': 10})
    # default_x = [i for i in range(1, 7)]
    # peaple = ['田中', '五十川', '三浦', '高山']
    # colors = ['b', 'g', 'r', 'y']
    # width = 0.2
    # for index, individual in enumerate(graph_data):
    #     left = [i + (width * index) for i in default_x]
    #     plt.bar(left=left, height=np.array(individual), width=width, label=peaple[index], color=colors[index],
    #             align='center')
    # plt.legend(loc="best")
    # plt.xticks([(i + (i + width * len(peaple))) / 2 - 0.1 for i in default_x],
    #            ['fluency_A', 'fluency_B', 'consistency_A',
    #             'consistency_B', 'domain_consistency', 'emotion'])
    # plt.show()

    print('fluency: ', '既存手法：', float(all_fluency[0]) / float(user_num),
          '提案手法：', float(all_fluency[1]) / float(user_num))
    print('consistency: ', '既存手法：', float(all_consistency[0]) / float(user_num),
          '提案手法：', float(all_consistency[1]) / float(user_num))
    print('domain_consistency: ', '既存手法：', 1 - (float(all_domain_consistency) / float(user_num)),
          '提案手法：', float(all_domain_consistency) / float(user_num))
    print('emotion: ', '既存手法：', 1 - (float(all_emotion) / float(user_num)),
          '提案手法：', float(all_emotion) / float(user_num), end='\n\n')


def evaluate_task2():
    """
    タスク２の評価用
    :return:
    """
    books2 = [
        # test
        # './annotation_files/_test/annotation2_tanaka.xlsx',
        # './annotation_files/_test/annotation2_isogawa.xlsx',
        # './annotation_files/_test/annotation2_miura.xlsx',
        # './annotation_files/_test/annotation2_takayama.xlsx',

        # production
        './annotation_files/_production/annotation2_maekawa.xlsx',
        './annotation_files/_production/annotation2_wakuta.xlsx',
        './annotation_files/_production/annotation2_sasaki.xlsx',
        './annotation_files/_production/annotation2_nagata.xlsx',
        './annotation_files/_production/annotation2_takebayashi.xlsx',
    ]
    with open('annotation_files/correct_emo_tag.txt', 'rb') as f:
        correct_emo_tags = pickle.load(f)
    print('正解タグ数：', len(correct_emo_tags))
    # tags = ['pos', 'neg', 'neu', 'none']         # none を考慮するケース
    tags = ['pos', 'neg', 'neu']                   # none を考慮しないケース

    # making data frames
    sheet_name = 'annotation2'
    data_frames = []
    for book in books2:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))

    # counting
    graph_data = []
    all_emotion_tag = 0
    user_num = len(data_frames)             # アノテータ数
    for data_frame in data_frames:
        emotion_tag = 0                     # 感情制御の成功数
        text_num = 0                        # テストデータ数
        for index, line in data_frame.iterrows():
            # data が入っている場合のみカウント
            if isinstance(line['emotion_tag'], str) and line['emotion_tag'] in tags:
                text_num += 1
                if line['emotion_tag'] == correct_emo_tags[index]:
                    emotion_tag += 1
        print('text num:', text_num)
        graph_data.append([float(emotion_tag / text_num)])
        all_emotion_tag += float(emotion_tag / text_num)
    print('emotion_tag : ', float(all_emotion_tag / user_num))

    # making graph
    # import matplotlib
    # import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    # font = {'family': 'Osaka'}
    # matplotlib.rc('font', **font)
    # plt.rcParams.update({'font.size': 10})
    # default_x = [i for i in range(1, 2)]
    # peaple = ['田中', '五十川', '三浦', '高山']
    # colors = ['b', 'g', 'r', 'y']
    # width = 0.1
    # for index, individual in enumerate(graph_data):
    #     left = [i + (width * index) for i in default_x]
    #     plt.bar(left=left, height=np.array(individual), width=width, label=peaple[index], color=colors[index],
    #             align='center')
    # plt.legend(loc="best")
    # plt.xticks([(i + (i + width * len(peaple))) / 2 - 0.1 for i in default_x], ['emotion_tag'])
    # plt.show()


def emotion_word():
    """
    評価指標：Emotion Word に対応
    感情語彙が正しく出力されているかどうかをカウントする関数
    :return:
    """

    # data path
    book = './annotation_files/_production/annotation2.xlsx'
    pos_path = './proposal_model/data/corpus/pos.set'
    neg_path = './proposal_model/data/corpus/neg.set'

    # load data
    with open(pos_path, 'rb') as f:
        pos_words = pickle.load(f)
    with open(neg_path, 'rb') as f:
        neg_words = pickle.load(f)

    # for w in pos_words:
    #     if w in neg_words:
    #         print("warning: the same word ", w, "in each dictionaries!")

    sheet_name = 'annotation2'
    df = pd.read_excel(book, sheet_name=sheet_name)
    with open('annotation_files/correct_emo_tag.txt', 'rb') as f:
        correct_emo_tags = pickle.load(f)

    # count correct emotion words
    contain_pos_dic = {}
    contain_neg_dic = {}
    pos_neg_text = 0
    correct = 0
    for index, line in df.iterrows():
        output_wakati = parse_ja_text(line['output'])
        if correct_emo_tags[index] == 'pos':
            pos_neg_text += 1
            for word in output_wakati:
                if word in pos_words:
                    correct += 1
                    if contain_pos_dic.get(word) is None:
                        contain_pos_dic[word] = 1
                    else:
                        contain_pos_dic[word] += 1
                    break
            # else:
            #     # 感情語彙が含まれていない応答のケース
            #     print(correct_emo_tags[index], output_wakati)
        elif correct_emo_tags[index] == 'neg':
            pos_neg_text += 1
            for word in output_wakati:
                if word in neg_words:
                    correct += 1
                    if contain_neg_dic.get(word) is None:
                        contain_neg_dic[word] = 1
                    else:
                        contain_neg_dic[word] += 1
                    break
            # else:
            #     # 感情語彙が含まれていない応答のケース
            #     print(correct_emo_tags[index], output_wakati)
        else:
            # if neu or none
            pass
    print('評価対象文（positive or negative）: ', pos_neg_text)
    print('正解数: ', correct)
    print('EmotionWord: ', float(correct / pos_neg_text), end='\n\n')
    print('----- positive words -----')
    for k, v in sorted(contain_pos_dic.items(), key=lambda x: -x[1]):
        print(str(k) + "," + str(v))
    print('----- negative words -----')
    for k, v in sorted(contain_neg_dic.items(), key=lambda x: -x[1]):
        print(str(k) + "," + str(v))


def check_corpus():
    """
    考察の章で使用するコーパス情報を取得する関数．
    ポジティブ文，ネガティブ文，ニュートラル文の個数を数える．
    :return:
    """
    from proposal_util import ProposalConvCorpus

    DATA_DIR = './proposal_model/data/corpus/'

    # call dictionary class
    corpus = ProposalConvCorpus(file_path=None)
    corpus.load(load_dir=DATA_DIR)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # check data
    emotion_freq = {'baseball': {'pos': 0, 'neg': 0, 'neu': 0}, 'pokemongo': {'pos': 0, 'neg': 0, 'neu': 0}}
    for num, input_sentence in enumerate(corpus.fine_posts):
        n_num = p_num = 0
        topic_label = ''
        for index, w_id in enumerate(corpus.fine_cmnts[num]):
            # comment の最初にトピックラベルが付いているものとする
            if index == 0:
                if w_id == corpus.dic.token2id['__label__0']:
                    topic_label = 'baseball'
                elif w_id == corpus.dic.token2id['__label__1']:
                    topic_label = 'pokemongo'
                else:
                    print('no label error: ', w_id)
                    raise ValueError
            # pos or neg word の判定
            else:
                if corpus.dic[w_id] in corpus.neg_words:
                    n_num += 1
                if corpus.dic[w_id] in corpus.pos_words:
                    p_num += 1
        if (n_num + p_num) == 0:
            emotion_freq[topic_label]['neu'] += 1
        elif n_num <= p_num:
            emotion_freq[topic_label]['pos'] += 1
        elif n_num > p_num:
            emotion_freq[topic_label]['neg'] += 1
        else:
            raise ValueError
        if num % 10000 == 0:
            print(num, 'end...')
    print(emotion_freq)
    for topic in emotion_freq:
        text = 0
        for tag in emotion_freq[topic]:
            text += emotion_freq[topic][tag]
        print(topic, ':', text)

    emotion_freq = {'pos': 0, 'neg': 0, 'neu': 0}
    for num, input_sentence in enumerate(corpus.rough_posts):
        n_num = p_num = 0
        for index, w_id in enumerate(corpus.rough_cmnts[num]):
            if corpus.dic[w_id] in corpus.neg_words:
                n_num += 1
            if corpus.dic[w_id] in corpus.pos_words:
                p_num += 1
        if (n_num + p_num) == 0:
            emotion_freq['neu'] += 1
        elif n_num <= p_num:
            emotion_freq['pos'] += 1
        elif n_num > p_num:
            emotion_freq['neg'] += 1
        else:
            raise ValueError
        if num % 100000 == 0:
            print(num, 'end...')

    print(emotion_freq)
    text = 0
    for tag in emotion_freq:
        text += emotion_freq[tag]
    print('fine texts: ', text)


def check_task2():
    """
    pos, neg, neu の分類失敗ケースの分析
    :return:
    """
    books2 = [
        # test
        # './annotation_files/_test/annotation2_tanaka.xlsx',
        # './annotation_files/_test/annotation2_isogawa.xlsx',
        # './annotation_files/_test/annotation2_miura.xlsx',
        # './annotation_files/_test/annotation2_takayama.xlsx',

        # production
        './annotation_files/_production/annotation2_maekawa.xlsx',
        './annotation_files/_production/annotation2_wakuta.xlsx',
        './annotation_files/_production/annotation2_sasaki.xlsx',
        './annotation_files/_production/annotation2_nagata.xlsx',
        './annotation_files/_production/annotation2_takebayashi.xlsx',
    ]

    # making data frames
    sheet_name = 'annotation2'
    data_frames = []
    for book in books2:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))
    with open('annotation_files/correct_emo_tag.txt', 'rb') as f:
        correct_emo_tags = pickle.load(f)
    print('正解タグ数：', len(correct_emo_tags))
    # tags = ['pos', 'neg', 'neu', 'none']         # none を考慮するケース
    tags = ['pos', 'neg', 'neu']                   # none を考慮しないケース

    # counting
    wrong_neg = {i: 0 for i in range(300)}  # pos だけど neg
    wrong_pos = {i: 0 for i in range(300)}  # neg だけど pos
    mat = np.zeros((3, 3))
    user_num = len(data_frames)             # アノテータ数
    for data_frame in data_frames:
        emotion_tag = 0                     # 感情制御の成功数
        text_num = 0                        # テストデータ数
        for index, line in data_frame.iterrows():
            # data が入っている場合のみカウント
            row = column = 0
            if isinstance(line['emotion_tag'], str) and line['emotion_tag'] in tags:
                text_num += 1
                if correct_emo_tags[index] == 'pos':
                    row = 0
                    if line['emotion_tag'] == 'pos':
                        column = 0
                    elif line['emotion_tag'] == 'neu':
                        column = 1
                    elif line['emotion_tag'] == 'neg':
                        column = 2
                        wrong_neg[index] += 1
                elif correct_emo_tags[index] == 'neu':
                    row = 1
                    if line['emotion_tag'] == 'pos':
                        column = 0
                    elif line['emotion_tag'] == 'neu':
                        column = 1
                    elif line['emotion_tag'] == 'neg':
                        column = 2
                elif correct_emo_tags[index] == 'neg':
                    row = 2
                    if line['emotion_tag'] == 'pos':
                        column = 0
                        wrong_pos[index] += 1
                    elif line['emotion_tag'] == 'neu':
                        column = 1
                    elif line['emotion_tag'] == 'neg':
                        column = 2
                mat[row, column] += 1
    print(mat)

    #call dictionary class
    from proposal_util import ProposalConvCorpus
    DATA_DIR = './proposal_model/data/corpus/'
    corpus = ProposalConvCorpus(file_path=None)
    corpus.load(load_dir=DATA_DIR)

    # show text
    for data_frame in data_frames:
        for index, line in data_frame.iterrows():
            if wrong_neg[index] >= 3:
                print(wrong_neg[index], ',', line['output'])
        break
    print('---------------------------------')
    for data_frame in data_frames:
        for index, line in data_frame.iterrows():
            if wrong_pos[index] >= 3:
                print(wrong_pos[index], ',', line['output'])
        break

    for data_frame in data_frames:
        for index, line in data_frame.iterrows():
            if wrong_neg[index] >= 3:
                nl = [word for word in line['output'].split(' ') if word in corpus.neg_words]
                pl = [word for word in line['output'].split(' ') if word in corpus.pos_words]
                print(wrong_neg[index], nl, pl)
        break
    print('---------------------------------')
    for data_frame in data_frames:
        for index, line in data_frame.iterrows():
            if wrong_pos[index] >= 3:
                nl = [word for word in line['output'].split(' ') if word in corpus.neg_words]
                pl = [word for word in line['output'].split(' ') if word in corpus.pos_words]
                print(wrong_pos[index], nl, pl)
        break

    # グラフの可視化
    import seaborn
    import matplotlib.pyplot as plt
    seaborn.set()
    # matplotlib.rc('font', family='sans-serif')
    plt.rcParams['font.family'] = 'IPAPGothic'
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
    seaborn.heatmap(mat, cmap=cmap, center=0, annot=True, fmt='g',
                    linewidths=0.5, xticklabels=['Positive', 'Neutral', 'Negative'],
                    yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.xlabel('評価者がアノテートしたラベル')
    plt.ylabel('学習時に利用した正解ラベル')
    plt.savefig('./annotation_files/heatmap.png')


if __name__ == '__main__':
    check_kappa1()
    check_kappa2()
    evaluate_task1()
    evaluate_task2()
    emotion_word()
    check_corpus()
    check_task2()