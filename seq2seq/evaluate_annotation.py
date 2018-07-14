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
    stat_data = []
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

        stat_data.append([float(fluency[0] / text_num), float(fluency[1] / text_num),
                          float(consistency[0] / text_num), float(consistency[1] / text_num),
                          float(domain_consistency / text_num), 1 - float(domain_consistency / text_num),
                          float(emotion / text_num), 1 - float(emotion / text_num)])

    # for index, individual in enumerate(graph_data):
    #     print(individual)
    import scipy.stats
    df = pd.DataFrame(stat_data, columns=['fluency_A', 'fluency_B',
                                          'consistency_A', 'consistency_B',
                                          'domain_consistency_A', 'domain_consistency_B',
                                          'emotion_A', 'emotion_B'])

    t, p = scipy.stats.ttest_rel(df['fluency_A'], df['fluency_B'])
    print(t, p)
    t, p = scipy.stats.ttest_rel(df['consistency_A'], df['consistency_B'])
    print(t, p)
    t, p = scipy.stats.ttest_rel(df['domain_consistency_A'], df['domain_consistency_B'])
    print(t, p)
    t, p = scipy.stats.ttest_rel(df['emotion_A'], df['emotion_B'])
    print(t, p, end='\n\n')

    t, p = scipy.stats.ttest_ind(df['fluency_A'], df['fluency_B'], equal_var=False)
    print(t, p)
    t, p = scipy.stats.ttest_ind(df['consistency_A'], df['consistency_B'], equal_var=False)
    print(t, p)
    t, p = scipy.stats.ttest_ind(df['domain_consistency_A'], df['domain_consistency_B'], equal_var=False)
    print(t, p)
    t, p = scipy.stats.ttest_ind(df['emotion_A'], df['emotion_B'], equal_var=False)
    print(t, p, end='\n\n')

    # df.to_excel('annotation_files/result.xlsx', encoding='utf-8')

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
        print('text num:', text_num, 'emotion tag (solo):', float(emotion_tag / text_num))
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
    ヒートマップの作成・失敗例と感情語彙の表示を行う
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

    # #call dictionary class
    # from proposal_util import ProposalConvCorpus
    # DATA_DIR = './proposal_model/data/corpus/'
    # corpus = ProposalConvCorpus(file_path=None)
    # corpus.load(load_dir=DATA_DIR)
    #
    # # show text
    # for data_frame in data_frames:
    #     for index, line in data_frame.iterrows():
    #         if wrong_neg[index] >= 3:
    #             print(wrong_neg[index], ',', line['output'])
    #     break
    # print('---------------------------------')
    # for data_frame in data_frames:
    #     for index, line in data_frame.iterrows():
    #         if wrong_pos[index] >= 3:
    #             print(wrong_pos[index], ',', line['output'])
    #     break
    #
    # for data_frame in data_frames:
    #     for index, line in data_frame.iterrows():
    #         if wrong_neg[index] >= 3:
    #             nl = [word for word in line['output'].split(' ') if word in corpus.neg_words]
    #             pl = [word for word in line['output'].split(' ') if word in corpus.pos_words]
    #             print(wrong_neg[index], nl, pl)
    #     break
    # print('---------------------------------')
    # for data_frame in data_frames:
    #     for index, line in data_frame.iterrows():
    #         if wrong_pos[index] >= 3:
    #             nl = [word for word in line['output'].split(' ') if word in corpus.neg_words]
    #             pl = [word for word in line['output'].split(' ') if word in corpus.pos_words]
    #             print(wrong_pos[index], nl, pl)
    #     break

    # グラフの可視化
    import seaborn
    import matplotlib.pyplot as plt
    seaborn.set(font_scale=2.0)
    # matplotlib.rc('font', family='sans-serif')
    plt.rcParams['font.family'] = 'IPAPGothic'
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

    mat_prob = np.zeros(mat.shape)
    for row in range(mat.shape[0]):
        mat_prob[row, :] = mat[row, :] / np.sum(mat[row, :])

    ax = seaborn.heatmap(mat_prob, cmap=cmap, center=0, annot=True, fmt='.01%', vmin=0, vmax=0.75,
                         linewidths=0.5, xticklabels=['Positive', 'Neutral', 'Negative'],
                         yticklabels=['Positive', 'Neutral', 'Negative'])
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .50, .75])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%'])
    # plt.xlabel('評価者がアノテートしたラベル')
    # plt.ylabel('学習時に利用した正解ラベル')
    plt.xlabel("Annotators' judgments")
    plt.ylabel('True sentiments')
    # plt.savefig('./annotation_files/heatmap.png')
    plt.savefig('./annotation_files/heatmap_en.png', bbox_inches="tight")


def check_length():
    """
    ACLの追加項目（平均文書長を求める）
    :return:
    """
    books = [
        './annotation_files/_production/annotation1.xlsx',
    ]
    with open('annotation_files/swap_keys.pkl', 'rb') as f:
        swap_keys = pickle.load(f)

    # making data frames
    sheet_name = 'annotation1'
    data_frames = []
    for book in books:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))

    text_num = 0
    existing_len = 0
    existing_hensa = []
    proposal_len = 0
    proposal_hensa = []
    for data_frame in data_frames:
        for index, line in data_frame.iterrows():
            # data が入っている場合のみカウント
            if isinstance(line['output_A'], str):
                text_num += 1
                if swap_keys[index]:
                    # swap している場合（A: proposal, B: existing）
                    proposal_len += len(line['output_A'])
                    proposal_hensa.append(len(line['output_A']))
                    existing_len += len(line['output_B'])
                    existing_hensa.append(len(line['output_B']))
                else:
                    # swap していない場合（A: existing, B: proposal）
                    existing_len += len(line['output_A'])
                    existing_hensa.append(len(line['output_A']))
                    proposal_len += len(line['output_B'])
                    proposal_hensa.append(len(line['output_B']))
    print("テキスト：", text_num)
    print("既存手法：", float(existing_len / text_num), np.std(existing_hensa))
    print("提案手法：", float(proposal_len / text_num), np.std(proposal_hensa))


def domain_and_sentiment_richness():
    """
    ドメインと感情両方で提案手法が選ばれた割合と件数
    :return:
    """
    # load result data
    books1 = [
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
    confidence = [0] * len(data_frames)
    exist_confidence = [0] * len(data_frames)
    results = [0] * 300
    exist_results = [0] * 300
    all_domain_and_sentiment = all_exist_domain_and_sentiment = 0
    user_num = len(data_frames)      # アノテータ数
    for user_index, data_frame in enumerate(data_frames):

        text_num = 0                 # テストデータ数
        domain_consistency = 0       # 会話ドメイン整合性の観点で提案手法が選択された数
        emotion = 0                  # 感情の豊かさで提案手法が選択された数
        domain_and_sentiment = 0     # ドメインと感情が両方発現している（既存手法よりも選ばれているケース）
        exist_domain_and_sentiment = 0

        for index, line in data_frame.iterrows():
            # data が入っている場合のみカウント
            if not np.isnan(line['fluency_A']):
                text_num += 1
                if swap_keys[index]:
                    # swap している場合（A: proposal, B: existing）
                    if line['domain_consistency'] == 'a' and line['emotion'] == 'a':
                        domain_and_sentiment += 1
                        results[index] += 1
                    if line['domain_consistency'] == 'b' and line['emotion'] == 'b':
                        exist_domain_and_sentiment += 1
                        exist_results[index] += 1
                else:
                    # swap していない場合（A: existing, B: proposal）
                    if line['domain_consistency'] == 'b' and line['emotion'] == 'b':
                        domain_and_sentiment += 1
                        results[index] += 1
                    if line['domain_consistency'] == 'a' and line['emotion'] == 'a':
                        exist_domain_and_sentiment += 1
                        exist_results[index] += 1
        print('user index:', user_index)
        print('ユーザが両方提案手法を選択した応答文数（300件中）:', domain_and_sentiment)
        print('ユーザが両方既存手法を選択した応答文数（300件中）:', exist_domain_and_sentiment)

        all_domain_and_sentiment += float(domain_and_sentiment / text_num)
        all_exist_domain_and_sentiment += float(exist_domain_and_sentiment / text_num)

        confidence[user_index] += float(domain_and_sentiment / text_num)
        exist_confidence[user_index] += float(exist_domain_and_sentiment / text_num)

    # overall results
    all_correct = all_exist_correct = 0
    for correct in results:
        if correct >= -(-user_num // 2):
            all_correct += 1

    for exist_correct in exist_results:
        if exist_correct >= -(-user_num // 2):
            all_exist_correct += 1

    print('両方提案手法が選ばれたケース（5人のアノテータの過半数が選んだ応答のみ考慮）：',
          all_correct, float(all_correct / len(results)))
    print('両方既存手法が選ばれたケース（5人のアノテータの過半数が選んだ応答のみ考慮）：',
          all_exist_correct, float(all_exist_correct / len(results)))
    print('両方ユーザが両方の評価指標で提案手法を選んだ応答文数の平均値：',
          float(all_domain_and_sentiment / user_num))
    print('両方ユーザが両方の評価指標で既存手法を選んだ応答文数の平均値：',
          float(all_exist_domain_and_sentiment / user_num))

    for _ in confidence:
        print(_)
    for _ in exist_confidence:
        print(_)


if __name__ == '__main__':
    # check_kappa1()
    # check_kappa2()
    # evaluate_task1()
    # evaluate_task2()
    # emotion_word()
    # check_corpus()
    # check_task2()
    # check_length()
    domain_and_sentiment_richness()