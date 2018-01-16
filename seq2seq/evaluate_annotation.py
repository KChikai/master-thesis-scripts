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


def fleiss_kappa(M):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects
              and `k` is the number of categories into which assignments are made.
              `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """
    N, k = M.shape                         # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)

    return kappa


def check_kappa():
    """
    fleiss's kappa を計算する
    :return:
    """
    # load some data
    books1 = [
        # annotation1
        './annotation_files/_test/annotation1_tanaka.xlsx',
        './annotation_files/_test/annotation1_isogawa.xlsx',
        './annotation_files/_test/annotation1_miura.xlsx',
        './annotation_files/_test/annotation1_takayama.xlsx',
    ]
    with open('annotation_files/swap_keys.pkl', 'rb') as f:
        swap_keys = pickle.load(f)

    # making data frames
    sheet_name = 'annotation1'
    data_frames = []
    for book in books1:
        data_frames.append(pd.read_excel(book, sheet_name=sheet_name))

    # making matrix
    # mat = np.zeros((50, 6))
    # for data_frame in data_frames:
    #     for index, line in data_frame.iterrows():
    #         if swap_keys[index]:
    #             # swap している場合（A: proposal, B: existing）
    #             row_order = [line['fluency_B'], line['fluency_A'], line['consistency_B'], line['consistency_A']]
    #             for column in range(mat.shape[1] - 2):
    #                 mat[index, column] += row_order[column]
    #
    #             if line['domain_consistency'] == 'a':
    #                 mat[index, len(row_order)] += 1
    #             else:
    #                 # none case
    #                 pass
    #             if line['emotion'] == 'a':
    #                 mat[index, len(row_order) + 1] += 1
    #             else:
    #                 pass
    #         else:
    #             # swap していない場合（A: existing, B: proposal）
    #             row_order = [line['fluency_A'], line['fluency_B'], line['consistency_A'], line['consistency_B']]
    #             for column in range(mat.shape[1] - 2):
    #                 mat[index, column] += row_order[column]
    #
    #             if line['domain_consistency'] == 'b':
    #                 mat[index, len(row_order)] += 1
    #             else:
    #                 # none case
    #                 pass
    #             if line['emotion'] == 'b':
    #                 mat[index, len(row_order) + 1] += 1
    #             else:
    #                 pass
    #
    #         if index == 49:
    #             break
    # print(mat)
    # result_value = fleiss_kappa(M=mat)
    # print('fleiss kappa:', result_value)

    # make a matrix
    mat = [[] for i in range(6)]
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
            if item_index == 49:
                break

    # show fleiss's kappa for each evaluation metrics
    for m in mat:
        task = nltk.AnnotationTask(data=m)
        print('fleiss kappa:', task.multi_kappa())
        # print(task.avg_Ao())
        # print(task.kappa())


def evaluate_task1():
    """
    アノテートされたデータを評価する関数
    :return:
    """
    books1 = [
        # annotation1
        './annotation_files/_test/annotation1_tanaka.xlsx',
        './annotation_files/_test/annotation1_isogawa.xlsx',
        './annotation_files/_test/annotation1_miura.xlsx',
        './annotation_files/_test/annotation1_takayama.xlsx',
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
    all_fluency = [0, 0]  # fluency[0]: 既存手法の成功数, fluency[1]: 提案手法の成功数
    all_consistency = [0, 0]  # consistency[0]: 既存手法の成功数, consistency[1]: 提案手法の成功数
    all_domain_consistency = 0  # 会話ドメイン整合性の観点で提案手法が選択された数
    all_emotion = 0  # 感情の豊かさで提案手法が選択された数
    user_num = len(data_frames)  # アノテータ数
    for data_frame in data_frames:

        text_num = 0  # テストデータ数
        fluency = [0, 0]  # fluency[0]: 既存手法の成功数, fluency[1]: 提案手法の成功数
        consistency = [0, 0]  # consistency[0]: 既存手法の成功数, consistency[1]: 提案手法の成功数
        domain_consistency = 0  # 会話ドメイン整合性の観点で提案手法が選択された数
        emotion = 0  # 感情の豊かさで提案手法が選択された数
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
    import matplotlib
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    font = {'family': 'Osaka'}
    matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 10})
    default_x = [i for i in range(1, 7)]
    peaple = ['田中', '五十川', '三浦', '高山']
    colors = ['b', 'g', 'r', 'y']
    width = 0.2
    for index, individual in enumerate(graph_data):
        left = [i + (width * index) for i in default_x]
        plt.bar(left=left, height=np.array(individual), width=width, label=peaple[index], color=colors[index],
                align='center')
    plt.legend(loc="best")
    plt.xticks([(i + (i + width * len(peaple))) / 2 - 0.1 for i in default_x],
               ['fluency_A', 'fluency_B', 'consistency_A',
                'consistency_B', 'domain_consistency', 'emotion'])
    plt.show()

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
        # annotation2
        './annotation_files/_test/annotation2_tanaka.xlsx',
        './annotation_files/_test/annotation2_isogawa.xlsx',
        './annotation_files/_test/annotation2_miura.xlsx',
        './annotation_files/_test/annotation2_takayama.xlsx',
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
    import matplotlib
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    font = {'family': 'Osaka'}
    matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 10})
    default_x = [i for i in range(1, 2)]
    peaple = ['田中', '五十川', '三浦', '高山']
    colors = ['b', 'g', 'r', 'y']
    width = 0.1
    for index, individual in enumerate(graph_data):
        left = [i + (width * index) for i in default_x]
        plt.bar(left=left, height=np.array(individual), width=width, label=peaple[index], color=colors[index],
                align='center')
    plt.legend(loc="best")
    plt.xticks([(i + (i + width * len(peaple))) / 2 - 0.1 for i in default_x], ['emotion_tag'])
    plt.show()


if __name__ == '__main__':
    check_kappa()
    # evaluate_task1()
    # evaluate_task2()
