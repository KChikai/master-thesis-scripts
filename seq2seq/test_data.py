# -*- coding:utf-8 -*-

"""
既存手法と提案手法からcsvファイルを出力するスクリプト
"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import unicodedata
import pickle
import numpy as np
import pandas as pd
from chainer import serializers

# model
from existing_model.existing_seq2seq import Seq2Seq
from existing_model.existing_util import ExistingConvCorpus
from proposal_model.proposal_util import ProposalConvCorpus
from proposal_model.external_seq2seq import MultiTaskSeq2Seq
from setting_param import FEATURE_NUM, HIDDEN_NUM, LABEL_NUM, LABEL_EMBED

# path info
PROPOSAL_DATA_DIR = './proposal_model/data/corpus/'
EXISTING_DATA_DIR = './existing_model/data/corpus/'
EXISTING_MODEL_PATH = './existing_model/data/10.model'
PROPOSAL_MODEL_PATH = './proposal_model/data/109.model'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--feature_num', '-f', default=FEATURE_NUM, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=HIDDEN_NUM, type=int, help='dimension of hidden layer')
parser.add_argument('--label_num', '-ln', default=LABEL_NUM, type=int, help='dimension of label layer')
parser.add_argument('--label_embed', '-le', default=LABEL_EMBED, type=int, help='dimension of label embed layer')
parser.add_argument('--bar', '-b', default='0', type=int, help='whether to show the graph of loss values or not')
parser.add_argument('--beam_search', '-be', default=True, type=bool, help='show results using beam search')
args = parser.parse_args()


def parse_ja_text(text):
    """
    Function to parse Japanese text.
    :param text: string: sentence written by Japanese
    :return: list: parsed text
    """
    import MeCab
    mecab = MeCab.Tagger("mecabrc")
    mecab.parse('')

    # list up noun
    mecab_result = mecab.parseToNode(text)
    parse_list = []
    while mecab_result is not None:
        if mecab_result.surface != "":
            parse_list.append(unicodedata.normalize('NFKC', mecab_result.surface).lower())
        mecab_result = mecab_result.next

    return parse_list


def test_run(existing_data_path, existing_model_path, proposal_data_path, proposal_model_path):
    """
    データをモデルに入力してcsv file化（アノテートファイル）にする
    :return:
    """
    #######################
    # load proposal model #
    #######################
    proposal_corpus = ProposalConvCorpus(file_path=None)
    proposal_corpus.load(load_dir=proposal_data_path)
    print('Vocabulary Size (number of words) :', len(proposal_corpus.dic.token2id))
    print('')
    proposal_model = MultiTaskSeq2Seq(all_vocab_size=len(proposal_corpus.dic.token2id),
                                      emotion_vocab_size=len(proposal_corpus.emotion_set),
                                      feature_num=args.feature_num, hidden_num=args.hidden_num,
                                      label_num=args.label_num, label_embed_num=args.label_embed,
                                      batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(proposal_model_path, proposal_model)

    #######################
    # load existing model #
    #######################
    existing_corpus = ExistingConvCorpus(file_path=None)
    existing_corpus.load(load_dir=existing_data_path)
    print('Vocabulary Size (number of words) :', len(existing_corpus.dic.token2id))
    print('')
    # existing_model = Seq2Seq(len(existing_corpus.dic.token2id), feature_num=args.feature_num,
    #                          hidden_num=args.hidden_num, batch_size=1, gpu_flg=args.gpu)
    existing_model = Seq2Seq(all_vocab_size=len(existing_corpus.dic.token2id),
                             emotion_vocab_size=len(existing_corpus.emotion_set),
                             feature_num=args.feature_num, hidden_num=args.hidden_num,
                             label_num=args.label_num, label_embed_num=args.label_embed,
                             batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(existing_model_path, existing_model)

    ######################
    # run an interpreter #
    ######################
    text_num = 0
    baseball_line = 0
    data = []
    for num, input_sentence in enumerate(proposal_corpus.fine_posts):
        id_sequence = input_sentence.copy()
        input_sentence_rev = input_sentence[::-1]

        word_list = [proposal_corpus.dic[w_id] for w_id in id_sequence]
        if "<unk>" not in word_list and len(word_list) < 20:

            # make label lists TODO: 3値分類
            n_num = p_num = 0
            topic_label_id = -1
            for index, w_id in enumerate(proposal_corpus.fine_cmnts[num]):
                # comment の最初にトピックラベルが付いているものとする
                if index == 0:
                    if w_id == proposal_corpus.dic.token2id['__label__0']:
                        topic_label_id = 0
                    elif w_id == proposal_corpus.dic.token2id['__label__1']:
                        topic_label_id = 1
                    else:
                        print('no label error: ', w_id)
                        raise ValueError
                # pos or neg word の判定
                else:
                    if proposal_corpus.dic[w_id] in proposal_corpus.neg_words:
                        n_num += 1
                    if proposal_corpus.dic[w_id] in proposal_corpus.pos_words:
                        p_num += 1

            # generate an output (proposal)
            neg_output = ''
            neu_output = ''
            pos_output = ''
            input_text = " ".join([proposal_corpus.dic[w_id] for w_id in id_sequence])
            print("input : ", input_text)
            for emo_label in range(LABEL_NUM):
                proposal_model.initialize(batch_size=1)
                sentence = proposal_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                                   emo_label_id=emo_label, topic_label_id=topic_label_id,
                                                   word2id=proposal_corpus.dic.token2id, id2word=proposal_corpus.dic)
                if emo_label == 0:
                    # print("neg -> ", sentence)
                    neg_output = sentence
                elif emo_label == 1:
                    # print("neu -> ", sentence)
                    neu_output = sentence
                else:
                    # print("pos -> ", sentence)
                    pos_output = sentence

            # choose pos or neg output
            th = np.random.randint(1, 11)
            if th <= 5 and neg_output != '':
                proposal_output = neg_output
            elif th > 5 and pos_output != '':
                proposal_output = pos_output
            else:
                raise ValueError
            print("proposal -> ", proposal_output)

            # generate an output (existing)
            input_words = [proposal_corpus.dic[w_id] for w_id in id_sequence]
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in input_words]
            input_vocab_rev = input_vocab[::-1]
            input_sentence = [existing_corpus.dic.token2id[word] for word in input_vocab if not existing_corpus.dic.token2id.get(word) is None]
            input_sentence_rev = [existing_corpus.dic.token2id[word] for word in input_vocab_rev if not existing_corpus.dic.token2id.get(word) is None]
            existing_model.initialize(batch_size=1)
            # existing_output = existing_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 30,
            #                                           word2id=existing_corpus.dic.token2id, id2word=existing_corpus.dic)
            existing_output = existing_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                                      label_id=-1, word2id=existing_corpus.dic.token2id, id2word=existing_corpus.dic)
            print("existing -> ", existing_output)

            if topic_label_id == 0:
                data.append(['野球', input_text, existing_output, proposal_output, None, None, None, None, None, None])
                text_num += 1
            if topic_label_id == 1:
                data.append(['ポケモンGO', input_text, existing_output, proposal_output, None, None, None, None, None, None])
                text_num += 1

            print('')

            if text_num == 150:
                baseball_line = num
                break

    ##############
    # second leg #
    ##############
    text_num = 0
    pokemongo_line = 0
    for num, input_sentence in enumerate(proposal_corpus.fine_posts[::-1]):
        id_sequence = input_sentence.copy()
        input_sentence_rev = input_sentence[::-1]

        word_list = [proposal_corpus.dic[w_id] for w_id in id_sequence]
        if "<unk>" not in word_list and len(word_list) < 20:

            # make label lists TODO: 3値分類
            n_num = p_num = 0
            topic_label_id = -1
            for index, w_id in enumerate(proposal_corpus.fine_cmnts[::-1][num]):
                # comment の最初にトピックラベルが付いているものとする
                if index == 0:
                    if w_id == proposal_corpus.dic.token2id['__label__0']:
                        topic_label_id = 0
                    elif w_id == proposal_corpus.dic.token2id['__label__1']:
                        topic_label_id = 1
                    else:
                        print('no label error: ', w_id)
                        raise ValueError
                # pos or neg word の判定
                else:
                    if proposal_corpus.dic[w_id] in proposal_corpus.neg_words:
                        n_num += 1
                    if proposal_corpus.dic[w_id] in proposal_corpus.pos_words:
                        p_num += 1

            # generate an output (proposal)
            neg_output = ''
            neu_output = ''
            pos_output = ''
            input_text = " ".join([proposal_corpus.dic[w_id] for w_id in id_sequence])
            print("input : ", input_text)
            for emo_label in range(LABEL_NUM):
                proposal_model.initialize(batch_size=1)
                sentence = proposal_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                                   emo_label_id=emo_label, topic_label_id=topic_label_id,
                                                   word2id=proposal_corpus.dic.token2id, id2word=proposal_corpus.dic)

                if emo_label == 0:
                    # print("neg -> ", sentence)
                    neg_output = sentence
                elif emo_label == 1:
                    # print("neu -> ", sentence)
                    neu_output = sentence
                else:
                    # print("pos -> ", sentence)
                    pos_output = sentence

            # choose pos or neg output
            th = np.random.randint(1, 11)
            if th <= 5 and neg_output != '':
                proposal_output = neg_output
            elif th > 5 and pos_output != '':
                proposal_output = pos_output
            else:
                raise ValueError
            print("proposal -> ", proposal_output)

            # generate an output (existing)
            input_words = [proposal_corpus.dic[w_id] for w_id in id_sequence]
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in input_words]
            input_vocab_rev = input_vocab[::-1]
            input_sentence = [existing_corpus.dic.token2id[word] for word in input_vocab if not existing_corpus.dic.token2id.get(word) is None]
            input_sentence_rev = [existing_corpus.dic.token2id[word] for word in input_vocab_rev if not existing_corpus.dic.token2id.get(word) is None]
            existing_model.initialize(batch_size=1)
            # existing_output = existing_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 30,
            #                                           word2id=existing_corpus.dic.token2id, id2word=existing_corpus.dic)
            existing_output = existing_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                                      label_id=-1, word2id=existing_corpus.dic.token2id, id2word=existing_corpus.dic)
            print("existing -> ", existing_output)

            if topic_label_id == 0:
                data.append(['野球', input_text, existing_output, proposal_output, None, None, None, None, None, None])
                text_num += 1
            if topic_label_id == 1:
                data.append(['ポケモンGO', input_text, existing_output, proposal_output, None, None, None, None, None, None])
                text_num += 1

            print('')

            if text_num == 150:
                break

    ###########################
    # save data as a csv file #
    ###########################
    data_frame = pd.DataFrame(data, columns=['topic', 'input', 'output_A', 'output_B', 'fluency_A',
                                             'fluency_B', 'consistency_A', 'consistency_B', 'domain_consistency', 'emotion'])
    data_frame.to_csv('annotation_files/annotation1_utf-8.csv', encoding='utf-8')


    #############
    # third leg #
    #############
    text_num = 0
    correct_emo_tags = []
    data = []
    for num, input_sentence in enumerate(proposal_corpus.fine_posts):
        if num > baseball_line:
            id_sequence = input_sentence.copy()
            input_sentence_rev = input_sentence[::-1]

            word_list = [proposal_corpus.dic[w_id] for w_id in id_sequence]
            if "<unk>" not in word_list and len(word_list) < 20:

                # make label lists TODO: 3値分類
                n_num = p_num = 0
                topic_label_id = -1
                for index, w_id in enumerate(proposal_corpus.fine_cmnts[num]):
                    # comment の最初にトピックラベルが付いているものとする
                    if index == 0:
                        if w_id == proposal_corpus.dic.token2id['__label__0']:
                            topic_label_id = 0
                        elif w_id == proposal_corpus.dic.token2id['__label__1']:
                            topic_label_id = 1
                        else:
                            print('no label error: ', w_id)
                            raise ValueError
                    # pos or neg word の判定
                    else:
                        if proposal_corpus.dic[w_id] in proposal_corpus.neg_words:
                            n_num += 1
                        if proposal_corpus.dic[w_id] in proposal_corpus.pos_words:
                            p_num += 1

                # generate an output (proposal)
                neg_output = ''
                neu_output = ''
                pos_output = ''
                input_text = " ".join([proposal_corpus.dic[w_id] for w_id in id_sequence])
                print("input : ", input_text)
                for emo_label in range(LABEL_NUM):
                    proposal_model.initialize(batch_size=1)
                    sentence = proposal_model.generate(input_sentence, input_sentence_rev,
                                                       sentence_limit=len(input_sentence) + 20,
                                                       emo_label_id=emo_label, topic_label_id=topic_label_id,
                                                       word2id=proposal_corpus.dic.token2id, id2word=proposal_corpus.dic)
                    if emo_label == 0:
                        neg_output = sentence
                    elif emo_label == 1:
                        neu_output = sentence
                    else:
                        pos_output = sentence

                # choose pos or neg output
                th = np.random.randint(1, 30)
                if th <= 8 and neg_output != '':
                    proposal_output = neg_output
                    correct_emo_tags.append('neg')
                elif th >= 18 and pos_output != '':
                    proposal_output = pos_output
                    correct_emo_tags.append('pos')
                else:
                    proposal_output = neu_output
                    correct_emo_tags.append('neu')
                print("-> ", proposal_output)
                if topic_label_id == 0:
                    data.append(['野球', input_text, proposal_output, None])
                    text_num += 1
                if topic_label_id == 1:
                    data.append(['ポケモンGO', input_text, proposal_output, None])
                    text_num += 1
                print('')
                if text_num == 150:
                    break

    #############
    # forth leg #
    #############
    text_num = 0
    for num, input_sentence in enumerate(proposal_corpus.fine_posts[::-1]):
        if num > pokemongo_line:
            id_sequence = input_sentence.copy()
            input_sentence_rev = input_sentence[::-1]

            word_list = [proposal_corpus.dic[w_id] for w_id in id_sequence]
            if "<unk>" not in word_list and len(word_list) < 20:

                # make label lists TODO: 3値分類
                n_num = p_num = 0
                topic_label_id = -1
                for index, w_id in enumerate(proposal_corpus.fine_cmnts[::-1][num]):
                    # comment の最初にトピックラベルが付いているものとする
                    if index == 0:
                        if w_id == proposal_corpus.dic.token2id['__label__0']:
                            topic_label_id = 0
                        elif w_id == proposal_corpus.dic.token2id['__label__1']:
                            topic_label_id = 1
                        else:
                            print('no label error: ', w_id)
                            raise ValueError
                    # pos or neg word の判定
                    else:
                        if proposal_corpus.dic[w_id] in proposal_corpus.neg_words:
                            n_num += 1
                        if proposal_corpus.dic[w_id] in proposal_corpus.pos_words:
                            p_num += 1

                # generate an output (proposal)
                neg_output = ''
                neu_output = ''
                pos_output = ''
                input_text = " ".join([proposal_corpus.dic[w_id] for w_id in id_sequence])
                print("input : ", input_text)
                for emo_label in range(LABEL_NUM):
                    proposal_model.initialize(batch_size=1)
                    sentence = proposal_model.generate(input_sentence, input_sentence_rev,
                                                       sentence_limit=len(input_sentence) + 20,
                                                       emo_label_id=emo_label, topic_label_id=topic_label_id,
                                                       word2id=proposal_corpus.dic.token2id, id2word=proposal_corpus.dic)
                    if emo_label == 0:
                        neg_output = sentence
                    elif emo_label == 1:
                        neu_output = sentence
                    else:
                        pos_output = sentence

                # choose pos or neg output
                th = np.random.randint(1, 30)
                if th <= 8 and neg_output != '':
                    proposal_output = neg_output
                    correct_emo_tags.append('neg')
                elif th >= 18 and pos_output != '':
                    proposal_output = pos_output
                    correct_emo_tags.append('pos')
                else:
                    proposal_output = neu_output
                    correct_emo_tags.append('neu')
                print("-> ", proposal_output)
                if topic_label_id == 0:
                    data.append(['野球', input_text, proposal_output, None])
                    text_num += 1
                if topic_label_id == 1:
                    data.append(['ポケモンGO', input_text, proposal_output, None])
                    text_num += 1
                print('')
                if text_num == 150:
                    break

    ###########################
    # save data as a csv file #
    ###########################
    with open('annotation_files/correct_emo_tag.txt', 'wb') as f:
        pickle.dump(correct_emo_tags, f)
    data_frame = pd.DataFrame(data, columns=['topic', 'input', 'output', 'emotion_tag'])
    data_frame.to_csv('annotation_files/annotation2_utf-8.csv', encoding='utf-8')


def load_test():
    """
    annotation files がロードできるかのテスト関数
    :return:
    """
    # books = [
    #          'annotation_files//annotation1.xlsx',
    #          ]
    #
    # sheet_name = 'annotation1'
    # data_flames = []
    # for book in books:
    #     data_flames.append(pd.read_excel(book, sheetname=sheet_name))
    #
    # for data_flame in data_flames:
    #     # while len(data_flame.columns) > 6:
    #     #     data_flame.drop(data_flame.columns[len(data_flame.columns) - 1], inplace=True, axis=1)
    #     data_flame.columns = ['topic', 'input', 'output_A', 'output_B', 'fluency_A',
    #                           'fluency_B', 'consistency_A', 'consistency_B', 'domain_consistency', 'emotion']
    #
    #     post_df = data_flame.ix[:, ['output_A']]
    #     for _, line in post_df.iterrows():
    #         print(line['output_A'])

    with open('annotation_files/correct_emo_tag.txt', 'rb') as f:
        correct_emo_tags = pickle.load(f)
    print(len(correct_emo_tags), correct_emo_tags)


def swap_output(swap=True):
    """
    出力をランダムスワップするスクリプト
    それを復元する鍵も作成
    :return:
    """
    if swap:
        previous_file = './annotation_files/annotation1.xlsx'
        save_file = './annotation_files/annotation1_swap.xlsx'

        sheet_name = 'annotation1'
        data_frame = pd.read_excel(previous_file, sheet_name=sheet_name)
        # data_flame.columns = ['topic', 'input', 'output_A', 'output_B', 'fluency_A',
        #                       'fluency_B', 'consistency_A', 'consistency_B', 'domain_consistency', 'emotion']
        swap_keys = []
        new_frame = data_frame.copy()
        for index, line in new_frame.ix[:, :].iterrows():
            binary = np.random.randint(0, 2)
            swap_keys.append(binary)
            if binary:
                a_output = line["output_A"]
                b_output = line["output_B"]
                # line["output_A"] = b_output
                # line["output_B"] = a_output
                new_frame.ix[[index], ["output_A"]] = b_output
                new_frame.ix[[index], ["output_B"]] = a_output

        # save a file
        writer = pd.ExcelWriter(save_file)
        new_frame.to_excel(writer, sheet_name)

        with open('annotation_files/swap_keys.pkl', 'wb') as f:
            pickle.dump(swap_keys, f)
        for k in swap_keys:
            print(k)
    else:
        pass


if __name__ == '__main__':
    # test_run(existing_data_path=EXISTING_DATA_DIR, existing_model_path=EXISTING_MODEL_PATH,
    #          proposal_data_path=PROPOSAL_DATA_DIR, proposal_model_path=PROPOSAL_MODEL_PATH)
    # load_test()
    # swap_output(swap=True)
    pass