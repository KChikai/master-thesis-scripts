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
from nltk import word_tokenize
from chainer import serializers, cuda

# model
from existing_model.existing_seq2seq import Seq2Seq
from existing_model.util import ExistingConvCorpus
from proposal_model.external_seq2seq import MultiTaskSeq2Seq
from proposal_model.tuning_util import JaConvCorpus
from setting_param import FEATURE_NUM, HIDDEN_NUM, LABEL_NUM, LABEL_EMBED, TOPIC_NUM


# path info
PROPOSAL_DATA_DIR = './proposal_model/data/corpus/'
EXISTING_DATA_DIR = './existing_model/data/corpus/'
EXISTING_MODEL_PATH = './existing_model/data/99_third.model'
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

    # proposal model
    proposal_corpus = JaConvCorpus(file_path=None)
    proposal_corpus.load(load_dir=proposal_data_path)
    print('Vocabulary Size (number of words) :', len(proposal_corpus.dic.token2id))
    print('')
    proposal_model = MultiTaskSeq2Seq(all_vocab_size=len(proposal_corpus.dic.token2id),
                                      emotion_vocab_size=len(proposal_corpus.emotion_set),
                                      feature_num=args.feature_num, hidden_num=args.hidden_num,
                                      label_num=args.label_num, label_embed_num=args.label_embed,
                                      batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(proposal_model_path, proposal_model)

    # existing model
    existing_corpus = ExistingConvCorpus(file_path=None)
    existing_corpus.load(load_dir=existing_data_path)
    print('Vocabulary Size (number of words) :', len(existing_corpus.dic.token2id))
    print('')
    existing_model = Seq2Seq(len(existing_corpus.dic.token2id), feature_num=args.feature_num,
                             hidden_num=args.hidden_num, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(existing_model_path, existing_model)

    # run an interpreter
    data = []
    for num, input_sentence in enumerate(proposal_corpus.fine_posts):
        id_sequence = input_sentence.copy()
        input_sentence_rev = input_sentence[::-1]

        # make label lists TODO: 3値分類
        n_num = p_num = 0
        topic_label_id = correct_emo_label = -1
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
        if (n_num + p_num) == 0:
            correct_emo_label = 1
        elif n_num <= p_num:
            correct_emo_label = 2
        elif n_num > p_num:
            correct_emo_label = 0
        else:
            raise ValueError

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
        print("-> ", proposal_output)

        # generate an output (existing)
        input_words = [proposal_corpus.dic[w_id] for w_id in id_sequence]
        input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in input_words]
        input_vocab_rev = input_vocab[::-1]
        input_sentence = [existing_corpus.dic.token2id[word] for word in input_vocab if not existing_corpus.dic.token2id.get(word) is None]
        input_sentence_rev = [existing_corpus.dic.token2id[word] for word in input_vocab_rev if not existing_corpus.dic.token2id.get(word) is None]
        existing_model.initialize(batch_size=1)
        existing_output = existing_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 30,
                                                  word2id=existing_corpus.dic.token2id, id2word=existing_corpus.dic)
        print("-> ", existing_output)

        if topic_label_id == 0:
            data.append(['野球', input_text, existing_output, proposal_output, None, None, None, None, None, None])
        if topic_label_id == 1:
            data.append(['ポケモンGO', input_text, existing_output, proposal_output, None, None, None, None, None, None])

        print('')

        if num == 149:
            break

    # second leg
    for num, input_sentence in enumerate(proposal_corpus.fine_posts[::-1]):
        id_sequence = input_sentence.copy()
        input_sentence_rev = input_sentence[::-1]

        # make label lists TODO: 3値分類
        n_num = p_num = 0
        topic_label_id = correct_emo_label = -1
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
        if (n_num + p_num) == 0:
            correct_emo_label = 1
        elif n_num <= p_num:
            correct_emo_label = 2
        elif n_num > p_num:
            correct_emo_label = 0
        else:
            raise ValueError

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
        print("-> ", proposal_output)

        # generate an output (existing)
        input_words = [proposal_corpus.dic[w_id] for w_id in id_sequence]
        input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in input_words]
        input_vocab_rev = input_vocab[::-1]
        input_sentence = [existing_corpus.dic.token2id[word] for word in input_vocab if not existing_corpus.dic.token2id.get(word) is None]
        input_sentence_rev = [existing_corpus.dic.token2id[word] for word in input_vocab_rev if not existing_corpus.dic.token2id.get(word) is None]
        existing_model.initialize(batch_size=1)
        existing_output = existing_model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 30,
                                                  word2id=existing_corpus.dic.token2id, id2word=existing_corpus.dic)
        print("-> ", existing_output)

        if topic_label_id == 0:
            data.append(['野球', input_text, existing_output, proposal_output, None, None, None, None, None, None])
        if topic_label_id == 1:
            data.append(['ポケモンGO', input_text, existing_output, proposal_output, None, None, None, None, None, None])

        print('')

        if num == 149:
            break

    data_frame = pd.DataFrame(data, columns=['topic', 'input', 'output_A', 'output_B', 'fluency_A',
                                             'fluency_B', 'consistency_A', 'consistency_B', 'domain_consistency', 'emotion'])
    data_frame.to_csv('proposal_model/data/annotation1_utf-8.csv', encoding='utf-8')


def load_test():
    books = [
             'proposal_model/data/annotation1.xlsx',
             ]

    sheet_name = 'annotation1'
    data_flames = []
    for book in books:
        data_flames.append(pd.read_excel(book, sheetname=sheet_name))

    for data_flame in data_flames:
        # while len(data_flame.columns) > 6:
        #     data_flame.drop(data_flame.columns[len(data_flame.columns) - 1], inplace=True, axis=1)
        data_flame.columns = ['topic', 'input', 'output_A', 'output_B', 'fluency_A',
                              'fluency_B', 'consistency_A', 'consistency_B', 'domain_consistency', 'emotion']

        post_df = data_flame.ix[:, ['output_A']]
        for _, line in post_df.iterrows():
            print(line['output_A'])


if __name__ == '__main__':
    # test_run(existing_data_path=EXISTING_DATA_DIR, existing_model_path=EXISTING_MODEL_PATH,
    #          proposal_data_path=PROPOSAL_DATA_DIR, proposal_model_path=PROPOSAL_MODEL_PATH)
    load_test()