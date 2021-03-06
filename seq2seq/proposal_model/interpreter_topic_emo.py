# -*- coding:utf-8 -*-

"""
Topic label + external memory 確認用
ラベルをデコード部分に入れる（emotion embedding, speaker model）
入力文の後に半角＋数字（トピック）＋半角＋数字（感情）を入力することでラベルを挿入する

ex)
こんにちは，今日もいい天気ですね！ 0 2

"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import unicodedata
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from chainer import serializers, cuda
from proposal_util import ConvCorpus, ProposalConvCorpus
from external_seq2seq import MultiTaskSeq2Seq
from setting_param import FEATURE_NUM, HIDDEN_NUM, LABEL_NUM, LABEL_EMBED, TOPIC_NUM


# path info
DATA_DIR = './data/corpus/'
MODEL_PATH = './data/109.model'
TRAIN_LOSS_PATH = './data/loss_train_data.pkl'
TEST_LOSS_PATH = './data/loss_test_data.pkl'
BLEU_SCORE_PATH = './data/bleu_score_data.pkl'
WER_SCORE_PATH = './data/wer_score_data.pkl'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--feature_num', '-f', default=FEATURE_NUM, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=HIDDEN_NUM, type=int, help='dimension of hidden layer')
parser.add_argument('--label_num', '-ln', default=LABEL_NUM, type=int, help='dimension of label layer')
parser.add_argument('--label_embed', '-le', default=LABEL_EMBED, type=int, help='dimension of label embed layer')
parser.add_argument('--bar', '-b', default='0', type=int, help='whether to show the graph of loss values or not')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
parser.add_argument('--beam_search', '-be', default=True, type=bool, help='show results using beam search')
args = parser.parse_args()

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()


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


def fixed_interpreter(data_path, model_path):
    """
    感情タグを入力しない方，トピックラベルの方は0,1の二値を入力
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    if args.lang == 'en':
        corpus = ConvCorpus(file_path=None)
    elif args.lang == 'ja':
        corpus = ProposalConvCorpus(file_path=None)
    else:
        print('You gave wrong argument to this system. Check out your argument about languages.')
        raise ValueError
    corpus.load(load_dir=data_path)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = MultiTaskSeq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                             feature_num=args.feature_num, hidden_num=args.hidden_num,
                             label_num=args.label_num, label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)
    emo_label_index = [index for index in range(LABEL_NUM)]
    topic_label_index = [index for index in range(TOPIC_NUM)]

    # run conversation system
    print('The system is ready to run, please talk to me!')
    print('( If you want to end a talk, please type "exit". )')
    print('')
    while True:
        print('>> ', end='')
        sentence = input()
        if sentence == 'exit':
            print('See you again!')
            break

        # check a topic tag
        input_vocab = sentence.split(' ')
        topic_label_id = input_vocab.pop(-1)
        label_false_flg = 1
        for index in topic_label_index:
            if topic_label_id == str(index):
                topic_label_id = index             # TODO: ラベルのインデックスに注意する．今は3値分類 (0, 1, 2)
                label_false_flg = 0
                break
        if label_false_flg:
            print('caution: you donot set any enable tags!')
            input_vocab = sentence.split(' ')
            topic_label_id = -1

        if args.lang == 'en':
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(sentence)]
        elif args.lang == 'ja':
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in parse_ja_text(sentence)]

        input_vocab.pop(-1)
        input_vocab_rev = input_vocab[::-1]

        # convert word into ID
        input_sentence = [corpus.dic.token2id[word] for word in input_vocab if not corpus.dic.token2id.get(word) is None]
        input_sentence_rev = [corpus.dic.token2id[word] for word in input_vocab_rev if not corpus.dic.token2id.get(word) is None]

        model.initialize(batch_size=1)
        for emo_label in range(LABEL_NUM):
            sentence = model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                      emo_label_id=emo_label, topic_label_id=topic_label_id,
                                      word2id=corpus.dic.token2id, id2word=corpus.dic)
            if emo_label == 0:
                print("neg -> ", sentence)
            elif emo_label == 1:
                print("neu -> ", sentence)
            elif emo_label == 2:
                print("pos -> ", sentence)
            else:
                raise ValueError
        print('')


def interpreter(data_path, model_path):
    """
    Run this function, if you want to talk to seq2seq model.
    if you type "exit", finish to talk.
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    if args.lang == 'en':
        corpus = ConvCorpus(file_path=None)
    elif args.lang == 'ja':
        corpus = ProposalConvCorpus(file_path=None)
    else:
        print('You gave wrong argument to this system. Check out your argument about languages.')
        raise ValueError
    corpus.load(load_dir=data_path)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = MultiTaskSeq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                             feature_num=args.feature_num, hidden_num=args.hidden_num,
                             label_num=args.label_num, label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)
    emo_label_index = [index for index in range(LABEL_NUM)]
    topic_label_index = [index for index in range(TOPIC_NUM)]

    # run conversation system
    print('The system is ready to run, please talk to me!')
    print('( If you want to end a talk, please type "exit". )')
    print('')
    while True:
        print('>> ', end='')
        sentence = input()
        if sentence == 'exit':
            print('See you again!')
            break

        # check a sentiment tag
        input_vocab = sentence.split(' ')
        emo_label_id = input_vocab.pop(-1)
        topic_label_id = input_vocab.pop(-1)
        label_false_flg = 1

        for index in emo_label_index:
            if emo_label_id == str(index):
                emo_label_id = index               # TODO: ラベルのインデックスに注意する．今は3値分類 (0, 1, 2)
                label_false_flg = 0
                break
        if label_false_flg:
            print('caution: you donot set any enable tags!')
            input_vocab = sentence.split(' ')
            emo_label_id = -1

        # check a topic tag TODO: 本当はユーザ側の指定ではなく，tweet2vecの判定から決定する
        label_false_flg = 1
        for index in topic_label_index:
            if topic_label_id == str(index):
                topic_label_id = index             # TODO: ラベルのインデックスに注意する．今は3値分類 (0, 1, 2)
                label_false_flg = 0
                break
        if label_false_flg:
            print('caution: you donot set any enable tags!')
            input_vocab = sentence.split(' ')
            topic_label_id = -1

        if args.lang == 'en':
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(sentence)]
        elif args.lang == 'ja':
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in parse_ja_text(sentence)]

        input_vocab_rev = input_vocab[::-1]

        # convert word into ID
        input_sentence = [corpus.dic.token2id[word] for word in input_vocab if not corpus.dic.token2id.get(word) is None]
        input_sentence_rev = [corpus.dic.token2id[word] for word in input_vocab_rev if not corpus.dic.token2id.get(word) is None]

        model.initialize(batch_size=1)
        if args.beam_search:
            hypotheses = model.beam_search(model.initial_state_function, model.generate_function,
                                           input_sentence, input_sentence_rev,
                                           start_id=corpus.dic.token2id['<start>'],
                                           end_id=corpus.dic.token2id['<eos>'], emo_label_id=emo_label_id,
                                           topic_label_id=topic_label_id)
            for hypothesis in hypotheses:
                generated_indices = hypothesis.to_sequence_of_values()
                generated_tokens = [corpus.dic[i] for i in generated_indices]
                print("--> ", " ".join(generated_tokens))
        else:
            sentence = model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                      emo_label_id=emo_label_id, topic_label_id=topic_label_id,
                                      word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("-> ", sentence)
        print('')


def test_run(data_path, model_path, n_show=80):
    """
    Test function.
    Input is training data.
    Output have to be the sentence which is correct data in training phase.
    :return:
    """

    # call dictionary class
    if args.lang == 'en':
        corpus = ConvCorpus(file_path=None)
    elif args.lang == 'ja':
        corpus = ProposalConvCorpus(file_path=None)
    else:
        print('You gave wrong argument to this system. Check out your argument about languages.')
        raise ValueError
    corpus.load(load_dir=data_path)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = MultiTaskSeq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                             feature_num=args.feature_num, hidden_num=args.hidden_num,
                             label_num=args.label_num, label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run an interpreter
    for num, input_sentence in enumerate(corpus.fine_posts):
        id_sequence = input_sentence.copy()
        input_sentence_rev = input_sentence[::-1]

        # make label lists TODO: 3値分類
        n_num = p_num = 0
        topic_label_id = correct_emo_label = -1
        for index, w_id in enumerate(corpus.fine_cmnts[num]):
            # comment の最初にトピックラベルが付いているものとする
            if index == 0:
                if w_id == corpus.dic.token2id['__label__0']:
                    topic_label_id = 0
                elif w_id == corpus.dic.token2id['__label__1']:
                    topic_label_id = 1
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
            correct_emo_label = 1
        elif n_num <= p_num:
            correct_emo_label = 2
        elif n_num > p_num:
            correct_emo_label = 0
        else:
            raise ValueError

        # generate an output
        print("input : ", " ".join([corpus.dic[w_id] for w_id in id_sequence]))
        print("train emotion label: ", correct_emo_label)
        print("correct :", " ".join([corpus.dic[w_id] for index, w_id in enumerate(corpus.fine_cmnts[num]) if index != 0]))
        print(input_sentence)
        print(input_sentence_rev)
        for emo_label in range(LABEL_NUM):
            model.initialize(batch_size=1)
            sentence = model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                      emo_label_id=emo_label, topic_label_id=topic_label_id,
                                      word2id=corpus.dic.token2id, id2word=corpus.dic)
            if emo_label == 0:
                print("neg -> ", sentence)
            elif emo_label == 1:
                print("neu -> ", sentence)
            else:
                print("pos -> ", sentence)
        print('')

        if num == n_show:
            break


def show_chart(train_loss_path, test_loss_path):
    """
    Show the graph of Losses for each epochs
    """
    with open(train_loss_path, mode='rb') as f:
        train_loss_data = np.array(pickle.load(f))
    #with open(test_loss_path, mode='rb') as f:
    #    test_loss_data = np.array(pickle.load(f))
    row = len(train_loss_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, train_loss_data, label="Train Loss Value", color="gray")
    #plt.plot(loop_num, test_loss_data, label="Test Loss Value", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc=2)
    plt.title("Learning Rate of Seq2Seq Model")
    plt.show()


def show_bleu_chart(bleu_score_path):
    """
    Show the graph of BLEU for each epochs
    """
    with open(bleu_score_path, mode='rb') as f:
        bleu_score_data = np.array(pickle.load(f))
    row = len(bleu_score_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, bleu_score_data, label="BLUE score", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.legend(loc=2)
    plt.title("BLEU score of Seq2Seq Model")
    plt.show()


def show_wer_chart(wer_score_path):
    """
    Show the graph of WER for each epochs
    """
    with open(wer_score_path, mode='rb') as f:
        wer_score_data = np.array(pickle.load(f))
    row = len(wer_score_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, wer_score_data, label="WER score", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.legend(loc=2)
    plt.title("WER score of Seq2Seq Model")
    plt.show()


if __name__ == '__main__':
    fixed_interpreter(DATA_DIR, MODEL_PATH)
    # interpreter(DATA_DIR, MODEL_PATH)
    # test_run(DATA_DIR, MODEL_PATH)
    # if args.bar:
    #     show_chart(TRAIN_LOSS_PATH, TEST_LOSS_PATH)
    #     show_bleu_chart(BLEU_SCORE_PATH)
    #     show_wer_chart(WER_SCORE_PATH)
