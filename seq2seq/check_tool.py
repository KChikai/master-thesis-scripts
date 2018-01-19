# -*- coding:utf-8 -*-

import sys
import pandas as pd

# load the names of files
argvs = sys.argv
if len(argvs) < 2:
    print('Usage: # python %s filename' % argvs[0])
    quit()
books = argvs[1:]

# check annotation files
sheet_names = ['annotation1', 'annotation2']
tags = [0, 1, 'a', 'b', 'pos', 'neg', 'neu', 'none']
for book in books:
    for sheet_name in sheet_names:
        try:
            data_frame = pd.read_excel(book, sheet_name=sheet_name)
            if sheet_name == sheet_names[0]:
                for item_index, line in data_frame.iterrows():
                    annotations = [line['fluency_B'], line['fluency_A'],
                                   line['consistency_B'], line['consistency_A'],
                                   str(line['domain_consistency']), str(line['emotion'])]
                    for annotation in annotations:
                        if annotation not in tags:
                            print("アノテーションタスク1にミスがあるようです．")
                            print("インデックス", item_index)
                            print("アノテーションミス：", annotation, end='\n\n')
                            raise ValueError
                print(argvs[1], "アノテーションタスク1成功しています．お疲れ様でした．")
            elif sheet_name == sheet_names[1]:
                for item_index, line in data_frame.iterrows():
                    annotations = [line['emotion_tag']]
                    for annotation in annotations:
                        if annotation not in tags:
                            print("アノテーションタスク2にミスがあるようです．")
                            print("インデックス", item_index)
                            print("アノテーションミス：", annotation, end='\n\n')
                            raise ValueError
                print(argvs[1], "アノテーションタスク2成功しています．お疲れ様でした．")
            else:
                pass
        except:
            # value error and sheet name error
            pass