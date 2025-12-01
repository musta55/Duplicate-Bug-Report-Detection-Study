import os
import sys
import jieba

sys.path.append("../../")


# perform word segment and return word token list
def word_segment2token(sample):
    result = jieba.cut(sample, cut_all=False)
    outstr_list = list()
    curpath = os.path.dirname(os.path.realpath(__file__))
    stopwords = [line.strip() for line in open(os.path.join(curpath, 'stopword.txt'), 'r', encoding='utf-8').readlines()]
    for word in result:
        if word not in stopwords:
            outstr_list.append(word)
    str_out = ' '.join(outstr_list)
    return str_out
