import os
import sys
import jieba

sys.path.append("../../")

# Cache stopwords
_stopwords = None

# perform word segment and return word token list
def word_segment2token(sample):
    global _stopwords
    if _stopwords is None:
        curpath = os.path.dirname(os.path.realpath(__file__))
        _stopwords = set(line.strip() for line in open(os.path.join(curpath, 'stopword.txt'), 'r', encoding='utf-8').readlines())
        
    result = jieba.cut(sample, cut_all=False)
    outstr_list = list()
    for word in result:
        if word not in _stopwords:
            outstr_list.append(word)
    str_out = ' '.join(outstr_list)
    return str_out
