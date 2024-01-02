import random
import math
import json
from gensim.models import word2vec
import jieba


def answer(input):
    similarityMax =0
    similarityIndex = -1

    input_word_arr = list(jieba.cut(input))

    #遍历规则库

    for i in range(len(content)):
        title_word_arr = list(jieba.cut(content[i]['title'].replace(' ','')))
        try:
            similarity = model.wv.n_similarity(input_word_arr,title_word_arr)
        except Exception:
            similarity =0

        if similarityMax < similarity:
            similarityMax = similarity
            similarityIndex = i
    
    if similarityIndex != -1:
        # 随机返回一个回答
        reply_index = math.floor(random.random() * len(content[similarityIndex]['reply']))
        if reply_index>=0:
            return {'title':content[similarityIndex]['title'],"reply": content[similarityIndex]['reply'][reply_index]}
    
    return {"title":"无", "reply":"抱歉我不太明白你的意思"}


if __name__ == '__main__':
    # 读取语料库
    f1 = open('corpus.txt',encoding="utf8")
    f2 = open('fenci_result.txt','a',encoding="utf8")

    lines = f1.readlines()

    for line in lines:
        line.replace('\t','').replace('\n','').replace(' ','')
        seg_list = jieba.cut(line)
        # 将词汇按照空格分开
        f2.write(' '.join(seg_list))

    f1.close()
    f2.close()

    sentences = word2vec.Text8Corpus('fenci_result.txt')

    model = word2vec.Word2Vec(sentences)
    model.save("model2Vec.model")

    f3 = open('temple.json',encoding="utf8")

    str = ''

    for line in f3.readlines():
        str+=line
    content = json.loads(str)

    f3.close()
    while True:
        input_str = input("用户：")
        result = answer(input_str)
        print("匹配到问题:%s 回答: %s" %(result['title'],result['reply']))

