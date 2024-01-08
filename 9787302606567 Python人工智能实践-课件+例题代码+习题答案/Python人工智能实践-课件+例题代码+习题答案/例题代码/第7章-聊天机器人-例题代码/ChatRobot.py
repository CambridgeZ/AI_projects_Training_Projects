# 导入中文分词库
import jieba

#对语料库进行分词，分词结果存入fenci_result.txt文件
f1 = open('corpus.txt', encoding="utf8")
f2 = open('fenci_result.txt', 'a', encoding="utf8")
lines = f1.readlines()
for line in lines:
	line.replace('\t', '').replace('\n', '').replace(' ', '')
	seg_list = jieba.cut(line)
	# 将词汇以空格隔开
	f2.write(' '.join(seg_list))

f1.close()
f2.close()

# 获取计算文本向量及其相似度模型
from gensim.models import word2vec

sentences = word2vec.Text8Corpus('fenci_result.txt')
model = word2vec.Word2Vec(sentences)
model.save("word2Vec.model")

#导入相关库
import json
import random
import math

# 加载和解析模板文件
f3 = open('templet.txt', encoding="utf8")
str = ''
for line in f3.readlines():
	str += line
content = json.loads(str)

f3.close()
#print(content)


# 寻找最大相似度的回答
def answer(input):
	# 存储最大相似度
	similarityMax = 0
	# 存储最大相似度问句的下标
	similarityIndex = -1
	# 对用户输入做分词处理
	input_word_arr = list(jieba.cut(input))
	# 遍历规则库
	for i in range(len(content)):
		title_word_arr = list(jieba.cut(content[i]['title'].replace(' ', '')))
		#print(title_word_arr)
		# 使用try...except语法来做余弦相似度计算，避免因词向量小而引发报错
        #similarity越大越相似
		try:
			similarity = model.wv.n_similarity(input_word_arr, title_word_arr)
		except Exception:
			similarity = 0
		# 储存当前最大相似度及其下标
		if similarityMax < similarity:
			similarityMax = similarity
			similarityIndex = i
	# 随机取一个回复，如果similarityIndex为-1，则说明未匹配到相似语句
	if similarityIndex != -1:	
		reply_index = math.floor(random.random() * len(content[similarityIndex]['reply']))
		if reply_index:
			return {"title": content[similarityIndex]['title'], "reply": content[similarityIndex]['reply'][reply_index]}
	return {"title": "无", "reply": "抱歉，我不太明白您的意思"}

# 对话流程
def main():
	while True:
		#接受用户输入
		input_str = input("用户：")
		#寻找匹配的答复
		result = answer(input_str)
		#输出结果
		print("匹配到问题: %s 回答： %s" % (result['title'], result['reply']))

main()
