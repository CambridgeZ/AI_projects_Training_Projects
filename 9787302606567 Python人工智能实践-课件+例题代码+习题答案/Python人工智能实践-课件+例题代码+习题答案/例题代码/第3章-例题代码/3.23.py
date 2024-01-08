import jieba
excludes = {'的','和','中','在','了','已经','多','从'}
txt="人工智能在过去十年中从实验室走向产业化生产，重塑传统行业模式、\
引领未来的价值已经凸显，并为全球经济和社会活动做出了不容忽视的贡献。\
当前，人工智能已经迎来其发展史上的第三次浪潮。人工智能理论和技术取得了\
飞速发展，在语音识别、文本识别、视频识别等感知领域取得了突破，达到或超过\
人类水准，成为引领新一轮科技革命和产业变革的战略性技术。人工智能的应用领域\
也快速向多方向发展，出现在与人们日常生活息息相关的越来越多的场景中。"
for c in '、，。！；':
     txt=txt.replace(c,'')
words  = jieba.lcut(txt)
counts = {}
for word in words:
    counts[word] = counts.get(word,0) + 1
for word in excludes:
    del counts[word]
newwords = list(counts.items())
newwords.sort(key=lambda x:x[1], reverse=True) 
for i in range(5):
    word, count = newwords[i]
    print ("{0:<6}{1:>3}".format(word, count))
