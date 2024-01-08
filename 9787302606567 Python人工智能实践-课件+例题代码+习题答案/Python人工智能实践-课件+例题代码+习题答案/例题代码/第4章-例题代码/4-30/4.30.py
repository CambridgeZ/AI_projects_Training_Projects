import jieba
import wordcloud
stopwords = ['需要','采用','以及','可以','可能' ]
f = open("data.txt", "r", encoding="utf-8")
t = f.read()
f.close()
ls = jieba.lcut(t)
txt = " ".join(ls)
wc = wordcloud.WordCloud(\
    width = 1000, height = 700,\
    background_color = "white",
    font_path = "msyh.ttc", stopwords=stopwords
    )
wc.generate(txt)
wc.to_file("mywc.png")
