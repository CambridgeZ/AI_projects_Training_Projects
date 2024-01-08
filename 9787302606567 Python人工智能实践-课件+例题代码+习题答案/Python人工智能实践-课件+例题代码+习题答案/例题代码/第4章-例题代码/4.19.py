#爬取豆瓣电影top250的电影信息

import requests
import bs4
import csv

# 1. 创建文件对象
f = open('douban_Top250.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
# 2. 构建列表头
csv_writer.writerow(["豆瓣电影Top250"])
csv_writer.writerow(["序号", "电影名称", "评分", "推荐语","网址"])

for x in range(10):
    #3. 标记了请求从什么设备，什么浏览器上发出    
    headers = {
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'

    } 
    url = 'https://movie.douban.com/top250?start=' + str(x*25) + '&filter='
    res = requests.get(url,headers=headers)
    bs = bs4.BeautifulSoup(res.text, 'html.parser')
    bs = bs.find('ol', class_="grid_view")
    for titles in bs.find_all('li'):
        num = titles.find('em',class_="").text
        title = titles.find('span', class_="title").text
        comment = titles.find('span',class_="rating_num").text
        url_movie = titles.find('a')['href']
        #4.AttributeError: 'NoneType' object has no attribute 'text'，为避免此类报错，增加一个条件判断
        if titles.find('span',class_="inq") != None:
            tes = titles.find('span',class_="inq").text
            csv_writer.writerow([num, title, comment, tes,url_movie])
        else:
            csv_writer.writerow([num, title,comment,'none', url_movie])
f.close()
