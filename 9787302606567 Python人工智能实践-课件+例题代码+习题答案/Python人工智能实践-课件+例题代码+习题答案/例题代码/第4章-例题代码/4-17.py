import requests
url="https://zhidao.baidu.com/"
try:
    res=requests.get(url)
    res.raise_for_status()
    res.encoding=res.apparent_encoding
    print(res.text[:300])
except:
    print("爬取失败")
