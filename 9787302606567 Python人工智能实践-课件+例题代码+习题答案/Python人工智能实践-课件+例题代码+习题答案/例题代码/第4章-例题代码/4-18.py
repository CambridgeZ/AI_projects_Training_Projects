import requests
url="http://product.dangdang.com/29311943.html"
try:
    header={'user-agent':'Mozilla/5.0'}
    res=requests.get(url,headers=header)
    res.raise_for_status()
    res.encoding=res.apparent_encoding
    print(res.text[:500])
except:
    print("爬取失败")
