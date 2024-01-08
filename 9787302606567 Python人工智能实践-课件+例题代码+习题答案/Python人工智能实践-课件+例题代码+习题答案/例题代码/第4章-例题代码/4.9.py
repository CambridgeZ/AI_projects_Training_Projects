import json
dic={'ID':'1001','age':22}
print(dic)
print(type(dic))
s=json.dumps(dic)
print('通过json.dumps处理后:')
print(s)
print(type(s))
