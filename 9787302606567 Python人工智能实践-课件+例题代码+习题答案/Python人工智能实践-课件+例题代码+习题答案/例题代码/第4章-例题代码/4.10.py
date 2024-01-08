import json
json_str='{"ID":"100","age":22}'
print(json_str)
print(type(json_str))
d=json.loads(json_str)
print('通过json.loads处理后:')
print(d)
print(type(d))
