#weight convert between kg and lb
'''
input:   a number ended with kg or lb
process: 1kg=2.2046226lb
output:  a float number with 2 decimal places
'''
InStr=input( )
if InStr[-2:] in ['kg', 'KG']:
    ConStr=eval(InStr[0:-2])*2.2046226
    print("{:.2f}lb".format(ConStr))
elif InStr[-2:] in ['lb', 'LB']:
    ConStr=eval(InStr[0:-2])/2.2046226
    print("{:.2f}kg".format(ConStr))
else:
    print("format error")

