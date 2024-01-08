import pandas as pd
data={'city':['北京','上海','广州','天津'],'code':['010','021','020','022']}
frame=pd.DataFrame(data)
print(frame)
frame.to_csv("citycode.csv")

