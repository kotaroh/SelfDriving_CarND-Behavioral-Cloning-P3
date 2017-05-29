import pandas as pd
import numpy as np
from ggplot import *

current_paths = ['./data/','./Data2/Test14/','./Data2/Test15/','./Data2/Test16/','./Data2/Test17/','./Data2/Test18/','./Data2/Test19/','./Data2/\
Test20/','./Data2/Test21/','./Data2/Test22/','./Data2/Test23/','./Data2/Test24/','./Data2/Test25/','./Data2/Test26/','./Data2/Test27/','./Data2/\
Test28/','./Data2/Test29/','./Data2/Test30/','./Data2/Test31/','./Data2/Test32/']

#load data


df2 = pd.DataFrame()
ranges = []

for i in range(0,20):
    ranges.append(-1 + i * 0.1)

print(ranges)

df_list = []
for current_path in current_paths:
    csv_path = current_path + 'driving_log.csv'
    print(csv_path)
    
    df = pd.read_csv(csv_path,usecols=[3],names = ["Angle"])

    df_list.append(df)

result = pd.concat(df_list)
data = result.groupby(pd.cut(result['Angle'], ranges)).count()
print(data)

#p = ggplot(aes(x="Angle", weight="Angle"), data) + geom_bar()
#ggsave(p,filename = 'test.png')
#p.save('test.png')

                                
