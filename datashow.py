import pandas as pd
import numpy as np

current_paths = ['./data/','./Data2/Test14/','./Data2/Test15/','./Data2/Test16/','./Data2/Test17/','./Data2/Test18/','./Data2/Test19/','./Data2/\
Test20/','./Data2/Test21/','./Data2/Test22/','./Data2/Test23/','./Data2/Test24/','./Data2/Test25/','./Data2/Test26/','./Data2/Test27/','./Data2/\
Test28/','./Data2/Test29/','./Data2/Test30/','./Data2/Test31/','./Data2/Test32/']

#set range of the angle for grouping
ranges = []

for i in range(0,20):
    ranges.append(-1 + i * 0.1)

#load data
df_list = []
for current_path in current_paths:
    csv_path = current_path + 'driving_log.csv'
    print(csv_path)
    
    df = pd.read_csv(csv_path,usecols=[3],names = ["angle"])

    df_list.append(df)

result = pd.concat(df_list)
range_list = pd.cut(result['angle'], ranges)
data = result.groupby(range_list).count()
data.index.name = 'Angle range'

print(data)

#Visualization
p = data.plot(legend = False,kind = 'bar')
p.set_ylabel("Count")
fig = p.get_figure()
fig.tight_layout()
fig.savefig('test.png')


                                
