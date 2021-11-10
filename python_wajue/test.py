# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/10/11 16:45 
# License: bupt
import pandas as pd

writer = pd.ExcelWriter('test.xlsx')

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [11, 12, 13, 14]})

df.to_excel(writer, sheet_name='test1')

df.to_excel(writer, sheet_name='test2')

writer.save()

writer.close()

print([*zip(range(1, 10))][5])
