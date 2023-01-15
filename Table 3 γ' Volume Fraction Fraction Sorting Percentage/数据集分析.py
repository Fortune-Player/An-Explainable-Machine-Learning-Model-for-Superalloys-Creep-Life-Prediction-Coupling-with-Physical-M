import pandas as pd

pd.set_option('display.max_columns',None)
df = pd.DataFrame(pd.read_excel("数据汇总.xlsx"))
a = df.describe(percentiles=[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])
# print(a)
df_out = pd.DataFrame(a)
df_out.to_excel("数据分析.xlsx")