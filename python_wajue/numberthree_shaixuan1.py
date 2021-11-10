import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.DataFrame({"A":[np.nan,2,9], "B":[4,14,6], "c":[987,8,9]})
f, ax= plt.subplots(figsize = (10, 10))

corr = data.corr()
# print(corr)
sns.heatmap(corr,cmap='rainbow', linewidths = 0.05, ax = ax,annot=True,linecolor='white')

# 设置Axes的标题
ax.set_title('Correlation between features')
plt.show()
plt.close()
f.savefig('sns_style_origin.jpg', dpi=100, bbox_inches='tight')
