import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv("./data.csv")
df.columns=["index","height","weight"]
print(df.head())

sns.scatterplot(
    data=df,
    x="height",
    y="weight"
)

#plt.show()

x=df["height"].values
y=df["weight"].values

N=x.shape[0]
m=(N*np.sum(x*y)-np.sum(x)*np.sum(y))/(N*np.sum(x**2)-(np.sum(x)**2))
b=(np.sum(y)-m*np.sum(x))/N
print(m,b)

x_min=np.min(x)
y_min=m*x_min+b
x_max=np.max(x)
y_max=m*x_max+b

fig, ax=plt.subplots()
sns.scatterplot(
    data=df,
    x="height",
    y="weight",
    ax=ax,
    alpha=0.4
)

sns.lineplot(
    x=[x_min,x_max],
    y=[y_min,y_max],
    linewidth=1.5,
    color="red"
)
plt.show()