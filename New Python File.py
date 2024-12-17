import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
df = pd.read_csv("StudentsPerformance.csv")
#print(df.head())
reading_score=df['reading score'].values
writing_score=df['writing score'].values
math_score=df['math score'].values
data={
    'reading score':reading_score,
    'writing score':writing_score,
    'math score':math_score
}
data=pd.DataFrame(data)
#print(data)
train, test = train_test_split(data, test_size=0.2, random_state=42)



#print('Train size:', len(train))
#print('Train test:', len(test))
#print(train)

X=np.array([train['reading score'].values, train['writing score'].values]).T
one=np.ones((X.shape[0],1))
Xbar=np.concatenate((one,X),axis=1)
#print(Xbar)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,np.array([train['math score'].values]).T)
#print(b)
w=np.dot(np.linalg.pinv(A),b)

#print(w)
w_0=w[0][0]
w_1=w[1][0]
w_2=w[2][0]
math_score_real=test['math score'].values

def predict(test):
  score_pre=[]
  for reading_score,writing_score in zip(test['reading score'],test['writing score']):
    pre=int(w_1* reading_score + w_2 * writing_score + w_0)
    score_pre.append(pre)
  return score_pre
score_pre=predict(test)

def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_predictions += 1
    return correct_predictions / len(y_true)

if __name__=='__main__':
    
    reading_score_ip=st.number_input("Nhập vào điểm đọc:",min_value=0,max_value=100)
    writing_score_ip=st.number_input('Nhập vào điểm viết:',min_value=0,max_value=100)
    math_score_ip=int(w_1* reading_score_ip + w_2 * writing_score_ip + w_0)
    y_pred=score_pre
    y_true=test['math score'].values
    mse = np.mean((y_true - y_pred)**2)  # Mean Squared Error
    rmse = np.sqrt(mse)
    st.write("Điểm toán dự đoán:",math_score_ip)
    st.write("Sai số trung bình:",f"{rmse:.2f}")    