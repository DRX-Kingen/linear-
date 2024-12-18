import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

df = pd.read_csv("StudentsPerformance.csv")

# Lấy dữ liệu
reading_score = df['reading score'].values
writing_score = df['writing score'].values
math_score = df['math score'].values
data = {
    'reading score': reading_score,
    'writing score': writing_score,
    'math score': math_score
}
data = pd.DataFrame(data)

# Chia tập train và test
train, test = train_test_split(data, test_size=0.2, random_state=42)

X = np.array([train['reading score'].values, train['writing score'].values]).T
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, np.array([train['math score'].values]).T)
w = np.dot(np.linalg.pinv(A), b)

w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]

math_score_real = test['math score'].values

def predict(test):
    score_pre = []
    for reading_score, writing_score in zip(test['reading score'], test['writing score']):
        pre = int(w_1 * reading_score + w_2 * writing_score + w_0)
        score_pre.append(pre)
    return score_pre

score_pre = predict(test)

def calculate_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_predictions += 1
    return correct_predictions / len(y_true)

if __name__ == '__main__':
    st.title("Phân tích và Dự đoán Điểm Toán")

    # Hiển thị biểu đồ phân phối điểm
    st.header("Biểu đồ phân phối điểm")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(data['reading score'], kde=True, ax=ax[0], color='blue')
    ax[0].set_title("Phân phối điểm đọc")
    sns.histplot(data['writing score'], kde=True, ax=ax[1], color='green')
    ax[1].set_title("Phân phối điểm viết")
    sns.histplot(data['math score'], kde=True, ax=ax[2], color='red')
    ax[2].set_title("Phân phối điểm toán")

    st.pyplot(fig)

    # Hiển thị biểu đồ quan hệ giữa các loại điểm
    st.header("Biểu đồ quan hệ giữa các loại điểm")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='reading score', y='math score', data=data, label='Reading vs Math', color='blue')
    sns.scatterplot(x='writing score', y='math score', data=data, label='Writing vs Math', color='green')
    ax.set_title("Quan hệ giữa điểm đọc, viết và toán")
    ax.legend()

    st.pyplot(fig)

    # Nhập điểm đọc và viết
    reading_score_ip = st.number_input("Nhập vào điểm đọc:", min_value=0, max_value=100)
    writing_score_ip = st.number_input('Nhập vào điểm viết:', min_value=0, max_value=100)

    # Dự đoán điểm toán
    math_score_ip = int(w_1 * reading_score_ip + w_2 * writing_score_ip + w_0)
    y_pred = score_pre
    y_true = test['math score'].values

    # Tính sai số
    mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)

    st.write("Điểm toán dự đoán:", math_score_ip)
    st.write("Sai số trung bình:", f"{rmse:.2f}")

    # Hộp kiểm để hiển thị thông tin chi tiết
    if st.checkbox("Hiển thị thông tin chi tiết trên tập kiểm tra"):
        st.subheader("Thông tin chi tiết")
        detailed_results = test.copy()
        detailed_results['Predicted Math Score'] = y_pred
        st.dataframe(detailed_results)

        # Hiển thị sai số từng điểm
        detailed_results['Error'] = detailed_results['math score'] - detailed_results['Predicted Math Score']
        st.write("Sai số từng điểm:")
        st.dataframe(detailed_results[['reading score', 'writing score', 'math score', 'Predicted Math Score', 'Error']])