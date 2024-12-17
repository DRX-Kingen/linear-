import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Extract features and target
reading_score = df['reading score'].values
writing_score = df['writing score'].values
math_score = df['math score'].values

# Create DataFrame
data = {
    'reading score': reading_score,
    'writing score': writing_score,
    'math score': math_score
}
data = pd.DataFrame(data)

# Split dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Features and target
X_train = train[['reading score', 'writing score']]
y_train = train['math score']
X_test = test[['reading score', 'writing score']]
y_test = test['math score']

# Feature Engineering: Add interaction and polynomial terms
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit app
if __name__ == '__main__':
    st.title("Dự đoán điểm Toán")

    # User input
    reading_score_ip = st.number_input("Nhập vào điểm đọc:", min_value=0, max_value=100, value=50)
    writing_score_ip = st.number_input("Nhập vào điểm viết:", min_value=0, max_value=100, value=50)

    # Prediction for user input
    input_data = np.array([[reading_score_ip, writing_score_ip]])
    input_data_poly = poly.transform(input_data)
    input_data_scaled = scaler.transform(input_data_poly)
    math_score_ip = model.predict(input_data_scaled)[0]

    st.write("\nĐiểm Toán dự đoán:", round(math_score_ip, 2))

    # Option to display detailed results
    if st.checkbox("Hiển thị thông tin chi tiết"):
        st.subheader("Kết quả chi tiết trên tập kiểm tra")
        detailed_results = pd.DataFrame({
            "Thực tế": y_test,
            "Dự đoán": y_pred,
            "Sai số": np.abs(y_test - y_pred)
        })
        st.write(detailed_results)

    # Display metrics
    st.write("\nSai số trung bình (RMSE):", round(rmse, 2))