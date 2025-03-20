import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import chardet

# Load trained model
with open("fashion_trend_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load encoders
with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

def read_file(uploaded_file):
    # Kiểm tra định dạng file
    if uploaded_file.name.endswith('.csv'):
        # Tự động phát hiện encoding
        raw_data = uploaded_file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        
        # Đọc file với encoding phát hiện được
        uploaded_file.seek(0)  # Reset file pointer
        df = pd.read_csv(uploaded_file, encoding=detected_encoding, low_memory=False)
    
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    return df

def explore_dataframe(df):
    # Shape
    shape_info = pd.DataFrame({"Shape of data": [f"Total: {df.shape[0]} rows, {df.shape[1]} columns"]})
    shape_info = shape_info.replace(np.nan, "-")

    # Data Types
    data_types_info = df.dtypes.to_frame().reset_index().rename(columns={"index": "Column", 0: "Data Type"})

    # Missing Values
    missing_values_info = df.isnull().sum().to_frame().reset_index().rename(columns={"index": "Column", 0: "Missing Values"})
    missing_values_info["Missing Values"] = missing_values_info["Missing Values"].fillna("-")

    # Duplicate Rows
    duplicate_rows_info = pd.DataFrame({"Column": ["Duplicate Rows"], "Value": [df.duplicated().sum()]})
    duplicate_rows_info = duplicate_rows_info.replace(np.nan, "-")

    # Unique Values
    unique_values_info = df.nunique().to_frame().reset_index().rename(columns={"index": "Column", 0: "Unique Values"})

    # Merge all tables into one
    info_table = pd.merge(data_types_info, missing_values_info, on="Column", how="outer")
    info_table = pd.merge(info_table, unique_values_info, on="Column", how="outer")

    # Append shape and duplicate row info at the bottom
    extra_info = pd.concat([shape_info, duplicate_rows_info], ignore_index=True)

    # Display tables
    st.subheader("🔍 Data Preview")
    st.write(df)
    st.subheader("📈 Statistical Summary")
    st.write(df.describe(include = 'all'))
    st.subheader("🧮 Overview Info")
    st.write(info_table)
    st.write(extra_info)

# Function to preprocess input data
def preprocess_input(data):
    categorical_features = ['Brand', 'Category', 'Style Attributes', 'Color', 'Season']
    for col in categorical_features:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])
    return data

def inverse_transform(df):
    """Chuyển các giá trị số về dạng text dựa trên encoders"""
    df_decoded = df.copy()
    for col, encoder in encoders.items():
        if col in df_decoded.columns:
            df_decoded[col] = encoder.inverse_transform(df_decoded[col])
    return df_decoded

# Function for data visualization
def visualize_data(df):
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize", df.columns)
    st.write(f'📈 Distribution of {selected_feature}')
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Function to make predictions
def predict_fashion_trend(input_data):
    processed_data = preprocess_input(pd.DataFrame([input_data]))
    prediction = model.predict(processed_data)
    proba = model.predict_proba(processed_data)
    prob = proba[:,1][0]*100
    return f"🔥 Trend ({prob:.2f}%)" if prediction[0] == 1 else f"📉 Not Trend ({prob:.2f}%)"

# Function to handle CSV uploads and batch predictions
def batch_predict(uploaded_file):
    df = read_file(uploaded_file)
    processed_df = preprocess_input(df)
    predictions = model.predict(processed_df)
    probability = model.predict_proba(processed_df)[:,1]
    df['Prediction'] = ["Trend" if p == 1 else "Not Trend" for p in predictions]
    df['Probability'] = probability*100
    df["Probability"] = df["Probability"].apply(lambda x: f"{x:.2f}%")
    # Chuyển từ dạng số về dạng chữ
    df_final = inverse_transform(df)
    return df_final


# Streamlit UI
st.title("🕺 Fashion Trend Prediction 💃")
if "page" not in st.session_state:
    st.session_state.page = "analysis"
if "analysis_file" not in st.session_state:
    st.session_state.analysis_file = None
if "batch_file" not in st.session_state:
    st.session_state.batch_file = None

# Tạo nút điều hướng
st.sidebar.title("📜 Menu")
if st.sidebar.button("📊 Data Analysis and Visualization"):
    st.session_state.page = "analysis"
if st.sidebar.button("🛒 Predict Trendy Product"):
    st.session_state.page = "prediction"
if st.sidebar.button("🤖 Batch Prediction"):
    st.session_state.page = "batch"

if st.session_state.page == "prediction":
    st.header("🛒 Predict Trendy Product")
    price = st.number_input("Price ($)", min_value=0.0, step=1.0)
    brand = st.selectbox("Brand", encoders['Brand'].classes_)
    category = st.selectbox("Category", encoders['Category'].classes_)
    style_attributes = st.selectbox("Style Attributes", encoders['Style Attributes'].classes_)
    color = st.selectbox("Color", encoders['Color'].classes_)
    season = st.selectbox("Season", encoders['Season'].classes_)
    
    input_data = {
        "Price": price,
        "Brand": brand,
        "Category": category,
        "Style Attributes": style_attributes,
        "Color": color,
        "Season": season
    }
    
    if st.button("🔍 Predict"):
        result = predict_fashion_trend(input_data)
        st.write(f"Prediction: {result}")

elif st.session_state.page == "analysis":
    st.header("📊 Data Analysis and Visualization")
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        st.session_state.analysis_file = uploaded_file
    if st.session_state.analysis_file is not None:
        df = read_file(st.session_state.analysis_file)
        explore_dataframe(df)
        visualize_data(df)
    else:
        st.warning("Please upload a file first!")

elif st.session_state.page == "batch":
    st.header("🤖 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for predicting", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        st.session_state.batch_file = uploaded_file
    if st.session_state.batch_file is not None:
        result_df = batch_predict(st.session_state.batch_file)
        st.subheader("🔎 Prediction Preview")
        st.write(result_df)
        st.download_button("📥 Download Predictions", result_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("Please upload a file first!")
        
st.markdown('<div style="visibility: hidden;">Fix preview bug</div>', unsafe_allow_html=True)
