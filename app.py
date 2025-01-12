import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# App Title
st.title("Dataset Summarizer and Visualizer")

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Upload Dataset", "About Dataset", "Statistics", "Future Insights", "Dashboard"]
)

# File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the dataset based on file type
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            df = None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        df = None
else:
    df = None

# Ensure dataset is loaded before navigating menus
if df is not None:
    if menu == "About Dataset":
        st.subheader("About Dataset")

        # Display Dataset Preview
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Display dataset info (types, non-null counts)
        st.write("### Dataset Information")
        st.write(df.info())

    elif menu == "Statistics":
        st.subheader("Statistics")
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())

        # Missing Value Information
        if st.checkbox("Show Missing Values"):
            st.write("### Missing Values")
            st.dataframe(df.isnull().sum())

    elif menu == "Future Insights":
        st.subheader("Future Insights")

        st.write("### Feature Importance")

        # Handle categorical data
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

        # Drop rows with missing values
        df_cleaned = df.dropna()

        # Split features and target
        if df_cleaned.shape[1] > 1:
            X = df_cleaned.iloc[:, :-1]
            y = df_cleaned.iloc[:, -1]

            if len(X) > 0 and len(y) > 0:
                model = RandomForestClassifier()
                model.fit(X, y)
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)
                st.dataframe(importance_df)
            else:
                st.write("Insufficient data for feature importance analysis.")
        else:
            st.write("Insufficient data for feature importance analysis.")

        st.write("### Feature Correlation")
        corr = df.corr()
        st.write("Correlation Matrix:")
        st.dataframe(corr)

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.write("### Top Positive and Negative Correlations")
        corr_pairs = corr.unstack().reset_index()
        corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        corr_pairs = corr_pairs[corr_pairs["Feature 1"] != corr_pairs["Feature 2"]]
        top_corr = corr_pairs.sort_values(by="Correlation", ascending=False)
        st.write("#### Top Positive Correlations")
        st.dataframe(top_corr.head(5))
        st.write("#### Top Negative Correlations")
        st.dataframe(top_corr.tail(5))

    elif menu == "Dashboard":
        st.subheader("Dashboard")
        st.write("### Visualization Options")
        chart_type = st.selectbox("Choose a chart type", ["Histogram", "Scatter Plot", "Bar Plot", "Box Plot", "Line Chart", "Pie Chart"])

        # Chart based on user choice
        if chart_type in ["Histogram", "Box Plot", "Line Chart"]:
            numeric_columns = df.select_dtypes(include=['int', 'float']).columns
            if len(numeric_columns) > 0:
                column = st.selectbox("Select column", numeric_columns)
            else:
                st.write("No numeric columns available for this chart type.")
                column = None

        if chart_type == "Histogram" and column:
            st.write(f"### Histogram of {column}")
            fig, ax = plt.subplots()
            ax.hist(df[column], bins=20, color='skyblue', edgecolor='black')
            st.pyplot(fig)
        
        elif chart_type == "Box Plot" and column:
            st.write(f"### Box Plot of {column}")
            fig = plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[column])
            st.pyplot(fig)
        
        elif chart_type == "Line Chart" and column:
            st.write(f"### Line Chart of {column}")
            fig = plt.figure(figsize=(8, 6))
            df[column].plot(kind='line')
            st.pyplot(fig)
        
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", df.columns)
            st.write(f"### Scatter Plot: {x_col} vs {y_col}")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig)

        elif chart_type == "Bar Plot":
            column = st.selectbox("Select column for bar plot", df.columns)
            if df[column].dtype in ['object', 'category'] or df[column].nunique() < 20:
                st.write(f"### Bar Plot of {column}")
                fig = px.bar(df[column].value_counts(), title=f"Bar Plot of {column}")
                st.plotly_chart(fig)
            else:
                st.write("Bar plot is more meaningful for categorical columns with fewer unique values.")
        
        elif chart_type == "Pie Chart":
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                column = st.selectbox("Select column for pie chart", categorical_columns)
                st.write(f"### Pie Chart of {column}")
                fig = px.pie(df, names=column, title=f"Pie Chart of {column}")
                st.plotly_chart(fig)
            else:
                st.write("No categorical columns available for pie chart.")

else:
    if menu != "Upload Dataset":
        st.warning("Please upload a dataset to proceed.")

# Footer message for better UX
if menu == "Upload Dataset" and df is None:
    st.info("Upload a CSV or Excel file to begin analyzing your dataset.")

# Add Footer with your name and username
st.markdown("""
    <style>
    .footer {
        text-align: center;
        font-size: 12px;
        color: grey;
        margin-top: 20px;
    }
    </style>
    <div class="footer">
        <p>Created by Rajveer Singhal | @rajveeringhall 2005</p>
    </div>
""", unsafe_allow_html=True)
