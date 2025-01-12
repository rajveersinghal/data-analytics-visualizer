# Dataset Visualizer and Analyzer

## Overview

The **Dataset Visualizer and Analyzer** is an interactive web application built using **Streamlit** that allows users to upload datasets (CSV or Excel format), explore the dataset, generate statistics, visualize correlations, and view feature importance. The app provides an easy way to perform exploratory data analysis (EDA) and visualize data in various formats.

Key features of the app include:
- Upload and explore CSV or Excel datasets
- Descriptive statistics and data summary
- Feature importance using Random Forest
- Correlation matrix and heatmaps
- Interactive visualizations like histograms, scatter plots, bar plots, pie charts, etc.

## Features

- **Upload Dataset**: Users can upload a dataset (CSV or Excel) to begin analysis.
- **About Dataset**: Get basic information about the dataset, including shape, column names, data types, and missing values.
- **Statistics**: View descriptive statistics and missing value information.
- **Feature Insights**: Get feature importance using Random Forest, along with correlation analysis and visualizations.
- **Dashboard**: Visualize the dataset through interactive charts such as histograms, scatter plots, bar plots, pie charts, and more.

## Technologies Used

- **Streamlit**: For building the web interface.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization (graphs and charts).
- **Plotly**: For interactive visualizations.
- **Scikit-Learn**: For machine learning tasks, such as Random Forest for feature importance.
- **Openpyxl**: For handling Excel files.

## Requirements

The following packages are required to run the app:

```bash
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
openpyxl
