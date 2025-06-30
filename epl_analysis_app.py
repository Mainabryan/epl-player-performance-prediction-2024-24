import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import uuid

# Set page configuration
st.set_page_config(page_title="EPL Player Analysis 2024/2025", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Visualizations", "Modeling", "Player Lookup"])

# Function to load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Function to preprocess data
def preprocess_data(data):
    # Convert percentage columns to numeric
    for col in ['Conversion %', 'Passes%', 'Crosses %', 'fThird Passes %', 'gDuels %', 'aDuels %', 'Saves %']:
        data[col] = data[col].str.rstrip('%').astype(float) / 100
    # Handle missing values
    data = data.fillna(0)
    return data

# Function to calculate shot accuracy
def calculate_shot_accuracy(row):
    if row['Shots'] == 0:
        return 0
    return (row['Shots On Target'] / row['Shots']) * 100

# Home page
if page == "Home":
    st.title("EPL Player Analysis Dashboard 2024/2025")
    st.markdown("""
    Welcome to the English Premier League Player Analysis App for the 2024/2025 season. 
    Upload your dataset and navigate through the sections to explore data, visualize insights, 
    train predictive models, and look up individual player statistics.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=200)

# Data Overview page
elif page == "Data Overview":
    st.title("Data Overview")
    uploaded_file = st.file_uploader("Upload EPL Player Stats CSV", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        data = preprocess_data(data)
        
        st.subheader("Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        st.subheader("Missing Values")
        st.write(data.isnull().sum())
        
        st.subheader("Basic Statistics")
        st.write(data.describe())
        
        st.subheader("First 5 Rows")
        st.dataframe(data.head())

# Visualizations page
elif page == "Visualizations":
    st.title("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload EPL Player Stats CSV", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        data = preprocess_data(data)
        
        # Boxplots
        st.subheader("Boxplots of Key Features")
        features = ['Goals', 'Shots', 'Assists', 'Minutes']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        for i, feature in enumerate(features):
            sns.boxplot(y=data[feature], ax=axes[i])
            axes[i].set_title(f"Boxplot of {feature}")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Top Scorers Bar Chart
        st.subheader("Top 10 Goal Scorers")
        top_scorers = data.nlargest(10, 'Goals')[['Player Name', 'Goals']]
        fig = px.bar(top_scorers, x='Player Name', y='Goals', title="Top 10 Goal Scorers")
        st.plotly_chart(fig)

# Modeling page
elif page == "Modeling":
    st.title("Predictive Modeling")
    uploaded_file = st.file_uploader("Upload EPL Player Stats CSV", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        data = preprocess_data(data)
        
        # Select features and target
        features = ['Minutes', 'Shots', 'Shots On Target', 'Big Chances Missed', 
                    'Hit Woodwork', 'Offsides', 'Crosses', 'Clearances', 'Tackles', 
                    'Aerial Duels', 'gDuels Won', 'aDuels %', 'Blocks', 'Yellow Cards', 'Fouls']
        X = data[features]
        y = data['Goals']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear Regression
        st.subheader("Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        st.write("Evaluation Metrics:")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_lr):.2f}")
        st.write(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred_lr):.2f}")
        
        # Random Forest
        st.subheader("Random Forest")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        st.write("Evaluation Metrics:")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf):.2f}")
        st.write(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred_rf):.2f}")
        
        # Predicted vs Actual Plot
        st.subheader("Predicted vs Actual Goals (Random Forest)")
        fig = px.scatter(x=y_test, y=y_pred_rf, labels={'x': 'Actual Goals', 'y': 'Predicted Goals'},
                        title="Predicted vs Actual Goals")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig)

# Player Lookup page
elif page == "Player Lookup":
    st.title("Player Lookup")
    uploaded_file = st.file_uploader("Upload EPL Player Stats CSV", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        data = preprocess_data(data)
        data['Shot Accuracy'] = data.apply(calculate_shot_accuracy, axis=1)
        
        player_name = st.text_input("Enter Player Name (e.g., Mohamed Salah)", "")
        
        if player_name:
            player_data = data[data['Player Name'].str.contains(player_name, case=False, na=False)]
            
            if not player_data.empty:
                player = player_data.iloc[0]
                st.subheader(f"Player Stats: {player['Player Name']}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(label="Total Goals", value=int(player['Goals']), 
                             delta_color="normal")
                with col2:
                    st.metric(label="Total Assists", value=int(player['Assists']), 
                             delta_color="normal")
                with col3:
                    st.metric(label="Minutes Played", value=int(player['Minutes']), 
                             delta_color="normal")
                with col4:
                    st.metric(label="Shot Accuracy", value=f"{player['Shot Accuracy']:.1f}%", 
                             delta_color="normal")
                
                st.write("Full Player Stats:")
                st.dataframe(player_data)
            else:
                st.error("Player not found. Please check the name and try again.")

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit | EPL Player Analysis 2024/2025")