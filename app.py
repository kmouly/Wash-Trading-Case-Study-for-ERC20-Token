import streamlit as st
import pandas as pd
import numpy as np
import joblib
import chardet
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Load models and scalers
def load_model(model_name):
    return joblib.load(f"{model_name}.pkl")

def load_scaler():
    return joblib.load("scaler.pkl")

# Function to build transaction graph and compute centrality features
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['From'], row['To'], weight=row['Quantity'])

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    df['Degree_Centrality'] = df['From'].map(degree_centrality).fillna(0)
    df['Betweenness_Centrality'] = df['From'].map(betweenness_centrality).fillna(0)
    df['Closeness_Centrality'] = df['From'].map(closeness_centrality).fillna(0)

    df['Is_Self_Trade'] = (df['From'] == df['To']).astype(int)
    df['Is_Circular_Trade'] = df.duplicated(subset=['From', 'To'], keep=False).astype(int)
    df['Is_High_Activity'] = (df['Quantity'] > df['Quantity'].quantile(0.75)).astype(int)
    df['Quantity_Capped'] = df['Quantity'].clip(upper=df['Quantity'].quantile(0.95))

    return df

# Function to preprocess data
def preprocess_data(df, scaler, model_type):
    feature_orders = [
        'Quantity', 'Quantity_Log', 'Time_Diff', 'Rolling_Mean_Quantity',
        'Rolling_Std_Quantity', 'Is_Self_Trade', 'Is_Circular_Trade',
        'Is_High_Activity', 'Quantity_Capped', 'Degree_Centrality',
        'Betweenness_Centrality', 'Closeness_Centrality'
    ]

    df["Quantity_Log"] = np.log1p(df["Quantity"])
    df["Time_Diff"] = df["UnixTimestamp"].diff().fillna(0)
    df["Rolling_Mean_Quantity"] = df["Quantity"].rolling(window=5, min_periods=1).mean()
    df["Rolling_Std_Quantity"] = df["Quantity"].rolling(window=5, min_periods=1).std()

    df = build_graph(df)

    for feature in feature_orders:
        if feature not in df.columns:
            df[feature] = 0

    df.fillna(0, inplace=True)
    return scaler.transform(df[feature_orders])

# Load models
models = {
    "Isolation Forest": load_model("iso_forest_model"),
    "Local Outlier Factor": load_model("lof_model"),
    "One-Class SVM": load_model("best_model"),
}

scaler = load_scaler()

# Function to build and train LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Function to train LSTM and make predictions
def predict_lstm(df, scaler):
    # Prepare data for LSTM
    df_processed = preprocess_data(df, scaler, "LSTM")
    X = df_processed.reshape((df_processed.shape[0], df_processed.shape[1], 1))  # LSTM expects 3D input

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, np.zeros(X.shape[0]), epochs=10, batch_size=32, verbose=0)  # Train on dummy labels

    # Predict anomalies
    predictions = model.predict(X)
    return ["Anomaly" if pred > 0.5 else "Normal" for pred in predictions]

# Function to build and train ARIMA model
def predict_arima(df):
    df_processed = preprocess_data(df, scaler, "ARIMA")
    model = ARIMA(df_processed, order=(5,1,0))  # ARIMA(p,d,q) parameters
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=len(df))
    return ["Anomaly" if forecast[i] > np.percentile(forecast, 95) else "Normal" for i in range(len(df))]

# Function to predict using the ensemble model
def predict_ensemble(models, df, scaler):
    # Prepare the features as done for the individual models
    processed_data = preprocess_data(df, scaler, "Ensemble")

    # Predict using each individual model
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(processed_data)
        predictions[model_name] = ["Normal" if p == -1 else "Anomaly" for p in pred]

    # Voting: Create a DataFrame of predictions from each model
    pred_df = pd.DataFrame(predictions)

    # Majority voting: For each row (transaction), predict the majority vote
    ensemble_predictions = pred_df.mode(axis=1)[0].values  # The mode (majority) of predictions per row

    return ensemble_predictions

# Function to plot anomalies
def plot_anomalies(df, model_name):
    normal_df = df[df['Prediction'] == "Normal"]
    anomaly_df = df[df['Prediction'] == "Anomaly"]

    plt.figure(figsize=(10, 5))
    plt.scatter(normal_df.index, normal_df['Quantity'], color='green', label="Normal", alpha=0.5)
    plt.scatter(anomaly_df.index, anomaly_df['Quantity'], color='red', label="Anomaly", alpha=0.5)
    plt.xlabel("Transaction Index")
    plt.ylabel("Quantity")
    plt.title(f"Anomalies detected by {model_name}")
    plt.legend()
    st.pyplot(plt)

    # Count plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['Prediction'], palette={"Normal": "green", "Anomaly": "red"})
    plt.title(f"Normal vs Anomaly Count - {model_name}")
    st.pyplot(plt)

# Streamlit UI
st.title("Wash Trading Detection System")

# Upload File
uploaded_file = st.file_uploader("Upload CSV, TSV, or Excel file", type=["csv", "tsv", "xlsx"])

# Model Selection Dropdown
model_choice = st.selectbox("Select Model", ["Select", "One-Class SVM", "Isolation Forest", "Local Outlier Factor", "LSTM", "ARIMA", "Ensemble", "All"])

# Predict Button
if st.button("Predict"):

    if uploaded_file is None:
        st.error("Please upload a file before predicting.")
        st.stop()

    rawdata = uploaded_file.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding'] if result['encoding'] is not None else 'ISO-8859-1'
    uploaded_file.seek(0)  # Reset file pointer

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding=encoding)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Unsupported file format. Please upload a CSV, TSV, or Excel file.")
        st.stop()

    st.write("Preview of uploaded data:")
    st.write(df.head())

    predictions = {}

    if model_choice != "Select":
        if model_choice == "All":
            for model_name, model in models.items():
                processed_data = preprocess_data(df, scaler, model_name)
                pred = model.predict(processed_data)
                predictions[model_name] = ["Normal" if p == -1 else "Anomaly" for p in pred]

            # Ensemble prediction
            ensemble_preds = predict_ensemble(models, df, scaler)

            result_df = df.copy()
            for model_name, preds in predictions.items():
                result_df[model_name] = preds

            # Add ensemble prediction to the result
            result_df["Ensemble_Prediction"] = ensemble_preds

            # Count anomalies detected by each model and ensemble
            anomaly_counts = {model_name: sum(np.array(preds) == "Anomaly") for model_name, preds in predictions.items()}
            anomaly_counts["Ensemble"] = sum(ensemble_preds == "Anomaly")

            # Create a DataFrame for plotting
            anomaly_df = pd.DataFrame(list(anomaly_counts.items()), columns=["Model", "Anomalies Detected"])

            # Plot the bar chart for anomaly count per model
            plt.figure(figsize=(8, 5))
            sns.barplot(x="Model", y="Anomalies Detected", data=anomaly_df, palette="coolwarm")
            plt.title("Number of Anomalies Detected by Each Model (including Ensemble)")
            plt.xlabel("Anomaly Detection Model")
            plt.ylabel("Count of Anomalies")
            st.pyplot(plt)

            st.write("Prediction Results:")
            st.write(result_df.head())

        elif model_choice == "LSTM":
            lstm_preds = predict_lstm(df, scaler)
            result_df = df.copy()
            result_df["Prediction"] = lstm_preds

            st.write("LSTM Prediction Results:")
            st.write(result_df.head())

            plot_anomalies(result_df, "LSTM")

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download LSTM Predictions", csv, "lstm_predictions.csv", "text/csv", key='download-lstm')

        elif model_choice == "ARIMA":
            arima_preds = predict_arima(df)
            result_df = df.copy()
            result_df["Prediction"] = arima_preds

            st.write("ARIMA Prediction Results:")
            st.write(result_df.head())

            plot_anomalies(result_df, "ARIMA")

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download ARIMA Predictions", csv, "arima_predictions.csv", "text/csv", key='download-arima')

        elif model_choice == "Ensemble":
            # Ensemble prediction
            ensemble_preds = predict_ensemble(models, df, scaler)
            result_df = df.copy()
            result_df["Ensemble_Prediction"] = ensemble_preds

            st.write("Ensemble Prediction Results:")
            st.write(result_df.head())

            plot_anomalies(result_df, "Ensemble")

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download Ensemble Predictions", csv, "ensemble_predictions.csv", "text/csv", key='download-ensemble')

        else:
            model = models[model_choice]
            processed_data = preprocess_data(df, scaler, model_choice)
            pred = model.predict(processed_data)
            result_df = df.copy()
            result_df["Prediction"] = ["Anomaly" if p == -1 else "Normal" for p in pred]

            st.write(f"Predictions using {model_choice}:")
           
