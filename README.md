Project Name: Wash Trading Detection and Anomaly Modeling

The project contributers
2025

Description
This project aims to detect wash trading activities and other anomalies in cryptocurrency transaction data. It employs various machine learning models such as Isolation Forest, LOF, ARIMA, LSTM, and SVM for anomaly detection, along with preprocessing techniques to handle and normalize data from ERC20 and NEAR token transactions.

The project contains multiple trained models and a Jupyter notebook for analyzing wash trading patterns. The dataset includes transaction data that is preprocessed and normalized to detect fraudulent activities effectively.

Features
Anomaly Detection Models: Multiple models trained for detecting anomalies in transaction data:

LOF Model: Local Outlier Factor for detecting anomalies.
Isolation Forest: Isolation-based anomaly detection.
ARIMA Model: Time-series forecasting and anomaly detection.
LSTM Model: Long Short-Term Memory model for anomaly detection in time-series data.
SVM Model: Support Vector Machine for classification tasks.
Wash Trading Analysis: Detection of wash trading activities through statistical analysis in the Wash_Trading_Final_Colab.ipynb notebook.

Data Preprocessing: Tools and scripts for cleaning and normalizing cryptocurrency transaction data.

Installation
Follow these steps to set up the project locally:

1. Clone the repository:
git clone https://github.com/SruthiMangu133/Wash-Trading-Case-Study-for-ERC20-Token.git

2. Navigate into the project directory:
cd Wash-Trading-Case-Study-for-ERC20-Token

3. Create a virtual environment (optional but recommended):
python -m venv venv

4. Activate the virtual environment:
On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

