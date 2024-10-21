# project-4_machine_learning: BEP Stock Price Forecasting with LSTM

Overview

This project aims to forecast the daily adjusted closing price of Brookfield Renewable Partners (BEP) using a Bi-directional Long Short-Term Memory (LSTM) model. The guide takes you through the complete pipeline, from data collection to model development and evaluation.

Project Structure

This project is divided into several key components:

1. Importing Dependencies: setting up the required libraries for data handling, model training, and visualization.
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

2. Data Collection and Preprocessing: collecting the stock data from Yahoo Finance, preprocessing it to clean, transform, and engineer new features.
# Fetch BEP daily stock data from 2011-12-01 to 2023-10-17
# Reset the index to convert 'Date' from index to a regular column
# Drop unnecessary 'Close' column and renamed the Adj Close column
# Drop rows with missing values
# Calculate daily returns and volume change
# Calculate moving averages
# Handle infinities and drop remaining NaN values
# Save cleaned data to CSV and confirm success
# Check data consistency

3. Exploratory Data Analysis (EDA): gaining insights into the dataset by visualizing distributions, moving averages, and stock volatility. This gave us a better understanding of the stock's behavior over time.
# Check columns
# Display the first few rows of the DataFrame
# Check for any remaining NaN values
# Check for duplicate rows
# Summary statistics for numeric columns
# Check for columns with constant values
# Ensure that plots are displayed inline
# Load and check the data
# 1. Plot the Distribution of Adjusted Close Prices
# 2. Plot Volatility (30-Day Rolling Standard Deviation of Returns)
# 3. Scatter Plot: Volume Change vs. Returns
# 4. Comparison of Short-term and Long-term Moving Averages
# Find Buy (bullish) and Sell (bearish) signals
# Identify the peak price and its date
# Plot Adjusted Close prices and SMAs
# Plot buy signals (bullish crossovers)
# Plot sell signals (bearish crossovers)
# Annotate the peak price
# Set title and labels
# Add legend and grid
# Improve x-axis date formatting for readability
# Show the plot

4. Preparing Data for Model Training: splitting the data into training and testing sets, and applying scaling techniques.
# Load the data from the correct file
# Select features and target
# Prepare the input (X) and target (y)
# Split the data into training (80%) and testing (20%) sets
# Scale the data using MinMaxScaler
# Reshape X to 3D for LSTM: (samples, timesteps, features)

5. Building and Training the LSTM Model: creating a Bi-directional LSTM model that captures time dependencies crucial for predicting stock prices.
 LSTMs are chosen because of their ability to capture time dependencies, which is crucial for stock prediction.
 Model Architecture: The model is a Bi-directional LSTM with two LSTM layers followed by a Dropout layer for regularization.
 Training: use early stopping to prevent overfitting and stop the training when the validation loss starts increasing.
# Define the input shape (1 timestep, number of features)
# Build the Bi-directional LSTM model
# Input Layer
# First Bidirectional LSTM Layer
# Second Bidirectional LSTM Layer
# Output Layer: predicting one value (future stock price)
# Compile the model
# Print the model summary
# Early stopping to prevent overfitting
# Train the model

6. Model Evaluation: assessing model performance using metrics like Mean Square error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.
# Plot training & validation loss: the sharp decrease in the blue line at the beginning indicates that the model quickly learned to moniize error, the orange line shows the error on the validation set, it fluctuates more at first, indicating that the model is adapting, but it eventually stabilizes at a low value, closely tracking the training loss, suggesting that the model is not overfitting.
# Predict on the test set
# Inverse transform the predictions and actual values
# Plot the actual vs predicted values: the predicted prices (orange line) closely follow the actual prices (blue line). The LSTM model is able to capture the general trends and fluctuations of the stock, although there are minor discrepancies in some peaks and troughs. Overall, the high similarity between the two lines indicates that the model has performed well in forecasting the future stock prices.
# Calculate performance metrics: a value of 0.5538 for MSE is relatively low, suggesting that the model is making accurate predictions. A MAE of 0.5724 shows that the average prediction error is quite small. R-squared is 0.9624, a score close to 1 indicated that the model has strong predictive power.

7. Saving the Model: saving the trained model for future use.

8. Future 
