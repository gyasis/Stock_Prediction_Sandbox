import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def calculate_extended_dates(base_date, history_days, predict_days, factor=2):
    # Calculate extended periods
    extended_history_days = history_days * factor
    extended_predict_days = predict_days * factor

    # Extended start date
    start_date = base_date - pd.Timedelta(days=extended_history_days)
    
    # Extended end date
    end_date = base_date + pd.Timedelta(days=extended_predict_days)

    # Debugging information
    print(f"Base date: {base_date}")
    print(f"Extended history days: {extended_history_days}, Extended predict days: {extended_predict_days}")
    print(f"Calculated extended start_date: {start_date}, end_date: {end_date}")
    
    # Ensure the end_date is in the future relative to base_date
    if end_date <= base_date:
        raise ValueError("Calculated end_date is not in the future. Check base_date, predict_days, and factor.")

    return start_date, end_date


class StockDataLoader:
    def __init__(self, symbol, date, predict_days, history_days, batch_size, factor=3):
        self.symbol = symbol
        self.history_days = history_days
        self.predict_days = predict_days
        self.batch_size = batch_size
        self.factor = factor
        
        # Initialize MinMaxScaler
        self.scaler = MinMaxScaler()
        
        # Calculate extended dates
        self.start_date, self.end_date = calculate_extended_dates(date, history_days, predict_days, factor)
        print(f"Start date: {self.start_date}, End date: {self.end_date}")
        # Load and preprocess data
        self.df = self.load_data()
        self.train_loader, self.val_loader, self.test_loader = self.preprocess_data()
    
    def load_data(self):
        # Download stock data using the extended date range
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        print(data.head())
        print(data.tail())
        
        if data.empty:
            raise ValueError(f"No data found for symbol {self.symbol} between {self.start_date} and {self.end_date}")
        
        # Convert the date index to an integer (seconds since epoch)
        data['Date'] = data.index.astype(np.int64) // 10**9
        print(data.head())
        print(data.tail())
        
        
        # Reset the index to make 'Date' a column instead of an index
        # data.reset_index(drop=True, inplace=True)
        
        return data

    def preprocess_data(self):
        # Convert to numpy array, ensuring all columns are included
        scaled_data = self.scaler.fit_transform(self.df)

        X, y = [], []
        total_sequences = len(scaled_data) - self.history_days - self.predict_days + 1
        print(f"Total data points: {len(scaled_data)}")
        print(f"Total valid sequences possible: {total_sequences}")
        
        for i in range(total_sequences):
            X.append(scaled_data[i:i + self.history_days])
            y.append(scaled_data[i + self.history_days:i + self.history_days + self.predict_days, 3])

        if not X or not y:
            raise ValueError("Preprocessing resulted in empty X or y arrays.")
        
        X, y = np.array(X), np.array(y)
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.2)
        test_size = len(X) - train_size - val_size

        # Check if any of the splits is too small to contain even one sequence
        if train_size == 0 or val_size == 0 or test_size == 0:
            raise ValueError("One of the splits (train, val, test) has zero sequences. Adjust the date range, split ratio, or history/predict days.")

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        # Print sizes and first 20 samples for each split
        print(f"\nTrain size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
        print("\nTraining data (first 20 samples):")
        print(X_train[:20], y_train[:20])
        
        print("\nValidation data (first 20 samples):")
        print(X_val[:20], y_val[:20])
        
        print("\nTest data (first 20 samples):")
        print(X_test[:20], y_test[:20])

        train_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False)
        
        test_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
