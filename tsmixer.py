# %%
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
from datetime import datetime
from IPython.display import display


# Import the new DataLoader class
from data_loader import StockDataLoader

# Configure logging for tracking the flow of the program
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TSMixerBlock(nn.Module):
    def __init__(self, input_dim, ff_dim, dropout=0.7):
        super(TSMixerBlock, self).__init__()

        # Time mixing layers: applied to each feature across all time steps
        self.time_mixing = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),  # back to the original dimension
            nn.Dropout(dropout)
        )
        
        # Feature mixing layers: applied to each time step across all features
        self.feature_mixing = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),  # back to the original dimension
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x is of shape [Batch Size, Sequence Length, Features]
        
        # Time Mixing: The Linear layer expects the last dimension to be input_dim
        res = x
        x = self.time_mixing(x)  # [Batch, Sequence Length, Features] remains
        
        # Add the residual connection (skip connection)
        x = x + res
        
        # Feature Mixing: The Linear layer expects the last dimension to be input_dim
        res = x
        x = self.feature_mixing(x)  # [Batch, Sequence Length, Features] remains
        
        # Add the residual connection (skip connection)
        return x + res

class TSMixerModel(nn.Module):
    def __init__(self, input_dim, history_days, predict_days, ff_dim, n_block):
        super(TSMixerModel, self).__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(input_dim, ff_dim) for _ in range(n_block)])
        self.temporal_projection = nn.Linear(history_days, predict_days)
        
        # Add a linear layer to reduce the feature dimension to 1 (for 'Close' price)
        self.output_projection = nn.Linear(input_dim, 1)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Debugging line
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # [Batch, Features, Sequence Length]
        x = self.temporal_projection(x)  # Project to [Batch, Features, predict_days]
        x = x.permute(0, 2, 1)  # Back to [Batch, predict_days, Features]
        x = self.output_projection(x)  # Reduce to [Batch, predict_days, 1]
        return x.squeeze(-1)  # Squeeze to get [Batch, predict_days]

def calculate_dates(base_date, history_days, predict_days):
    start_date = base_date - pd.Timedelta(days=history_days)
    end_date = base_date + pd.Timedelta(days=predict_days)
    return start_date, end_date



class StockPredictor:
    def __init__(self, symbol, date, history_days, predict_days, batch_size=32, ff_dim=64, n_block=8, learning_rate=1e-4):
        self.symbol = symbol
        self.date = pd.to_datetime(date)
        print(f"start date: {(self.date - pd.Timedelta(days=history_days)).date()}")
        print(f"end date: {(self.date + pd.Timedelta(days=predict_days)).date()}")
        self.start_date = (self.date - pd.Timedelta(days=history_days)).date()  # Start date is calculated here
        self.end_date = (self.date + pd.Timedelta(days=predict_days)).date()  # End date is calculated here
        self.history_days = history_days
        self.predict_days = predict_days
        self.batch_size = batch_size
        self.ff_dim = ff_dim
        self.n_block = n_block
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize DataLoader with the calculated start_date and end_date
        self.data_loader = StockDataLoader(
            symbol=self.symbol,
            date=self.date,
            history_days=self.history_days,
            predict_days=self.predict_days,
            batch_size=self.batch_size,
            factor=3
        )
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.train_loader, self.data_loader.val_loader, self.data_loader.test_loader
        self.scaler = self.data_loader.scaler
        self.df = self.data_loader.df

        self.model = None

    def build_model(self):
        logger.info("Building the model...")
        # Get the shape of the input data from the first batch of the train_loader
        X_sample, _ = next(iter(self.train_loader))
        input_dim = X_sample.shape[2]  # Number of features in the dataset
        self.model = TSMixerModel(input_dim, self.history_days, self.predict_days, self.ff_dim, self.n_block).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean Squared Error loss
        logger.info(f"Model built successfully with input_dim: {input_dim}")


    def train_model(self, epochs=30):
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            logger.error("Train loader is not set. Please run preprocess_data() first.")
            return
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            logger.error("Validation loader is not set. Please run preprocess_data() first.")
            return

        logger.info("Starting model training...")
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                # The output is already the 'Close' price prediction
                close_predictions = outputs
                
                loss = self.criterion(close_predictions.squeeze(), y_batch)  # Add .squeeze() here
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(self.train_loader.dataset)
            val_loss = self.evaluate_model(self.val_loader)
            logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'tsmixer_best_model.pth')

        self.model.load_state_dict(torch.load('tsmixer_best_model.pth'))
        logger.info("Model training completed.")


    def evaluate_model(self, loader):
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            logger.error("Validation loader is not set. Please run preprocess_data() first.")
            return

        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(loader, desc="Evaluating"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                
                # The output is already the 'Close' price prediction
                close_predictions = outputs
                
                loss = self.criterion(close_predictions.squeeze(), y_batch)  # Add .squeeze() here
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(loader.dataset)
        return val_loss



    def predict(self):
        logger.info("Starting prediction...")
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # List to store all predictions and their corresponding dates
        all_predictions = []
        all_dates = []
        
        for X_batch, _ in tqdm(self.test_loader, desc="Predicting"):
            X_batch = X_batch.to(self.device)
            
            # Predict for each sequence individually to avoid batch mismatches
            for i, sequence in enumerate(X_batch):
                sequence = sequence.unsqueeze(0)  # Add batch dimension
                
                # Get the corresponding dates for this sequence
                dates_for_sequence = self.df.index[self.history_days + i : self.history_days + i + self.predict_days]
                
                # Predict for the sequence
                with torch.no_grad():
                    pred = self.model(sequence)
                
                # Add the predictions and dates to the respective lists
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_dates.extend(dates_for_sequence)
        
        # Convert lists to numpy arrays
        predictions = np.array(all_predictions)
        dates = np.array(all_dates)
        
        # Inverse transform only the 'close' price predictions (assuming 'close' is the 2nd column, index 1)
        close_predictions = predictions.reshape(-1, 1)
        scaled_preds = self.scaler.inverse_transform(close_predictions)
        
        logger.info("Prediction completed.")
        return scaled_preds.flatten(), dates




    def predict_final(self):
        logger.info(f"Starting final prediction for {self.predict_days} business days...")

        # Convert self.start_date to a Unix timestamp
        start_date_timestamp = int(time.mktime(self.date.timetuple()))
        logger.info(f"Converted start_date to Unix timestamp: {start_date_timestamp}")

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Get the sequence ending at the start_date
        last_sequence = self.df[self.df['Date'] < start_date_timestamp].values[-self.history_days:]  # Get the last history_days worth of data before start_date
        last_sequence = self.scaler.transform(last_sequence)  # Scale the data
        last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)  # Convert to tensor and add batch dimension

        logger.info(f"Shape of last_sequence: {last_sequence.shape}")
        logger.info(f"Last sequence sample (before scaling):\n{self.df[self.df['Date'] < start_date_timestamp].head()}")

        predictions = []
        prediction_dates = []

        # Generate the next predict_days business days starting from the given start_date
        business_days = pd.bdate_range(start=self.date, periods=self.predict_days, freq='B')

        for date in tqdm(business_days, desc="Predicting final days"):
            # Skip the date if it's not in the DataFrame index (i.e., if it's a holiday or non-trading day)
            if date not in self.df.index:
                logger.info(f"Skipping {date} as it's not a trading day.")
                continue

            with torch.no_grad():
                # Use the full sequence for prediction
                pred = self.model(last_sequence)

                # Check if the prediction is 2D or 3D
                if pred.dim() == 2:
                    last_pred_close = pred[:, -1].unsqueeze(-1)  # Shape [1, 1]
                else:
                    last_pred_close = pred[:, -1, -1].unsqueeze(-1)  # Shape [1, 1]

                # Reshape pred_close to match the feature dimension of last_sequence, but only fill in the last feature
                pred_expanded = torch.zeros_like(last_sequence[:, -1:, :])  # Shape [1, 1, 7]
                pred_expanded[:, :, -1] = last_pred_close  # Fill only the "close" price

                # Append the prediction and date to the respective lists
                predictions.append(last_pred_close.cpu().numpy().flatten())
                prediction_dates.append(date)

                # Update the sequence: remove the first element and add the new prediction
                last_sequence = torch.cat((last_sequence[:, 1:, :], pred_expanded), dim=1)

        # Convert predictions to a numpy array
        predictions = np.array(predictions).reshape(-1, 1)
        logger.info(f"Predictions before inverse scaling:\n{predictions}")

        print(predictions)

        close_column_index = self.df.columns.get_loc("Close") 

        # Create a placeholder with zeros
        placeholder = np.zeros((predictions.shape[0], self.df.shape[1]))

        # Replace the 'Close' column in the placeholder with the predictions
        placeholder[:, close_column_index] = predictions.flatten()

        # Perform inverse transform on the entire placeholder array
        scaled_preds = self.scaler.inverse_transform(placeholder)

        # Extract the inverse transformed 'Close' prices
        scaled_preds = scaled_preds[:, close_column_index]
        logger.info(f"Scaled predictions (after inverse transform):\n{scaled_preds}")

        logger.info("Final prediction completed.")

        # Convert the prediction_dates to human-readable format
        human_readable_dates = [datetime.strftime(date, '%Y-%m-%d') for date in prediction_dates]

        print("DataFrame before extracting true values:")
        print(self.df.head())  # Print the first few rows for a quick look
        print(self.df.tail())  # Print the last few rows as well

        available_dates = business_days.intersection(self.df.index)



        print(f"Business days:\n{business_days}")
        print(f"Available dates:\n{available_dates}")

        # Now, retrieve the true values for the predicted dates
        true_values = self.df.loc[available_dates, 'Close'].values
        print(f"True values for the available dates:\n{true_values}")
        logger.info(f"True values for the available dates:\n{true_values}")

        # Print the predictions and true values
        for date, pred, true_val in zip(human_readable_dates, scaled_preds, true_values):
            print(f"Date: {date}, Predicted Close Price: {pred}, True Close Price: {true_val}")

        return scaled_preds, human_readable_dates, true_values




    def plot_validation(self, prediction_dates, predicted_close_prices, true_close_prices):
        # Calculate the MSE error
        mse_error = mean_squared_error(true_close_prices, predicted_close_prices)
        print(f"Mean Squared Error: {mse_error}")

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Date': prediction_dates,
            'Actual': true_close_prices,
            'Predicted': predicted_close_prices,
        })

        # Plot the actual vs predicted prices
        plt.figure(figsize=(14, 7))

        # Plot the actual values
        sns.lineplot(x=plot_df['Date'], y=plot_df['Actual'], label='Actual Values', color='green')

        # Plot the predicted values
        sns.lineplot(x=plot_df['Date'], y=plot_df['Predicted'], label='Predicted Values', color='red')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction vs Actual Prices')
        plt.legend()
        plt.show()

        # Optional: Return the DataFrame for further inspection
        return plot_df




# Assuming all necessary classes and functions (StockPredictor, StockDataLoader) are defined above

if __name__ == "__main__":
    def initialize_and_train(symbol, date, history_days, predict_days, batch_size, epochs):
        # Initialize the StockPredictor
        predictor = StockPredictor(symbol=symbol, date=date, history_days=history_days, predict_days=predict_days, batch_size=batch_size)
        
        # Build the model
        predictor.build_model()

        # Train the model
        try:
            predictor.train_model(epochs=epochs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("Out of memory error, reducing batch size and retrying...")
                batch_size //= 2  # Reduce batch size by half
                return initialize_and_train(symbol, date, history_days, predict_days, batch_size, epochs)
            else:
                raise e

        return predictor

    # Parameters for the model
    symbol = 'TSLA'
    date = pd.to_datetime('2023-03-01')  # The date from which predictions start
    history_days = 360  # Number of historical days used for training
    predict_days = 7  # Define the number of days to predict
    batch_size = 32
    epochs = 90

    # Initialize, preprocess, and train the model
    predictor = initialize_and_train(symbol=symbol, date=date, history_days=history_days, predict_days=predict_days, batch_size=batch_size, epochs=epochs)

    # Make final prediction for the specified number of days
    final_prediction, prediction_dates, true_values = predictor.predict_final()

    # Validation and scoring
    y_actual = true_values  # Use the true 'close' prices for the predicted dates
    tsmixer_preds = final_prediction  # Predictions are already only 'close' price

    print(y_actual)
    print(tsmixer_preds)

    # Ensure the lengths match
    if len(y_actual) != len(tsmixer_preds):
        raise ValueError(f"Mismatch in lengths: y_actual has {len(y_actual)} samples, while tsmixer_preds has {len(tsmixer_preds)} samples.")

    # Calculate evaluation metrics
    data = {
        'TSMixer': [
            mean_absolute_error(y_actual, tsmixer_preds),
            mean_squared_error(y_actual, tsmixer_preds)
        ]
    }



    # Display metrics with styling in Jupyter
    metrics_df = pd.DataFrame(data=data)
    metrics_df.index = ['mae', 'mse']
    display(metrics_df.style.highlight_min(
        color='lightgreen', 
        axis=1,
        props='color: black; background-color: lightgreen;'
    ))


    # Plot the validation results
    predictor.plot_validation(prediction_dates, final_prediction, true_values)

    # %%
