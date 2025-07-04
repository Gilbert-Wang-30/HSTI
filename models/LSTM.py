import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """A simple LSTM-based model for TS1 time series prediction.
    
    This model uses a single-layer LSTM to process an input sequence of TS1 readings 
    and predicts the next (final) TS1 value. It expects sequences of length 60 with 
    one feature (TS1) per time step, and outputs a single scalar prediction.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        """
        Initialize the SimpleLSTMModel.
        
        Args:
            input_size (int): Number of features in the input at each time step. 
                              Default is 1 (for TS1 readings).
            hidden_size (int): Number of features in the LSTM hidden state. 
                               Default is 64.
            num_layers (int): Number of stacked LSTM layers. Default is 1 (a single-layer LSTM).
        """
        super(LSTMModel, self).__init__()
        # Define a single-layer LSTM. By default, this is unidirectional (not bidirectional).
        # Setting batch_first=True to accept input shape (batch, seq_len, input_size).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define a fully connected layer that maps from hidden_size to 1 output value.
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 60, input_size), 
                              where input_size=1 for TS1 sequences.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), containing the predicted 
                          final TS1 value for each sequence in the batch.
        """
        # Pass the input through the LSTM layer.
        # LSTM returns output for all timesteps and the final hidden, cell states (h_n, c_n).
        out, _ = self.lstm(x)  # out.shape = (batch_size, seq_len, hidden_size)
        # Take the output from the last time step (index -1) for each sequence in the batch.
        last_out = out[:, -1, :]  # shape = (batch_size, hidden_size)
        # Pass the last output through the fully connected layer to get the prediction.
        prediction = self.fc(last_out)  # shape = (batch_size, 1)
        return prediction
