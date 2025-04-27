import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scapy.all import sniff, IP
import threading
import pickle
import time
from sklearn.preprocessing import StandardScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Select the last time step
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        
        # Calculate the size after flattening (depends on input_size)
        flat_size = 128 * (input_size // 4)  # Two pooling layers with kernel_size=2
        
        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class HybridModel(nn.Module):
    def __init__(self, input_size):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Size after pooling
        cnn_output_size = input_size // 2
        
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape: [batch_size, 1, features]
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        
        # Change from [batch_size, channels, seq_len] to [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm1(x)
        x = self.dropout2(x)
        x, _ = self.lstm2(x)
        x = self.dropout3(x)
        x = x[:, -1, :]  # Select the last time step
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class DeepLearningIDS:
    def __init__(self, model_type='hybrid', threshold=0.8):
        """
        Initialize the IDS with specified model architecture
        
        Args:
            model_type: Type of model to use ('lstm', 'cnn', or 'hybrid')
            threshold: Detection threshold for anomaly classification
        """
        self.model_type = model_type
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.packet_buffer = []
        self.buffer_lock = threading.Lock()
        self.feature_columns = None
        self.is_running = False
        self.alert_callbacks = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def build_model(self, input_shape):
        """Build the deep learning model based on specified architecture"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(input_shape[2])
        elif self.model_type == 'cnn':
            self.model = CNNModel(input_shape[2])
        else:  # hybrid model (CNN-LSTM)
            self.model = HybridModel(input_shape[2])
            
        self.model = self.model.to(self.device)
        return self.model
    
    def train(self, training_data, labels, validation_split=0.2, epochs=20, batch_size=64):
        """
        Train the model with labeled training data
        
        Args:
            training_data: Feature matrix of training samples
            labels: Binary labels (0 for normal, 1 for attack)
            validation_split: Portion of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Store feature columns for future reference
        if isinstance(training_data, pd.DataFrame):
            self.feature_columns = training_data.columns
            training_data = training_data.values
            
        # Scale features
        self.scaler = self.scaler.fit(training_data)
        scaled_data = self.scaler.transform(training_data)
        
        # Split into training and validation sets
        val_size = int(len(scaled_data) * validation_split)
        train_size = len(scaled_data) - val_size
        
        # Reshape based on model type
        if self.model_type in ['lstm', 'hybrid']:
            # Reshape to [samples, timesteps, features]
            # For LSTM, we'll use 1 timestep for simplicity
            reshaped_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
            input_shape = (None, 1, scaled_data.shape[1])
        else:
            # For CNN, reshape to [samples, channels, features]
            reshaped_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
            input_shape = (None, 1, scaled_data.shape[1])
        
        # Build the model
        self.build_model(input_shape)
        
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(reshaped_data, dtype=torch.float32)
        y_tensor = torch.tensor(labels.to_numpy().reshape(-1, 1), dtype=torch.float32)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
        val_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_samples += inputs.size(0)
            
            train_loss = train_loss / train_samples
            train_acc = train_correct / train_samples
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs >= 0.5).float()
                    val_correct += (predicted == targets).sum().item()
                    val_samples += inputs.size(0)
            
            val_loss = val_loss / val_samples
            val_acc = val_correct / val_samples
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - '
                  f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def extract_features(self, packet):
        """
        Extract relevant features from a network packet
        
        Args:
            packet: Network packet captured by scapy
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic IP header features
        if IP in packet:
            features['ip_len'] = packet[IP].len
            features['ip_ttl'] = packet[IP].ttl
            features['ip_proto'] = packet[IP].proto
            
            # Calculate entropy of IP payload as a feature to detect encrypted/obfuscated traffic
            if packet[IP].payload:
                payload = bytes(packet[IP].payload)
                if payload:
                    # Simple Shannon entropy calculation
                    occurrences = [payload.count(x) for x in range(256)]
                    entropy = 0
                    for x in occurrences:
                        if x:
                            p_x = float(x) / len(payload)
                            entropy -= p_x * np.log2(p_x)
                    features['payload_entropy'] = entropy
                else:
                    features['payload_entropy'] = 0
            else:
                features['payload_entropy'] = 0
                
        # TCP features
        if packet.haslayer('TCP'):
            features['tcp_sport'] = packet['TCP'].sport
            features['tcp_dport'] = packet['TCP'].dport
            features['tcp_flags'] = int(packet['TCP'].flags)
            features['tcp_window'] = packet['TCP'].window
        else:
            features['tcp_sport'] = 0
            features['tcp_dport'] = 0
            features['tcp_flags'] = 0
            features['tcp_window'] = 0
            
        # UDP features
        if packet.haslayer('UDP'):
            features['udp_sport'] = packet['UDP'].sport
            features['udp_dport'] = packet['UDP'].dport
            features['udp_len'] = packet['UDP'].len
        else:
            features['udp_sport'] = 0
            features['udp_dport'] = 0
            features['udp_len'] = 0
            
        # ICMP features
        if packet.haslayer('ICMP'):
            features['icmp_type'] = packet['ICMP'].type
            features['icmp_code'] = packet['ICMP'].code
        else:
            features['icmp_type'] = -1
            features['icmp_code'] = -1
        
        # Add time-based features
        features['timestamp'] = time.time()
        
        return features
    
    def packet_handler(self, packet):
        """Callback function for packet capture"""
        try:
            features = self.extract_features(packet)
            with self.buffer_lock:
                self.packet_buffer.append(features)
        except Exception as e:
            print(f"Error processing packet: {str(e)}")
    
    def preprocess_features(self, features_list):
        """Convert a list of feature dictionaries to a model-compatible format"""
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(features_list)
        
        # Handle missing columns from training data
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_columns]
        
        # Scale features
        scaled_features = self.scaler.transform(df.values)
        
        # Reshape based on model type
        if self.model_type in ['lstm', 'hybrid']:
            # Reshape to [samples, timesteps, features]
            return scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
        else:
            # Reshape to [samples, channels, features] for CNNs
            return scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
    
    def analyze_buffer(self):
        """Process buffered packets and detect anomalies"""
        while self.is_running:
            time.sleep(1)  # Analyze every second
            
            with self.buffer_lock:
                if not self.packet_buffer:
                    continue
                
                current_buffer = self.packet_buffer.copy()
                self.packet_buffer = []
            
            if not current_buffer:
                continue
                
            try:
                # Preprocess features
                X = self.preprocess_features(current_buffer)
                
                # Convert to PyTorch tensor
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                
                # Make predictions
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(X_tensor).cpu().numpy()
                
                # Check for anomalies
                for i, pred in enumerate(predictions):
                    if pred[0] >= self.threshold:
                        # Alert on detected intrusion
                        alert_info = {
                            'confidence': float(pred[0]),
                            'timestamp': current_buffer[i]['timestamp'],
                            'features': current_buffer[i]
                        }
                        
                        print(f"ALERT: Potential intrusion detected! Confidence: {pred[0]:.4f}")
                        
                        # Call any registered alert callbacks
                        for callback in self.alert_callbacks:
                            callback(alert_info)
                
            except Exception as e:
                print(f"Error analyzing packets: {str(e)}")
    
    def register_alert_callback(self, callback):
        """Register a callback function to be called when an intrusion is detected"""
        self.alert_callbacks.append(callback)
    
    def start_capturing(self, interface=None, filter=""):
        """
        Start capturing and analyzing packets
        
        Args:
            interface: Network interface to capture on (None for all interfaces)
            filter: BPF filter for packet capture
        """
        if self.model is None:
            raise ValueError("Model must be trained before starting capture")
        
        self.is_running = True
        
        # Start the analysis thread
        analysis_thread = threading.Thread(target=self.analyze_buffer)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Start packet capture
        print(f"Starting packet capture on interface: {interface or 'all'}")
        sniff(iface=interface, filter=filter, prn=self.packet_handler, store=0)
    
    def stop_capturing(self):
        """Stop the IDS"""
        self.is_running = False
        print("IDS stopped")
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler"""
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save")
    
    def load_model(self, model_path, scaler_path, feature_columns=None):
        """Load a previously trained model and scaler"""
        # Load feature columns
        self.feature_columns = feature_columns
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Determine input shape based on feature columns
        if feature_columns:
            input_size = len(feature_columns)
        else:
            raise ValueError("Feature columns must be provided")
        
        # Create model architecture
        if self.model_type == 'lstm':
            self.model = LSTMModel(input_size)
        elif self.model_type == 'cnn':
            self.model = CNNModel(input_size)
        else:  # hybrid
            self.model = HybridModel(input_size)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")


# Example usage function for demonstration
def example_alert_handler(alert_info):
    """Example callback function for handling alerts"""
    print(f"Alert received at {time.ctime(alert_info['timestamp'])}")
    print(f"Confidence: {alert_info['confidence']:.4f}")
    print(f"Alert details: {alert_info['features']}")
    # In a real system, you might log to a database, send an email, or trigger a response

def train_model_with_dataset(dataset_path):
    """
    Train the IDS model with a labeled dataset
    
    Args:
        dataset_path: Path to the CSV dataset with features and labels
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    data = pd.read_csv(dataset_path)
    
    # Separate features and labels
    # Assuming the last column is the label (0 for normal, 1 for attack)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    print(f"Dataset loaded: {len(data)} samples, {X.shape[1]} features")
    
    # Create and train the IDS
    ids = DeepLearningIDS(model_type='hybrid', threshold=0.85)
    history = ids.train(X, y, epochs=25, batch_size=128)
    
    # Save the trained model
    ids.save_model('ids_model.pt', 'ids_scaler.pkl')
    
    # Save feature columns for future reference
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    return ids

def start_ids_monitoring(model_path='ids_model.pt', scaler_path='ids_scaler.pkl', 
                         feature_cols_path='feature_columns.pkl', interface=None):
    """
    Start the IDS with a pre-trained model
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        feature_cols_path: Path to the saved feature columns
        interface: Network interface to monitor
    """
    # Load feature columns
    with open(feature_cols_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Create IDS and load model
    ids = DeepLearningIDS()
    ids.load_model(model_path, scaler_path, feature_columns)
    
    # Register alert handler
    ids.register_alert_callback(example_alert_handler)
    
    # Start monitoring
    try:
        print("Starting IDS monitoring. Press Ctrl+C to stop.")
        ids.start_capturing(interface=interface)
    except KeyboardInterrupt:
        print("\nStopping IDS...")
        ids.stop_capturing()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Learning Intrusion Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model with dataset')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring with pre-trained model')
    parser.add_argument('--dataset', type=str, help='Path to training dataset CSV')
    parser.add_argument('--interface', type=str, default=None, help='Network interface to monitor')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.dataset:
            print("Error: Dataset path required for training")
        else:
            ids = train_model_with_dataset(args.dataset)
            if args.monitor:
                # Start monitoring after training
                ids.register_alert_callback(example_alert_handler)
                try:
                    print("Starting IDS monitoring. Press Ctrl+C to stop.")
                    ids.start_capturing(interface=args.interface)
                except KeyboardInterrupt:
                    print("\nStopping IDS...")
                    ids.stop_capturing()
    elif args.monitor:
        start_ids_monitoring(interface=args.interface)
    else:
        print("Please specify either --train or --monitor mode")