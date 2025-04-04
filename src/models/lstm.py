import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_rate=0.3, learning_rate=0.001, optimizer='Adam', bidirectional=True,
                 device=None):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        print(f"🔄 Using device: {self.device} ({num_gpus} GPUs available)")

        # Core LSTM
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            batch_first=True,
                            bidirectional=self.bidirectional)  # Bidirectional LSTM
        
        lstm_output_size = hidden_size * (2 if self.bidirectional else 1)  # Adjust hidden size for bidirectional LSTM

        # Fully connected output layers
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Send model to device
        self.to(self.device)
        
        # initialise weights using He initialisation
        self.initialize_weights()

        # Enable multi-GPU support if multiple GPUs are available
        if num_gpus > 1:
            self = nn.DataParallel(self)  # Ensure it's on the correct device

        # Assign optimizer to `self.module` if using DataParallel
        if isinstance(self, nn.DataParallel):
            # Optimizer setup
            if optimizer == "Adam":
                self.module.optimizer = optim.Adam(self.module.parameters(), lr=learning_rate)
            elif optimizer == "SGD":
                self.module.optimizer = optim.SGD(self.module.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer == "AdamW":
                self.module.optimizer = optim.AdamW(self.module.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            self.module.criterion = nn.CrossEntropyLoss()
        else:
            # Optimizer setup
            if optimizer == "Adam":
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            elif optimizer == "SGD":
                self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer == "AdamW":
                self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            self.criterion = nn.CrossEntropyLoss()
        
        # debug line
        print(f"DEBUG: Optimizer initialized? {'optimizer' in self.__dict__}")

    def initialize_weights(self):
        """
        Initialize model weights using He (Kaiming) initialization.
        He initialization (also known as Kaiming initialization) is a weight initialization method designed to help deep neural networks train faster and prevent vanishing/exploding gradients. It is especially useful for layers with ReLU activation functions.

        ✅ Why Use He Initialization?
        1. Prevents Vanishing/Exploding Gradients – Ensures stable learning in deep networks.
        2. Speeds Up Convergence – Helps gradients flow properly during backpropagation.
        3. Optimized for ReLU Activation – Distributes weights in a way that keeps activations properly scaled.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set bias to zero

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take the last hidden state
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def train_model(self, train_loader, val_loader=None, epochs=50, patience=10, save_path=None):
        '''
        Train the model using the provided DataLoader and optional validation DataLoader.
        Saves the best model based on the validation accuracy.
        '''
        # CUDA
        torch.cuda.empty_cache()  # Free unused memory
        print("✅ Running on CUDA!" if self.device.type == 'cuda' else "❌ Still on CPU...")

        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch_X, batch_y in train_loader:
                # Move data to GPU if available
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                # GPU check
                # print("Data device:", batch_X.device)
                
                # Ensure optimiser is correctly accessed if using DataParallel
                if isinstance(self, nn.DataParallel):
                    model = self.module  # Get the underlying model
                else:
                    model = self  # Use the model as is

                model.optimizer.zero_grad()  # Reset gradients; also use the correct optimiser from the model
                outputs = model(batch_X)  # Forward pass
                loss = model.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backpropagation
                model.optimizer.step()  # Update weights

                total_loss += loss.item()
                correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
                total += batch_y.size(0)
            
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                history['val_acc'].append(val_acc)
                print(f"Validation Acc: {val_acc:.4f}")

                # save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    # only save the model if a save path is provided
                    if save_path is not None:
                        torch.save(self.state_dict(), save_path)
                        print(f"✅ Model saved at {save_path} (Best Validation Acc: {best_val_acc:.4f})")
                # ealry stopping
                else:
                    epochs_no_improve += 1
                    print(f"No improvement ({epochs_no_improve}/{patience}).")

                if epochs_no_improve >= patience:
                    print(f"Stopping early! No improvement for {patience} epochs.")
                    break

        print("Training complete!")        
        return history
    
    def evaluate(self, data_loader):
        '''
        Evaluate the model on a dataset and return accuracy
        '''
        self.eval()  # Set model to evaluation mode, dropout disabled in eval mode. 
        correct, total = 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                # Move data to GPU if available
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self(batch_X)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total

    def predict(self, X):
        '''
        Make predictions on new data
        '''
        self.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            # Move data to GPU if available
            X = X.to(self.device)
            outputs = self(X)
            return torch.argmax(outputs, dim=1)
        
    def load_model(self, file_path):
        '''
        Load model from a file
        '''
        # Load model to the correct device
        self.load_state_dict(torch.load(file_path, map_location=self.device))  
        # Ensure model is on the correct device
        self.to(self.device)  
        self.eval()
        print(f"🔄 Model loaded from {file_path}")
