#!/usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_rate=0.3, learning_rate=0.001, optimizer='Adam', bidirectional=True, use_conv1d=False,
                 use_attention=True, num_attention_heads=4, use_auxiliary=False, aux_weight=0.1,
                 device=None):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_auxiliary = use_auxiliary
        self.num_attention_heads = num_attention_heads
        self.aux_weight = aux_weight
        self.use_conv1d = use_conv1d

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        print(f"🔄 Using device: {self.device} ({num_gpus} GPUs available)")

        # Add Conv1D pre-processing to highlight bursts
        if self.use_conv1d:
            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            # LSTM input size is now out_channels from Conv1D
            lstm_input_size = 32
        else:
            self.conv1d = None
            lstm_input_size = input_size

        # Core LSTM
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        lstm_output_size = hidden_size * (2 if self.bidirectional else 1)

        # Optional attention
        if self.use_attention:
            self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_output_size,
            num_heads=self.num_attention_heads,
            batch_first=True)

        # Fully connected layers (modularized)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.Softmax(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )

        if self.use_auxiliary:
            self.aux_output = nn.Sequential(
                nn.Linear(lstm_output_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Output: predicted peak count
            )

        self.dropout = nn.Dropout(dropout_rate)
        self.initialize_weights()
        
        # Initialize optimizer and loss functions before DataParallel
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.aux_criterion = nn.MSELoss()
        
        # Move model to device first (important for proper initialization)
        self.to(self.device)
        
        # Apply DataParallel after moving to device and initializing optimizer
        self.is_parallel = False
        if num_gpus > 1:
            print(f"🚀 Enabling data parallel training on {num_gpus} GPUs")
            self.model = nn.DataParallel(self)
            self.is_parallel = True

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
            if isinstance(layer, nn.LSTM): # Apply to LSTM layers
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set bias to zero

    def forward(self, x):
        if self.use_conv1d:
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
            context = torch.mean(attn_output, dim=1)
        else:
            context = lstm_out[:, -1, :]

        features = self.dropout(context)
        class_logits = self.fc_layers(features)

        if self.use_auxiliary:
            aux_out = self.aux_output(features)
            return class_logits, aux_out.squeeze(1)
        else:
            return class_logits 

    def train_model(self, train_loader, val_loader=None, epochs=100, patience=10, save_path=None):
        '''
        Train the model using the provided DataLoader and optional validation DataLoader.
        Saves the best model based on the validation accuracy.
        '''
        
        torch.cuda.empty_cache()
        print("✅ Running on CUDA!" if self.device.type == 'cuda' else "❌ Still on CPU...")
        
        if self.is_parallel:
            print(f"🔄 Training with DataParallel on multiple GPUs")
            # Use the wrapper model for forward pass when in parallel mode
            training_model = self.model
        else:
            training_model = self

        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch_X, batch_y_class in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                batch_y_aux = (batch_X[:, :, 0] > batch_X[:, :, 0].mean(dim=1, keepdim=True)).float().diff(dim=1).gt(0).sum(dim=1).float()

                # Use the optimizer on the base model, not the wrapped one
                self.optimizer.zero_grad()
                
                # Forward pass through parallel model if available
                outputs = training_model(batch_X)

                if self.use_auxiliary:
                    class_logits, aux_pred = outputs
                    class_loss = self.criterion(class_logits, batch_y_class)
                    aux_loss = self.aux_criterion(aux_pred, batch_y_aux)
                    loss = class_loss + self.aux_weight * aux_loss
                else:
                    class_logits = outputs
                    loss = self.criterion(class_logits, batch_y_class)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (torch.argmax(class_logits, dim=1) == batch_y_class).sum().item()
                total += batch_y_class.size(0)

            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                history['val_acc'].append(val_acc)
                print(f"Validation Acc: {val_acc:.4f}")

                self.scheduler.step(val_acc)  # Fixed: Changed from model.scheduler to self.scheduler

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                        print(f"✅ Model saved at {save_path} (Best Validation Acc: {best_val_acc:.4f})")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement ({epochs_no_improve}/{patience}).")

                if epochs_no_improve >= patience:
                    print(f"Stopping early! No improvement for {patience} epochs.")
                    break

        print("Training complete!")
        return history

    def evaluate(self, data_loader):
        self.eval()
        correct, total = 0, 0
        
        # Use the wrapped model if we have DataParallel setup
        eval_model = self.model if hasattr(self, 'is_parallel') and self.is_parallel else self
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = eval_model(batch_X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        return correct / total

    def predict(self, X):
        self.eval()
        # Use the wrapped model if we have DataParallel setup
        pred_model = self.model if hasattr(self, 'is_parallel') and self.is_parallel else self
        
        with torch.no_grad():
            X = X.to(self.device)
            outputs = pred_model(X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return torch.argmax(outputs, dim=1)

    def load_model(self, file_path):
        # Load state dict
        state_dict = torch.load(file_path, map_location=self.device)
        
        # If we have DataParallel setup, need to handle loading differently
        if hasattr(self, 'is_parallel') and self.is_parallel:
            # Check if the saved model was also using DataParallel
            if all(k.startswith('module.') for k in state_dict.keys()):
                self.model.load_state_dict(state_dict)
            else:
                # Convert state_dict for DataParallel module
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict['module.' + k] = v
                self.model.load_state_dict(new_state_dict)
        else:
            # Remove 'module.' prefix if the saved model was using DataParallel
            if all(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
                
        self.eval()
        print(f"🔄 Model loaded from {file_path}")


# predictor
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1,
                 dropout_rate=0.3, learning_rate=0.001, optimizer='Adam', bidirectional=True, 
                 use_conv1d=False, use_attention=True, num_attention_heads=4,
                 device=None):
        super(LSTMRegressor, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        self.use_conv1d = use_conv1d
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"🔄 Using device: {self.device} ({num_gpus} GPUs available)")
        
        # Add Conv1D pre-processing to highlight temporal patterns
        if self.use_conv1d:
            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            # LSTM input size is now out_channels from Conv1D
            lstm_input_size = 32
        else:
            self.conv1d = None
            lstm_input_size = input_size

        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        lstm_output_size = hidden_size * (2 if self.bidirectional else 1)
        
        # Optional attention
        if self.use_attention:
            self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_output_size,
                                                      num_heads=self.num_attention_heads,
                                                      batch_first=True)
        
        # Fully connected layers (modularized like LSTMClassifier)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights before optimizer setup
        self.initialize_weights()
        
        # Initialize optimizer before wrapping with DataParallel
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.criterion = nn.MSELoss()
        
        # Add a scheduler like in the classifier
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Move to device first
        self.to(self.device)
        
        # Apply DataParallel if multiple GPUs are available
        self.is_parallel = False
        if num_gpus > 1:
            print(f"🚀 Enabling data parallel training on {num_gpus} GPUs")
            self.model = nn.DataParallel(self)
            self.is_parallel = True
            
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
            if isinstance(layer, nn.LSTM): # Apply to LSTM layers
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set bias to zero

    def forward(self, x):
        if self.use_conv1d:
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
            context = torch.mean(attn_output, dim=1)
        else:
            context = lstm_out[:, -1, :]

        features = self.dropout(context)
        output = self.fc_layers(features)
        return output

    def train_model(self, train_loader, val_loader=None, epochs=50, patience=10, save_path=None):
        torch.cuda.empty_cache()
        print("✅ Running on CUDA!" if self.device.type == 'cuda' else "❌ Still on CPU...")
        
        # Use the wrapped model if we have DataParallel setup
        if self.is_parallel:
            print(f"🔄 Training with DataParallel on multiple GPUs")
            training_model = self.model
        else:
            training_model = self

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = training_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Update learning rate scheduler based on validation loss
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    if save_path:
                        # Save the state dict of the base model, not the wrapped one
                        model_to_save = self.model.module if self.is_parallel else self
                        torch.save(model_to_save.state_dict(), save_path)
                        print(f"✅ Model saved at {save_path} (Best Val Loss: {best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement ({epochs_no_improve}/{patience}).")

                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping! No improvement for {patience} epochs.")
                    break

        print("Training complete!")
        return history

    def evaluate(self, data_loader):
        self.eval()
        total_loss = 0
        
        # Use the wrapped model if we have DataParallel setup
        eval_model = self.model if hasattr(self, 'is_parallel') and self.is_parallel else self
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = eval_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def predict(self, X):
        self.eval()
        # Use the wrapped model if we have DataParallel setup
        pred_model = self.model if hasattr(self, 'is_parallel') and self.is_parallel else self
        
        with torch.no_grad():
            X = X.to(self.device)
            return pred_model(X)

    def load_model(self, file_path):
        # Load state dict
        state_dict = torch.load(file_path, map_location=self.device)
        
        # If we have DataParallel setup, need to handle loading differently
        if hasattr(self, 'is_parallel') and self.is_parallel:
            # Check if the saved model was also using DataParallel
            if all(k.startswith('module.') for k in state_dict.keys()):
                self.model.load_state_dict(state_dict)
            else:
                # Convert state_dict for DataParallel module
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict['module.' + k] = v
                self.model.load_state_dict(new_state_dict)
        else:
            # Remove 'module.' prefix if the saved model was using DataParallel
            if all(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
                
        self.eval()
        print(f"🔄 Model loaded from {file_path}")