import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size:list, output_size, 
                 dropout_rate=0.3, learning_rate=0.001, 
                 device=None):
        # Call the parent class (nn.Module) constructor
        super(MLP, self).__init__() 

        # Automatically detect GPU(s) if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()  # Count number of available GPUs
        print(f"🔄 Using device: {self.device} ({num_gpus} GPUs available)")

        # Define the architecture of the model
        self.fc1 = nn.Linear(input_size, hidden_size[0])  # 1st Fully Connected Layer
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])  # 2nd Fully Connected Layer
        self.fc3 = nn.Linear(hidden_size[1], output_size)  # Output layer (Fully connected layer mapping hidden layer to output.)

        # Activation function (Exponential Activation with α=1)
        self.exp_activation = lambda x: torch.where(x > 0, x, torch.exp(x) - 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Move model to GPU if available
        self.to(self.device)

        # initialise weights using He initialisation
        self.initialize_weights()

        # Enable multi-GPU support if multiple GPUs are available
        if num_gpus > 1:
            self = nn.DataParallel(self)  # Wrap model in DataParallel

        # loss function and optimiser
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
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
        '''
        Forward pass: Pass the input through the layers of the model
        '''
        x = self.exp_activation(self.fc1(x))  # Exponential activation
        x = self.dropout(x)
        x = self.exp_activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer (logits)
        return x          # Return the final output

    def train_model(self, train_loader, val_loader=None, epochs=10, save_path=None):
        '''
        Train the model using the provided DataLoader and optional validation DataLoader.
        Saves the best model based on the validation accuracy.
        '''

        best_val_acc = 0.0
        # losses = []

        for epoch in range(epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0

            for batch_X, batch_y in train_loader:
                # Move data to GPU if available
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()  # Reset gradients
                outputs = self(batch_X)  # Forward pass
                loss = self.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

                total_loss += loss.item()
                correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
                total += batch_y.size(0)
            
            train_acc = correct / total
            # avg_loss = total_loss / len(train_loader)
            # losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                print(f"Validation Acc: {val_acc:.4f}")

                # save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # only save the model if a save path is provided
                    if save_path is not None:
                        torch.save(self.state_dict(), save_path)
                        print(f"✅ Model saved at {save_path} (Best Validation Acc: {best_val_acc:.4f})")

        print("Training complete!")        
        # return losses
    
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