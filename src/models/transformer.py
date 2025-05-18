#!/usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    
    This adds positional information to the input embeddings, which is essential
    for transformer models since they don't inherently capture sequence order.
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not model parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size,
                 dropout_rate=0.2, learning_rate=0.001, optimizer='Adam', 
                 use_conv1d=False, use_auxiliary=False, aux_weight=0.1,
                 pooling_strategy='last', use_mask=False, gradient_clip=1.0,
                 device=None):
        """
        Transformer model for time series classification.
        
        Args:
            input_size: Number of features in the input
            d_model: The dimension of the transformer model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer encoder layers
            output_size: Number of classes
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer to use ('Adam', 'SGD', or 'AdamW')
            use_conv1d: Whether to use Conv1D preprocessing
            use_auxiliary: Whether to use auxiliary task
            aux_weight: Weight for auxiliary loss
            pooling_strategy: How to pool sequence outputs ('last', 'mean', 'learnable')
            use_mask: Whether to use attention masking for variable length sequences
            gradient_clip: Value for gradient clipping
            device: Device to run model on
        """
        super(TransformerClassifier, self).__init__()
        self.use_conv1d = use_conv1d
        self.use_auxiliary = use_auxiliary
        self.aux_weight = aux_weight
        self.pooling_strategy = pooling_strategy
        self.use_mask = use_mask
        self.gradient_clip = gradient_clip
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        print(f"🔄 Using device: {self.device} ({num_gpus} GPUs available)")
        
        # Add Conv1D pre-processing to extract local temporal features
        if self.use_conv1d:
            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            # Transformer input size is now out_channels from Conv1D
            transformer_input_size = 32
        else:
            self.conv1d = None
            transformer_input_size = input_size
            
        # Input projection to match d_model dimensionality
        self.input_projection = nn.Linear(transformer_input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Transformer encoder - using newer version with normalization first pattern
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4*d_model,
            dropout=dropout_rate, 
            batch_first=True,
            norm_first=True  # Apply normalization before attention - helps training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Learnable pooling weights if using learnable pooling strategy
        if pooling_strategy == 'learnable':
            self.pool_weights = nn.Parameter(torch.ones(1) / d_model)
        
        # Fully connected layers for classification with ReLU instead of Softmax
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),  # Using ReLU instead of Softmax here
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
        
        # Optional auxiliary task (e.g., predicting peak count in time series)
        if self.use_auxiliary:
            self.aux_output = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Output: predicted peak count
            )
            
        self.dropout = nn.Dropout(dropout_rate)
        self.to(self.device)
        self.initialize_weights()
        
        # Set up parallel processing if multiple GPUs available
        if num_gpus > 1:
            self = nn.DataParallel(self)
            
        # Set up optimizer and loss functions
        if isinstance(self, nn.DataParallel):
            if optimizer == "Adam":
                self.module.optimizer = optim.Adam(self.module.parameters(), lr=learning_rate)
            elif optimizer == "SGD":
                self.module.optimizer = optim.SGD(self.module.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer == "AdamW":
                self.module.optimizer = optim.AdamW(self.module.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            # Learning rate scheduler with warmup
            self.module.scheduler = self._get_lr_scheduler(self.module.optimizer)
            self.module.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.module.aux_criterion = nn.MSELoss()
        else:
            if optimizer == "Adam":
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            elif optimizer == "SGD":
                self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer == "AdamW":
                self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            # Learning rate scheduler with warmup
            self.scheduler = self._get_lr_scheduler(self.optimizer)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.aux_criterion = nn.MSELoss()
            
        print(f"DEBUG: Optimizer initialized? {'optimizer' in self.__dict__}")
        
    def _get_lr_scheduler(self, optimizer):
        """
        Create a learning rate scheduler with warmup and cosine decay.
        Uses OneCycleLR which provides both warmup and annealing.
        """
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimizer.param_groups[0]['lr'],
            steps_per_epoch=10,  # Will be updated in train_model
            epochs=50,           # Will be updated in train_model
            pct_start=0.1,       # Spend 10% of training in warmup
            anneal_strategy='cos'  # Cosine annealing
        )
        
    def initialize_weights(self):
        """
        Initialize model weights using Xavier and He (Kaiming) initialization.
        
        Xavier initialization is used for the transformer components, while 
        He initialization is used for the linear layers with ReLU activations.
        """
        # Xavier initialization for transformer components
        for name, param in self.named_parameters():
            if 'transformer' in name and 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                    
        # He (Kaiming) initialization for fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x, mask=None):
        # Apply Conv1D preprocessing if needed
        if self.use_conv1d:
            # Change from [batch, seq_len, features] to [batch, features, seq_len]
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            # Change back to [batch, seq_len, features]
            x = x.permute(0, 2, 1)
            
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate attention mask if needed
        if self.use_mask and mask is None:
            # Create a simple padding mask (all tokens attended to)
            # In real usage, you'd compute this from actual sequence lengths
            mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        
        # Apply transformer encoder, with or without mask
        if self.use_mask:
            output = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(x)
        
        # Pooling strategies
        if self.pooling_strategy == 'last':
            # Use the representation from the last token for classification
            context = output[:, -1, :]  # [batch, d_model]
        elif self.pooling_strategy == 'mean':
            # Global average pooling across sequence length
            context = torch.mean(output, dim=1)  # [batch, d_model]
        elif self.pooling_strategy == 'learnable':
            # Weighted pooling with learnable weights
            # Apply softmax to get attention weights that sum to 1
            attn_weights = F.softmax(self.pool_weights, dim=0).unsqueeze(0).unsqueeze(0)
            # Apply attention weights and sum across sequence length
            context = torch.sum(output * attn_weights, dim=1)  # [batch, d_model]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Apply dropout for regularization
        features = self.dropout(context)
        
        # Generate classification logits
        class_logits = self.fc_layers(features)
        
        # If using auxiliary task, return both main and auxiliary outputs
        if self.use_auxiliary:
            aux_out = self.aux_output(features)
            return class_logits, aux_out.squeeze(1)
        else:
            return class_logits

    def train_model(self, train_loader, val_loader=None, epochs=50, patience=10, save_path=None):
        '''
        Train the model using the provided DataLoader and optional validation DataLoader.
        Saves the best model based on the validation accuracy.
        '''
        
        torch.cuda.empty_cache()
        print("✅ Running on CUDA!" if self.device.type == 'cuda' else "❌ Still on CPU...")
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        # Setup OneCycleLR scheduler properly with the right number of steps
        model = self.module if isinstance(self, nn.DataParallel) else self
        if hasattr(model, 'scheduler') and isinstance(model.scheduler, optim.lr_scheduler.OneCycleLR):
            total_steps = epochs * len(train_loader)
            model.scheduler = optim.lr_scheduler.OneCycleLR(
                model.optimizer,
                max_lr=model.optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=0.1,  # Spend 10% of training in warmup
                anneal_strategy='cos'  # Cosine annealing
            )
        
        for epoch in range(epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0
            
            for batch_X, batch_y_class in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y_class = batch_y_class.to(self.device)
                
                # Compute auxiliary targets (peak count) if using auxiliary task
                if self.use_auxiliary:
                    batch_y_aux = (batch_X[:, :, 0] > batch_X[:, :, 0].mean(dim=1, keepdim=True)).float().diff(dim=1).gt(0).sum(dim=1).float()
                
                # Handle DataParallel module if being used
                model = self.module if isinstance(self, nn.DataParallel) else self
                
                # Zero gradients
                model.optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self, 'use_mask') and self.use_mask:
                    # Create sequence mask based on lengths (in real usage, you'd have actual sequence lengths)
                    # This is a placeholder that treats all sequences as full length
                    seq_lengths = torch.full((batch_X.size(0),), batch_X.size(1), device=self.device)
                    mask = torch.arange(batch_X.size(1), device=self.device)[None, :] >= seq_lengths[:, None]
                    outputs = model(batch_X, mask)
                else:
                    outputs = model(batch_X)
                
                # Compute loss
                if self.use_auxiliary:
                    class_logits, aux_pred = outputs
                    class_loss = model.criterion(class_logits, batch_y_class)
                    aux_loss = model.aux_criterion(aux_pred, batch_y_aux)
                    loss = class_loss + self.aux_weight * aux_loss
                else:
                    class_logits = outputs
                    loss = model.criterion(class_logits, batch_y_class)
                    
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                
                # Optimization step
                model.optimizer.step()
                
                # Update OneCycleLR scheduler after each batch
                if hasattr(model, 'scheduler') and isinstance(model.scheduler, optim.lr_scheduler.OneCycleLR):
                    model.scheduler.step()
                
                # Calculate statistics
                total_loss += loss.item()
                correct += (torch.argmax(class_logits, dim=1) == batch_y_class).sum().item()
                total += batch_y_class.size(0)
                
            # Compute epoch statistics
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validation step
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                history['val_acc'].append(val_acc)
                print(f"Validation Acc: {val_acc:.4f}")
                
                # Update learning rate for ReduceLROnPlateau scheduler
                if hasattr(model, 'scheduler') and isinstance(model.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    model.scheduler.step(val_acc)
                
                # Early stopping and model saving
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                        print(f"✅ Model saved at {save_path} (Best Val Acc: {best_val_acc:.4f})")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement ({epochs_no_improve}/{patience}).")
                    
                if epochs_no_improve >= patience:
                    print(f"Stopping early! No improvement for {patience} epochs.")
                    break
                    
        print("Training complete!")
        return history
        
    def evaluate(self, data_loader):
        """Evaluate the model on the provided data loader"""
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self(batch_X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Get class predictions if auxiliary is used
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        return correct / total
    
    def predict(self, X):
        """Generate predictions for the given input tensor"""
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self(X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get class predictions if auxiliary is used
            return torch.argmax(outputs, dim=1)

class TransformerRegressor(nn.Module):
    """Transformer model for time series regression tasks."""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size=1,
                 dropout_rate=0.3, learning_rate=0.001, optimizer='Adam',
                 use_conv1d=False, pooling_strategy='last', 
                 use_mask=False, gradient_clip=1.0, device=None):
        """
        Initialize a transformer model for regression tasks.
        
        Args:
            input_size: Number of features in input data
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            output_size: Number of output features (default: 1 for single regression)
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer to use ('Adam', 'SGD', or 'AdamW')
            use_conv1d: Whether to use Conv1D preprocessing
            pooling_strategy: How to pool sequence outputs ('last', 'mean', 'learnable')
            use_mask: Whether to use attention masking for variable length sequences
            gradient_clip: Value for gradient clipping
            device: Device to run the model on
        """
        super(TransformerRegressor, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_conv1d = use_conv1d
        self.pooling_strategy = pooling_strategy
        self.use_mask = use_mask
        self.gradient_clip = gradient_clip
        
        # Add Conv1D pre-processing to extract local temporal features if needed
        if self.use_conv1d:
            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            # Transformer input size is now out_channels from Conv1D
            transformer_input_size = 32
        else:
            self.conv1d = None
            transformer_input_size = input_size
            
        # Input projection
        self.input_projection = nn.Linear(transformer_input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Transformer encoder with normalization first pattern for better stability
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True  # Apply normalization before attention - helps training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Learnable pooling weights if using learnable pooling strategy
        if pooling_strategy == 'learnable':
            self.pool_weights = nn.Parameter(torch.ones(1) / d_model)
            
        # Output projection for regression
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Move model to device
        self.to(self.device)
        
        # Initialize weights
        self.initialize_weights()
        
        # Set up optimizer and loss function
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
            
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=learning_rate,
            steps_per_epoch=10,  # Will be updated in train_model
            epochs=50,           # Will be updated in train_model
            pct_start=0.1,       # Spend 10% of training in warmup
            anneal_strategy='cos'  # Cosine annealing
        )
        self.criterion = nn.MSELoss()
        
    def initialize_weights(self):
        """
        Initialize model weights using Xavier and He (Kaiming) initialization.
        
        Xavier initialization is used for the transformer components, while 
        He initialization is used for the linear layers with ReLU activations.
        """
        # Xavier initialization for transformer components
        for name, param in self.named_parameters():
            if 'transformer' in name and 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                    
        # He (Kaiming) initialization for fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x, mask=None):
        # Apply Conv1D preprocessing if needed
        if self.use_conv1d:
            # Change from [batch, seq_len, features] to [batch, features, seq_len]
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            # Change back to [batch, seq_len, features]
            x = x.permute(0, 2, 1)
            
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate attention mask if needed
        if self.use_mask and mask is None:
            # Create a simple padding mask (all tokens attended to)
            # In real usage, you'd compute this from actual sequence lengths
            mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        
        # Apply transformer encoder, with or without mask
        if self.use_mask:
            output = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(x)
            
        # Pooling strategies
        if self.pooling_strategy == 'last':
            # Use the representation from the last token for regression
            context = output[:, -1, :]  # [batch, d_model]
        elif self.pooling_strategy == 'mean':
            # Global average pooling across sequence length
            context = torch.mean(output, dim=1)  # [batch, d_model]
        elif self.pooling_strategy == 'learnable':
            # Weighted pooling with learnable weights
            # Apply softmax to get attention weights that sum to 1
            attn_weights = F.softmax(self.pool_weights, dim=0).unsqueeze(0).unsqueeze(0)
            # Apply attention weights and sum across sequence length
            context = torch.sum(output * attn_weights, dim=1)  # [batch, d_model]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Apply dropout for regularization
        out = self.dropout(context)
        
        # Generate regression output
        out = self.output_projection(out)
        return out
        
    def train_model(self, train_loader, val_loader=None, epochs=50, patience=10, save_path=None):
        """Train the regression model on the provided data"""
        torch.cuda.empty_cache()
        print("✅ Running on CUDA!" if self.device.type == 'cuda' else "❌ Still on CPU...")
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Setup OneCycleLR scheduler properly with the right number of steps
        if hasattr(self, 'scheduler') and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            total_steps = epochs * len(train_loader)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=0.1,  # Spend 10% of training in warmup
                anneal_strategy='cos'  # Cosine annealing
            )
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self, 'use_mask') and self.use_mask:
                    # Create sequence mask based on lengths (in real usage, you'd have actual sequence lengths)
                    # This is a placeholder that treats all sequences as full length
                    seq_lengths = torch.full((batch_X.size(0),), batch_X.size(1), device=self.device)
                    mask = torch.arange(batch_X.size(1), device=self.device)[None, :] >= seq_lengths[:, None]
                    outputs = self(batch_X, mask)
                else:
                    outputs = self(batch_X)
                    
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                
                # Optimization step
                self.optimizer.step()
                
                # Update OneCycleLR scheduler after each batch
                if hasattr(self, 'scheduler') and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Update learning rate for ReduceLROnPlateau scheduler
                if hasattr(self, 'scheduler') and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                        print(f"✅ Model saved at {save_path} (Best Val Loss: {best_val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement ({epochs_no_improve}/{patience}).")
                    
                if epochs_no_improve >= patience:
                    print("🛑 Early stopping.")
                    break
                    
        print("Training complete!")
        return history
        
    def evaluate(self, data_loader):
        """Evaluate the model on the provided data"""
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self(batch_X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Get main predictions if tuple is returned
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(data_loader)
        
    def predict(self, X):
        """Generate predictions for the given input"""
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self(X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get main predictions if tuple is returned
            return outputs
            
    def load_model(self, file_path):
        """Load a saved model from file"""
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.to(self.device)
        self.eval()
        print(f"🔄 Model loaded from {file_path}")