import pandas as pd

# Read the IY006B.csv file
df = pd.read_csv('/home/ianyang/stochastic_simulations/experiments/EXP-25-IY006/data/IY006B.csv')

# Sort by test accuracy and print top 5
top_models = df.sort_values(by='test_acc', ascending=False).head(5)
print(top_models[['d_model', 'nhead', 'num_layers', 'dropout_rate', 'learning_rate', 'batch_size', 'pooling_strategy', 'use_conv1d', 'use_auxiliary', 'gradient_clip', 'test_acc']])
