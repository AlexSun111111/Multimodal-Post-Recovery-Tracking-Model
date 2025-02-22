import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch

def load_and_preprocess_numerical_data(data_file, numerical_columns, binary_columns, multiclass_columns):
    # Load dataset
    df = pd.read_csv(data_file)

    # Extract numerical features
    imputer = SimpleImputer(strategy='median')
    numerical_features = imputer.fit_transform(df[numerical_columns].values.astype(np.float32))

    # Normalize the numerical features
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    # Extract labels
    binary_labels = df[binary_columns].values.astype(np.float32)
    multiclass_labels = df[multiclass_columns].values.astype(np.int64)

    # Convert to tensors
    numerical_tensor = torch.tensor(numerical_features, dtype=torch.float32)
    binary_tensor = torch.tensor(binary_labels, dtype=torch.float32)
    multiclass_tensor = torch.tensor(multiclass_labels, dtype=torch.long)

    return numerical_tensor, binary_tensor, multiclass_tensor
