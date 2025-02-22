import torch
import torch.nn as nn

class NumericalModel(nn.Module):
    def __init__(self, num_numerical_features, num_binary, num_multiclass):
        super(NumericalModel, self).__init__()

        # Numerical feature processing part
        self.fc_num = nn.Sequential(
            nn.Linear(num_numerical_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # Binary classification output
        self.fc_binary = nn.Linear(256, num_binary)

        # Multiclass classification output
        self.fc_multiclass = nn.ModuleList([nn.Linear(256, n) for n in num_multiclass])

    def forward(self, numerical_features):
        # Process numerical features
        num_features = self.fc_num(numerical_features)

        # Output binary and multiclass results
        binary_logits = self.fc_binary(num_features)
        multiclass_logits = [fc(num_features) for fc in self.fc_multiclass]

        return binary_logits, multiclass_logits
