import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, config, num_binary, num_multiclass):
        super(TextModel, self).__init__()

        # BERT Text Processing
        self.bert = BertModel(config)
        self.fc_text = nn.Linear(config.hidden_size, 256)

        # Binary classification output
        self.fc_binary = nn.Linear(256, num_binary)

        # Multiclass classification output
        self.fc_multiclass = nn.ModuleList([nn.Linear(256, n) for n in num_multiclass])

    def forward(self, input_ids, attention_mask):
        # BERT output
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.fc_text(bert_output.pooler_output)

        # Output binary and multiclass results
        binary_logits = self.fc_binary(text_features)
        multiclass_logits = [fc(text_features) for fc in self.fc_multiclass]

        return binary_logits, multiclass_logits
