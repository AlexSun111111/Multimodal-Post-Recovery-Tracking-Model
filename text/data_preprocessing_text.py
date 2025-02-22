import pandas as pd
from transformers import BertTokenizer
import torch


def load_and_preprocess_text(data_file, text_column='medical_indicator'):
    # Load dataset
    df = pd.read_csv(data_file)

    # Extract text data
    texts = df[text_column].values

    # Load BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert_localpath/')

    # Tokenize texts
    inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    return input_ids, attention_mask
