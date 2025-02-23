# Multimodal-Post-Recovery-Tracking-Model (M-PRTM)

This repository contains the implementation of a **Multimodal Post-Recovery Tracking Model (M-PRTM)** aimed at predicting heart failure (HF) prognosis and detecting myocardial fibrosis using a combination of **textual**, **numerical**, and **cinematic (Cardiac MRI)** data. The model integrates multimodal data sources to offer a more accurate and holistic evaluation of patient outcomes compared to traditional single-modality models.

## Introduction
Heart failure (HF) is a complex disease that requires careful monitoring and treatment to improve patient outcomes. The current clinical practice often uses single-modality data such as clinical measurements or imaging, but these approaches may be insufficient for accurate prediction. This project introduces a **Multimodal Post-Recovery Tracking Model (M-PRTM)**, which combines textual (medical prescriptions), numerical (clinical metrics), and cinematic (Cardiac MRI) data to improve the prediction accuracy of heart failure prognosis.

![Overview](https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model-/blob/main/logo/Overview.png)

## Model Overview

The **Multimodal Post-Recovery Tracking Model (M-PRTM)** integrates three data modalities:

- **Cinematic Data**: Processed Cardiac MRI images used for myocardial fibrosis detection.
- **Numerical Data**: Clinical indicators, such as age, gender, blood pressure, and other key metrics.
- **Textual Data**: Medical records, including prescriptions and treatment history.

These modalities are processed through different specialized models:

- Cinematic data is processed using the DAE-Former model.
- Numerical data is processed via a fully connected neural network.
- Textual data is processed using a pre-trained BERT model.

The features from these modalities are fused using an attention mechanism to dynamically prioritize the most critical information for predictions.

![Framework](https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model-/blob/main/logo/Framework.png)


## Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model.git
cd Multimodal-Post-Recovery-Tracking-Model
```

### Example Code for Model Training
```python
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch import nn

# Define model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example data: Textual and Numerical inputs
texts = ['This is an example text for heart failure patient data']
numerical_data = torch.randn(1, 10)  # Random numerical features (e.g., age, weight, etc.)

# Tokenize textual data
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Pass through the model
outputs = model(**inputs)
text_features = outputs.last_hidden_state

# Combine with numerical features
combined_features = torch.cat((text_features[:, 0, :], numerical_data), dim=1)

# Forward pass through the final layers
output = nn.Linear(768 + 10, 1)(combined_features)  # Assuming binary output for simplicity
```

### Model Evaluation
During training, the model's performance is evaluated on a validation set, and accuracy is measured for each task (binary classification, multiclass classification, and days prediction).

## Code Structure

Overview of the codebase:

- `cinematic/`: Contains code related to cinematic data processing (e.g., Cardiac MRI images).
- `logo/`: Contains the logo and related images for the project.
- `numerical/`: Contains the code for processing numerical data (e.g., clinical metrics like age, blood pressure, etc.).
- `text/`: Contains the code for processing textual data (e.g., medical prescriptions, patient history).
- `.gitignore`: Specifies which files and directories to ignore in Git version control.
- `LICENSE`: The license for the project (MIT License).
- `README.md`: Project documentation providing an overview and setup instructions.
- `attention_fushion.py`: The script responsible for multi-modal feature fusion using attention mechanisms.
- `config.json`: Configuration file for the model, including hyperparameters and architecture details.
- `data_example.csv`: Example dataset for testing and validation of the model.


## Dataset
The dataset used in this project consists of preprocessed clinical, textual, and cinematic data for heart failure patients. The data includes:
- **Textual data**: Medical prescriptions and patient notes, processed using BERT.
- **Numerical data**: Clinical metrics such as blood pressure, heart rate, age, and weight.
- **Cinematic data**: Cardiac MRI images for detecting myocardial fibrosis, processed using the DAE-Former model.

You can access the dataset in the `data_example.csv` file.

## Results
The model achieved the following results during evaluation:
- **Fibrosis Segmentation**: 87.2% accuracy
- **Heart Failure Prognosis Prediction**: 96.5% accuracy
- **Event Prediction**: 97.6% accuracy
- **Risk Prediction**: 93.8% accuracy
  
![Framework](https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model-/blob/main/logo/Results.png)

- **(a)** Training curves of the DAE-former model showing the DSC and HD across 500 epochs.
- **(b)** Example of Myocardial Fibrosis prediction results. The original CMR image, ground truth label, and predicted fibrosis region show strong alignment with an accuracy of approximately 87%.
- **(c)** Training curves of the proposed M-PRTM model showing the test loss and accuracy across 500 epochs.
- **(d)** Prediction examples of clinical outcomes, including death, cause of death, rehospitalization days, and MACCES events.
- **(e)** Timeline of patient risk stratification and recovery prognosis. The model dynamically assesses risks and predicts critical events, such as heart failure recovery and ejection fraction decline.


## Contributions
Part of contributors:
- Xiumei Wang, College of Electronic and Optical Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Xingping Zhou, Institute of Quantum Information and Technology, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Jinyang Sun, Portland Institute, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Xi Chen, College of Integrated Circuit Science and Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
 
‡ zxp@njupt.edu.cn  

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments
Thanks to the funding sources, contributors, and any research institutions involved.
