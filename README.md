# Multimodal-Post-Recovery-Tracking-Model

This repository contains the implementation of a **Multimodal Post-Recovery Tracking Model (M-PRTM)** aimed at predicting heart failure (HF) prognosis and detecting myocardial fibrosis using a combination of **textual**, **numerical**, and **cinematic (Cardiac MRI)** data. The model integrates multimodal data sources to offer a more accurate and holistic evaluation of patient outcomes compared to traditional single-modality models.

## Abstract
Heart failure (HF) is one of the leading causes of mortality worldwide, and its complexity continues to challenge clinical management. This project develops a multimodal deep learning framework to predict heart failure prognosis using diverse data types, including clinical records, patient prescriptions, and cardiac MRI images. By integrating these modalities, our model provides a more comprehensive and precise prognosis, which aids in personalized treatment planning and early intervention.

## Introduction
Heart failure (HF) is a complex disease that requires careful monitoring and treatment to improve patient outcomes. The current clinical practice often uses single-modality data such as clinical measurements or imaging, but these approaches may be insufficient for accurate prediction. This project introduces a **Multimodal Post-Recovery Tracking Model (M-PRTM)**, which combines textual (medical prescriptions), numerical (clinical metrics), and cinematic (Cardiac MRI) data to improve the prediction accuracy of heart failure prognosis.

![M-PRTM Overview](https://github.com/your-username/Multimodal-Post-Recovery-Tracking-Model/blob/main/Logo/README.png)

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/your-username/Multimodal-Post-Recovery-Tracking-Model.git
cd Multimodal-Post-Recovery-Tracking-Model
pip install -r requirements.txt
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
During training, the model's performance is evaluated on a validation set, and accuracy is measured for each task (binary classification and multiclass classification).

## Code Structure
Overview of the codebase:
- `v4.py`: Main script for running simulations and training the model.
- `config.json`: Contains the model configuration, including hyperparameters and architecture details.
- `preprocessed_data_11.26.csv`: Preprocessed dataset with patient data.
- `Train/`: Directory for saving model weights and predictions.
  - `weights/`: Subdirectory for saving model weights.
  - `predictions/`: Subdirectory for saving predictions.

## Dataset
The dataset used in this project consists of preprocessed clinical, textual, and cinematic data for heart failure patients. The data includes:
- **Textual data**: Medical prescriptions and patient notes, processed using BERT.
- **Numerical data**: Clinical metrics such as blood pressure, heart rate, age, and weight.
- **Cinematic data**: Cardiac MRI images for detecting myocardial fibrosis, processed using the DAE-Former model.

You can access the dataset in the `preprocessed_data_11.26.csv` file.

## Results
The model achieved the following results during evaluation:
- **Fibrosis Segmentation**: 87.2% accuracy
- **Heart Failure Prognosis Prediction**: 96.5% accuracy
- **Event Prediction**: 97.6% accuracy
- **Risk Prediction**: 93.8% accuracy

## Contributions
List of contributors:
- Jinyang Sun, Portland Institute, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Xi Chen, College of Integrated Circuit Science and Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Xiumei Wang, College of Electronic and Optical Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Dandan Zhu, Institute of AI Education, East China Normal University, Shanghai 200333, China
- Xingping Zhou, Institute of Quantum Information and Technology, Nanjing University of Posts and Telecommunications, Nanjing 210003, China

† ddzhu@mail.ecnu.edu.cn  
‡ zxp@njupt.edu.cn  

## Citing
If you use this project or the associated paper in your research, please cite it as follows:
```bibtex
@article{
  title={Multimodal Post-Recovery Tracking Model for Heart Failure Prognosis},
  author={Jinyang Sun, Xi Chen, Xiumei Wang, Dandan Zhu and Xingping Zhou},
  journal={xxx},
  year={2025},
  volume={xx},
  pages={xx-xx}
}
```

## License
Specify the license under which the project is released.

## Acknowledgments
Thanks to the funding sources, contributors, and any research institutions involved.
