# Multimodal-Post-Recovery-Tracking-Model (M-PRTM)

This repository contains the implementation of the **Multimodal Post-Recovery Tracking Model (M-PRTM)** for heart failure (HF) prognosis prediction and myocardial fibrosis detection. The framework integrates **textual medical records**, **structured numerical clinical indicators**, and **cine cardiac magnetic resonance (cine CMR)** data to support multimodal prediction of heart failure outcomes.

This repository accompanies the article:

**A composable multimodal framework for cine CMR-text–driven prediction of heart failure outcomes**

Repository link:  
https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model

---

## Introduction

Heart failure (HF) is a complex and multifactorial disease that requires careful monitoring, prognosis assessment, and treatment optimization. Conventional clinical prediction approaches often rely on single-modality information, such as clinical measurements, imaging features, or textual records alone. However, single-modality models may provide only partial information and may be insufficient for accurate prognosis prediction.

This project introduces the **Multimodal Post-Recovery Tracking Model (M-PRTM)**, which combines:

- textual medical records, including prescriptions and treatment history;
- structured numerical clinical indicators, such as demographic information, vital signs, laboratory indices, and other clinical metrics;
- cine CMR data for myocardial fibrosis-related feature extraction.

By integrating these heterogeneous data sources, M-PRTM aims to provide a more comprehensive assessment of patient status and improve the prediction of heart failure outcomes.

![Overview](https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model/blob/main/logo/Overview.png)

---

## Model Overview

The **Multimodal Post-Recovery Tracking Model (M-PRTM)** integrates three data modalities:

- **Cine CMR Data**: cine cardiac magnetic resonance images used for myocardial fibrosis detection and imaging feature extraction.
- **Numerical Data**: structured clinical indicators, such as age, gender, blood pressure, heart rate, body mass index, laboratory measurements, and other clinical variables.
- **Textual Data**: unstructured or semi-structured medical records, including prescriptions, medication history, and treatment-related textual information.

These modalities are processed through dedicated feature extraction modules:

- cine CMR data are processed using the DAE-Former model;
- numerical data are processed using a fully connected neural network;
- textual data are encoded using a pre-trained BERT-based model.

The extracted features are then fused through an attention-based multimodal fusion module, which dynamically prioritizes the most informative modalities and features for prognosis prediction.

![Framework](https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model/blob/main/logo/Framework.png)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model.git
cd Multimodal-Post-Recovery-Tracking-Model
```

Install the required Python packages according to your local environment. The core dependencies include:

```bash
pip install torch transformers numpy pandas scikit-learn
```

Additional packages may be required depending on the specific imaging, training, or evaluation scripts used in each module.

---

## Example Code for Model Training

```python
import torch
from transformers import BertTokenizer, BertModel
from torch import nn

# Define tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Example textual and numerical inputs
texts = ["This is an example text for heart failure patient data."]
numerical_data = torch.randn(1, 10)  # Example numerical features

# Tokenize textual data
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

# Extract textual features
outputs = bert_model(**inputs)
text_features = outputs.last_hidden_state[:, 0, :]

# Combine textual and numerical features
combined_features = torch.cat((text_features, numerical_data), dim=1)

# Example output layer for binary prediction
classifier = nn.Linear(768 + 10, 1)
output = classifier(combined_features)
```

---

## Model Evaluation

During training and validation, the model can be evaluated on multiple tasks, including:

- myocardial fibrosis segmentation;
- heart failure prognosis prediction;
- event prediction;
- risk prediction;
- auxiliary time-to-event estimation.

The evaluation metrics may include accuracy, sensitivity, specificity, AUPRC, Dice similarity coefficient, Hausdorff distance, MAE, RMSE, and R², depending on the task.

---

## Code Structure

The repository is organized as follows:

- `cinematic/`: code related to cine CMR processing and imaging feature extraction.
- `numerical/`: code for processing structured numerical clinical indicators.
- `text/`: code for processing textual medical records and prescriptions.
- `logo/`: figures and images used in the repository documentation.
- `attention_fushion.py`: script for multimodal feature fusion using attention mechanisms.
- `config.json`: configuration file for model hyperparameters and architecture settings.
- `data_example.csv`: example data file for demonstrating the model workflow.
- `.gitignore`: Git ignore file.
- `LICENSE`: MIT License file.
- `README.md`: project documentation.

---

## Data Availability

This repository provides the source code and an example data file (`data_example.csv`) for demonstrating the model workflow.

The full clinical dataset used in the study is not publicly available due to patient privacy and ethical restrictions. Access to the clinical data may be considered by the corresponding authors upon reasonable request and subject to institutional approval.

---

## Results

The model achieved the following representative results during evaluation:

- **Fibrosis Segmentation**: 87.2% accuracy
- **Heart Failure Prognosis Prediction**: 96.5% accuracy
- **Event Prediction**: 97.6% accuracy
- **Risk Prediction**: 93.8% accuracy

![Results](https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model/blob/main/logo/Results.png)

The result figure includes:

- **(a)** Training curves of the DAE-Former model showing segmentation-related metrics across training epochs.
- **(b)** Example myocardial fibrosis prediction results, including the original CMR image, reference fibrosis label, and predicted fibrosis region.
- **(c)** Training curves of the proposed M-PRTM model showing prediction performance across training epochs.
- **(d)** Representative prediction examples of clinical outcomes, including death, cause of death, rehospitalization days, and MACCE-related events.
- **(e)** Timeline of patient risk stratification and prognosis prediction, illustrating how the model dynamically assesses patient risk and supports individualized follow-up planning.

---

## Citation

If you use this repository, code, or example data in your research, please cite it as:

Sun J and Chen X 2026 *Multimodal Post-Recovery Tracking Model* GitHub repository. Available at: https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model

### BibTeX

```bibtex
@misc{sun2026mprtm,
  author       = {Jinyang Sun and Xi Chen},
  title        = {Multimodal Post-Recovery Tracking Model},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/AlexSun111111/Multimodal-Post-Recovery-Tracking-Model}
}
```

---

## Contributions

This repository accompanies the article:

**A composable multimodal framework for cine CMR-text–driven prediction of heart failure outcomes**

### Code Authors and Maintainers

- **Jinyang Sun**, School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University, Shanghai 200240, China
- **Xi Chen**, College of Integrated Circuit Science and Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China

The code was mainly developed by Jinyang Sun and Xi Chen, and organized and maintained by Jinyang Sun.

### Article Authors

- **Jianzhou Chen**, Department of Cardiology, Nanjing Drum Tower Hospital, State Key Laboratory of Pharmaceutical Biotechnology, Nanjing University, Nanjing 210008, China
- **Jinyang Sun**, School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University, Shanghai 200240, China
- **Xiumei Wang**, College of Electronic and Optical Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- **Xi Chen**, College of Integrated Circuit Science and Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- **Heyu Chu**, Department of Cardiology, Nanjing Drum Tower Hospital Clinical College of Nanjing Medical University, Nanjing 210008, China
- **Guo Song**, Department of Cardiology, Nanjing Drum Tower Hospital Clinical College of Nanjing Medical University, Nanjing 210008, China
- **Yuji Luo**, Department of Cardiology, Nanjing Drum Tower Hospital Clinical College of Nanjing Medical University, Nanjing 210008, China
- **Xingping Zhou**, Institute of Quantum Information and Technology, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- **Rong Gu**, Department of Cardiology, Nanjing Drum Tower Hospital, State Key Laboratory of Pharmaceutical Biotechnology, Nanjing University, Nanjing 210008, China

Correspondence: zxp@njupt.edu.cn and gurong.nju@163.com

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 12404365.

The authors thank Professor Xiaofei Li for insightful discussions.
