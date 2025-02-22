# Multimodal-Post-Recovery-Tracking-Model

This repository contains the implementation of a **Multimodal Post-Recovery Tracking Model (M-PRTM)** aimed at predicting heart failure (HF) prognosis and detecting myocardial fibrosis using a combination of **textual**, **numerical**, and **cinematic (Cardiac MRI)** data. The model integrates multimodal data sources to offer a more accurate and holistic evaluation of patient outcomes compared to traditional single-modality models.

## Problem Background

Heart failure (HF) is one of the leading causes of mortality worldwide, with increasing prevalence due to the aging population and the rise in cardiovascular diseases. Despite advances in treatment, the complexity of HF and the challenges in managing this condition persist, often leading to increased medical resource consumption. The current methods often focus on isolated data modalities, such as clinical measurements or imaging, which may lead to partial or inaccurate evaluations.

### Objective

Our goal is to propose a **composable strategy framework** that combines **video-text based language models** to enhance heart failure prognosis prediction. By leveraging **multimodal algorithms**, the model analyzes a range of data sources, including patient medical history, physical examination results, video (Cardiac MRI images), and textual data (prescriptions), to offer a more comprehensive assessment and optimized treatment plan for heart failure patients.

## Model Overview

The core of our research is the **Multimodal Post-Recovery Tracking Model (M-PRTM)**, which integrates three key data types:
1. **Numerical Data**: Patient demographic information (e.g., age, weight, vital signs) and clinical metrics (e.g., heart rate, blood pressure).
2. **Textual Data**: Medical records such as prescriptions, treatment plans, and clinical notes, processed using a **BERT-based model**.
3. **Cinematic Data**: CMR images, which are processed using the **DAE-Former model** for myocardial fibrosis detection.

The model employs an attention mechanism to dynamically fuse these features, prioritizing the most relevant data points for accurate prognosis and treatment planning.

### Key Features of the Model:
- **Multimodal Integration**: Combines three data sources (text, numerical, cinematic) to predict heart failure outcomes.
- **Dynamic Attention Mechanism**: Allocates attention to different data modalities based on their relevance for specific tasks.
- **High Accuracy**: Achieves **96.5% accuracy** in predicting clinical outcomes, outperforming traditional single-modality approaches.
- **Personalized Treatment**: Provides personalized risk assessments and early intervention recommendations, reducing unnecessary medical burdens for low-risk patients.

## Project Structure

The project consists of the following files and directories:

- **`v4.py`**: Python script containing data preprocessing, model definition, training loop, and evaluation.
- **`config.json`**: Model configuration file with hyperparameters and architecture settings.
- **`preprocessed_data_11.26.csv`**: Preprocessed dataset containing patient data, including clinical, textual, and imaging data.
- **`Train/`**: Directory for storing model weights and prediction results.
  - **`weights/`**: Subdirectory for saving the best model weights.
  - **`predictions/`**: Subdirectory for storing prediction results at different epochs.
  
## Requirements

The following dependencies are required to run the project:

- Python 3.x
- PyTorch
- Hugging Face Transformers (for BERT)
- Scikit-learn
- Pandas
- NumPy
- imbalanced-learn
- Matplotlib

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
Data Processing
The model handles three types of data:

Textual Data: Processed using a BERT-based tokenizer, which tokenizes medical prescriptions and clinical notes into embeddings that capture semantic relationships.
Numerical Data: Clinical metrics such as blood pressure, weight, and heart rate, processed using standard scaling techniques.
Cinematic Data: Cardiac MRI images are processed using the DAE-Former model, a specialized deep learning model for medical image segmentation, to detect myocardial fibrosis.
Training the Model
To train the model, run the following command:

bash
复制
编辑
python v4.py
This will initiate the training process, which includes data loading, preprocessing, model training, and evaluation. The model will be saved periodically during training.

Training Parameters:
Epochs: 200
Batch Size: 16
Optimizer: AdamW
Learning Rate: 1e-4
Early Stopping: Implemented to prevent overfitting
Loss Functions: Binary Cross-Entropy (BCE) loss for binary classification tasks, Cross-Entropy loss for multiclass classification tasks.
Evaluation
During training, the model's performance is evaluated on a validation set. The evaluation metrics include:

Accuracy for each binary and multiclass classification task.
Loss curves to track the training and validation progress.
Feature importance to understand which modalities contribute most to predictions.
Results
The model achieved the following results in our validation set:

Fibrosis Segmentation: 87.2% accuracy
HF Prognosis Prediction: 96.5% accuracy
Event Prediction: 97.6% accuracy
Risk Prediction: 93.8% accuracy
Future Directions
While the current model performs well in heart failure prognosis, there are several areas for future improvement:

Expanding to Other Domains: The model can be extended to other chronic diseases such as diabetes and oncology.
Data Diversity: Future work will include expanding the dataset to include multi-center data for better generalization.
Improving Interpretability: Further work on model explainability and transparency to ensure trust in clinical settings.
Citation
If you use this code in your research, please cite the following paper:

Composable Strategy Framework with Integrated Video-Text Based Language Models for Heart Failure Assessment
Journal of Medical AI, 2025

Acknowledgments
We would like to acknowledge the contribution of all the researchers and institutions involved in the development and validation of the model, as well as the datasets provided by the collaborating medical centers.
