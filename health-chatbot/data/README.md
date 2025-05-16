# Medical Chatbot Datasets

This directory contains the datasets used by the medical chatbot for symptom analysis and diagnosis prediction.

## Required Datasets

1. **Disease-Symptom Knowledge Database (disease_symptom_db.csv)**
   - Source: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
   - Description: Contains disease-symptom relationships with severity scores
   - Required columns: Disease, Symptom, Weight/Severity

2. **Symptom Checker Dataset (symptom_checker.csv)**
   - Source: https://www.kaggle.com/datasets/paultimothymooney/symptom-checker
   - Description: Contains real-world symptom combinations and diagnoses
   - Required columns: Symptoms, Disease, Severity

3. **Medical Transcriptions (medical_transcriptions.csv)**
   - Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
   - Description: Used for natural language understanding of symptom descriptions
   - Required columns: transcription, medical_specialty

## Dataset Integration

1. Download the above datasets from Kaggle
2. Rename them according to the filenames mentioned above
3. Place them in this directory
4. Run the data preprocessing script: `python preprocess_datasets.py`

Note: If you don't have access to these exact datasets, the system will fall back to the built-in knowledge base. 