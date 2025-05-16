"""
Preprocess and integrate medical datasets for the chatbot.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Set

def load_disease_symptom_db() -> pd.DataFrame:
    """Load and preprocess the disease-symptom knowledge database."""
    try:
        df = pd.read_csv('disease_symptom_db.csv')
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print("Warning: disease_symptom_db.csv not found")
        return pd.DataFrame()

def load_symptom_descriptions() -> pd.DataFrame:
    """Load and preprocess the symptom descriptions."""
    try:
        df = pd.read_csv('symptom_descriptions.csv')
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print("Warning: symptom_descriptions.csv not found")
        return pd.DataFrame()

def load_symptom_precautions() -> pd.DataFrame:
    """Load and preprocess the symptom precautions."""
    try:
        df = pd.read_csv('symptom_precautions.csv')
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print("Warning: symptom_precautions.csv not found")
        return pd.DataFrame()

def load_symptom_severity() -> pd.DataFrame:
    """Load and preprocess the symptom severity data."""
    try:
        df = pd.read_csv('symptom_severity.csv')
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print("Warning: symptom_severity.csv not found")
        return pd.DataFrame()

def load_medical_transcriptions() -> pd.DataFrame:
    """Load and preprocess the medical transcriptions."""
    try:
        df = pd.read_csv('medical_transcriptions.csv')
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print("Warning: medical_transcriptions.csv not found")
        return pd.DataFrame()

def create_integrated_knowledge_base():
    """Create an integrated knowledge base from all datasets."""
    # Load all datasets
    disease_symptom_df = load_disease_symptom_db()
    symptom_desc_df = load_symptom_descriptions()
    symptom_precautions_df = load_symptom_precautions()
    symptom_severity_df = load_symptom_severity()
    transcriptions_df = load_medical_transcriptions()
    
    # Create integrated knowledge base
    knowledge_base = {
        'diseases': {},
        'symptoms': set(),
        'symptom_descriptions': {},
        'symptom_precautions': {},
        'symptom_severity': {},
        'natural_language_examples': {}
    }
    
    # Process disease-symptom relationships
    if not disease_symptom_df.empty:
        for _, row in disease_symptom_df.iterrows():
            disease = str(row.get('disease', '')).lower().strip()
            symptoms = [s.lower().strip() for s in row.dropna()[1:] if isinstance(s, str)]
            
            if disease and symptoms:
                if disease not in knowledge_base['diseases']:
                    knowledge_base['diseases'][disease] = {'symptoms': set()}
                knowledge_base['diseases'][disease]['symptoms'].update(symptoms)
                knowledge_base['symptoms'].update(symptoms)
    
    # Process symptom descriptions
    if not symptom_desc_df.empty:
        for _, row in symptom_desc_df.iterrows():
            symptom = str(row.get('symptom', '')).lower().strip()
            description = str(row.get('description', '')).strip()
            if symptom and description:
                knowledge_base['symptom_descriptions'][symptom] = description
    
    # Process symptom precautions
    if not symptom_precautions_df.empty:
        for _, row in symptom_precautions_df.iterrows():
            symptom = str(row.get('symptom', '')).lower().strip()
            precautions = [p.strip() for p in row.dropna()[1:] if isinstance(p, str)]
            if symptom and precautions:
                knowledge_base['symptom_precautions'][symptom] = precautions
    
    # Process symptom severity
    if not symptom_severity_df.empty:
        for _, row in symptom_severity_df.iterrows():
            symptom = str(row.get('symptom', '')).lower().strip()
            severity = float(row.get('weight', 0))
            if symptom:
                knowledge_base['symptom_severity'][symptom] = severity
    
    # Process medical transcriptions
    if not transcriptions_df.empty:
        for _, row in transcriptions_df.iterrows():
            specialty = str(row.get('medical_specialty', '')).lower().strip()
            description = str(row.get('transcription', '')).strip()
            keywords = str(row.get('keywords', '')).lower().split(',')
            
            if specialty and description:
                if specialty not in knowledge_base['natural_language_examples']:
                    knowledge_base['natural_language_examples'][specialty] = []
                example = {
                    'text': description,
                    'keywords': [k.strip() for k in keywords if k.strip()]
                }
                knowledge_base['natural_language_examples'][specialty].append(example)
    
    # Convert sets to lists for JSON serialization
    knowledge_base['symptoms'] = list(knowledge_base['symptoms'])
    for disease in knowledge_base['diseases']:
        knowledge_base['diseases'][disease]['symptoms'] = list(
            knowledge_base['diseases'][disease]['symptoms']
        )
    
    # Save integrated knowledge base
    with open('../medical_knowledge_enhanced.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print("Successfully created integrated knowledge base")

if __name__ == "__main__":
    create_integrated_knowledge_base()