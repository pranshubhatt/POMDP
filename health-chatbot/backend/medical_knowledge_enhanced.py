"""
Enhanced medical knowledge base combining structured data with natural language understanding.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class Symptom:
    name: str
    severity: float  # 0-1 scale
    description: List[str]  # Natural language variations
    related_symptoms: Set[str]

@dataclass
class Condition:
    name: str
    symptoms: Dict[str, float]  # Symptom name -> probability
    severity: float  # 0-1 scale
    description: str
    common_patterns: List[str]  # Common symptom combinations
    
class MedicalKnowledgeBase:
    def __init__(self):
        self.symptoms: Dict[str, Symptom] = {}
        self.conditions: Dict[str, Condition] = {}
        self.symptom_embeddings = {}  # For natural language matching
        self.load_base_knowledge()
        self.load_gretel_dataset()
        self.load_kaggle_dataset()
    
    def load_base_knowledge(self):
        """Load core medical knowledge"""
        # Core high-confidence conditions
        CONDITION_SYMPTOMS = {
            "common_cold": {
                "runny_nose": 0.9,
                "sore_throat": 0.8,
                "cough": 0.7,
                "fever": 0.5,
                "fatigue": 0.6
            },
            "flu": {
                "fever": 0.9,
                "body_aches": 0.8,
                "fatigue": 0.9,
                "cough": 0.7,
                "headache": 0.6
            },
            "covid19": {
                "fever": 0.8,
                "cough": 0.8,
                "fatigue": 0.7,
                "loss_of_taste": 0.6,
                "shortness_of_breath": 0.6
            }
        }

        SYMPTOM_SEVERITY = {
            "fever": 0.7,
            "cough": 0.5,
            "fatigue": 0.4,
            "shortness_of_breath": 0.8,
            "loss_of_taste": 0.6,
            "body_aches": 0.5,
            "runny_nose": 0.3,
            "sore_throat": 0.4,
            "headache": 0.5
        }

        # Initialize base symptoms
        for symptom, severity in SYMPTOM_SEVERITY.items():
            self.symptoms[symptom] = Symptom(
                name=symptom,
                severity=severity,
                description=[],  # Will be populated from datasets
                related_symptoms=set()
            )

        # Initialize base conditions
        for condition, symptoms in CONDITION_SYMPTOMS.items():
            self.conditions[condition] = Condition(
                name=condition,
                symptoms=symptoms,
                severity=sum(symptoms.values()) / len(symptoms),
                description="",  # Will be populated from datasets
                common_patterns=[]
            )

    def load_gretel_dataset(self):
        """
        Load and integrate the Gretel.ai symptom-to-diagnosis dataset
        for natural language understanding
        """
        try:
            # Load Gretel dataset (you'll need to download and place it in the correct path)
            gretel_data = pd.read_json("data/gretel_symptom_diagnosis.json")
            
            for _, row in gretel_data.iterrows():
                condition = row["output_text"].lower()
                symptoms_text = row["input_text"]
                
                # Update existing conditions or add new ones
                if condition not in self.conditions:
                    self.conditions[condition] = Condition(
                        name=condition,
                        symptoms={},
                        severity=0.5,  # Default severity
                        description="",
                        common_patterns=[]
                    )
                
                # Add natural language pattern
                self.conditions[condition].common_patterns.append(symptoms_text)
                
        except Exception as e:
            print(f"Warning: Could not load Gretel dataset: {e}")

    def load_kaggle_dataset(self):
        """
        Load and integrate the Kaggle disease prediction dataset
        for expanded condition coverage
        """
        try:
            # Load Kaggle dataset
            kaggle_data = pd.read_csv("data/kaggle_disease_prediction.csv")
            
            # Process symptoms and conditions
            for _, row in kaggle_data.iterrows():
                condition = row["prognosis"].lower()
                symptoms = [col for col in kaggle_data.columns[:-1] if row[col] == 1]
                
                if condition not in self.conditions:
                    self.conditions[condition] = Condition(
                        name=condition,
                        symptoms={},
                        severity=0.5,
                        description="",
                        common_patterns=[]
                    )
                
                # Update symptom probabilities
                for symptom in symptoms:
                    if symptom not in self.symptoms:
                        self.symptoms[symptom] = Symptom(
                            name=symptom,
                            severity=0.5,  # Default severity
                            description=[],
                            related_symptoms=set()
                        )
                    
                    # Update condition-symptom relationship
                    if symptom not in self.conditions[condition].symptoms:
                        self.conditions[condition].symptoms[symptom] = 0.7  # Default probability
                        
        except Exception as e:
            print(f"Warning: Could not load Kaggle dataset: {e}")

    def get_condition_probability(self, symptoms: List[str]) -> Dict[str, float]:
        """Calculate probability of each condition given symptoms"""
        probabilities = {}
        
        for condition_name, condition in self.conditions.items():
            # Calculate basic probability based on symptom match
            matched_symptoms = set(symptoms) & set(condition.symptoms.keys())
            if not matched_symptoms:
                continue
                
            prob = sum(condition.symptoms[s] for s in matched_symptoms) / len(condition.symptoms)
            probabilities[condition_name] = prob
            
        # Normalize probabilities
        if probabilities:
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}
            
        return probabilities

    def get_follow_up_questions(self, current_symptoms: List[str], condition_probs: Dict[str, float]) -> List[str]:
        """Generate follow-up questions based on current symptoms and probable conditions"""
        potential_symptoms = set()
        
        # Get top 3 most likely conditions
        top_conditions = sorted(condition_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for condition, _ in top_conditions:
            # Add symptoms associated with these conditions
            potential_symptoms.update(self.conditions[condition].symptoms.keys())
        
        # Remove symptoms we already know about
        potential_symptoms = potential_symptoms - set(current_symptoms)
        
        # Convert to questions
        questions = []
        for symptom in potential_symptoms:
            if symptom in self.symptoms:
                questions.append(f"Are you experiencing {symptom.replace('_', ' ')}?")
                
        return questions[:3]  # Return top 3 most relevant questions

    def get_severity_score(self, symptoms: List[str]) -> float:
        """Calculate overall severity score based on symptoms"""
        if not symptoms:
            return 0.0
            
        severity_sum = sum(self.symptoms[s].severity for s in symptoms if s in self.symptoms)
        return severity_sum / len(symptoms)

    def should_escalate(self, symptoms: List[str], severity_score: float) -> bool:
        """Determine if the case should be escalated to immediate medical attention"""
        HIGH_SEVERITY_THRESHOLD = 0.8
        CRITICAL_SYMPTOMS = {
            "shortness_of_breath",
            "chest_pain",
            "loss_of_consciousness",
            "severe_bleeding"
        }
        
        if severity_score > HIGH_SEVERITY_THRESHOLD:
            return True
            
        if any(s in CRITICAL_SYMPTOMS for s in symptoms):
            return True
            
        return False

# Singleton instance
knowledge_base = MedicalKnowledgeBase() 