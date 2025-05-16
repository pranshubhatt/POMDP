"""
Medical knowledge base containing symptoms, conditions, and their relationships.
With support for external dataset integration.
"""

from typing import Dict, List, Set, Optional, Union
import json
import pandas as pd
from pathlib import Path

# Hierarchical condition categories
CONDITION_CATEGORIES = {
    "respiratory": {
        "upper_respiratory": ["common_cold", "flu", "sinusitis"],
        "lower_respiratory": ["bronchitis", "pneumonia", "covid19"]
    },
    "neurological": {
        "headaches": ["migraine", "tension_headache"],
        "cognitive": ["concussion", "meningitis"]
    },
    "gastrointestinal": {
        "stomach": ["gastritis", "food_poisoning"],
        "intestinal": ["ibs", "gastroenteritis"]
    }
}

# Dictionary mapping conditions to their common symptoms with severity levels
CONDITION_SYMPTOMS = {
    "common_cold": [
        "runny nose", "congestion", "sore throat", "cough", "sneezing",
        "mild fever", "fatigue"
    ],
    "flu": [
        "high fever", "body aches", "fatigue", "cough", "headache",
        "sore throat", "chills"
    ],
    "covid19": [
        "fever", "dry cough", "fatigue", "loss of taste",
        "loss of smell", "difficulty breathing", "body aches"
    ],
    "migraine": [
        "severe headache", "nausea", "sensitivity to light",
        "sensitivity to sound", "visual aura"
    ],
    "sinusitis": [
        "facial pain", "nasal congestion", "thick nasal discharge",
        "reduced sense of smell", "headache"
    ],
    "bronchitis": [
        "persistent cough", "chest discomfort", "fatigue",
        "mild fever", "shortness of breath"
    ],
    "pneumonia": [
        "high fever", "severe cough", "difficulty breathing",
        "chest pain", "fatigue", "rapid breathing"
    ],
    "tension_headache": [
        "dull headache", "pressure around head", "neck pain",
        "shoulder tension", "mild sensitivity to light"
    ],
    "gastritis": [
        "stomach pain", "nausea", "bloating", "indigestion",
        "loss of appetite"
    ],
    "food_poisoning": [
        "nausea", "vomiting", "diarrhea", "stomach cramps",
        "fever", "weakness"
    ]
}

# Severity levels for symptoms (0-1 scale)
SYMPTOM_SEVERITY = {
    # Fever-related
    "mild fever": 0.3,
    "fever": 0.5,
    "high fever": 0.7,
    
    # Respiratory
    "runny nose": 0.2,
    "congestion": 0.3,
    "nasal congestion": 0.3,
    "thick nasal discharge": 0.4,
    "sore throat": 0.4,
    "cough": 0.4,
    "dry cough": 0.5,
    "persistent cough": 0.6,
    "severe cough": 0.8,
    "sneezing": 0.2,
    "shortness of breath": 0.7,
    "difficulty breathing": 0.9,
    "rapid breathing": 0.8,
    
    # Pain
    "headache": 0.4,
    "dull headache": 0.3,
    "severe headache": 0.8,
    "facial pain": 0.5,
    "chest pain": 0.8,
    "chest discomfort": 0.6,
    "stomach pain": 0.5,
    "stomach cramps": 0.6,
    
    # Neurological
    "sensitivity to light": 0.5,
    "mild sensitivity to light": 0.3,
    "sensitivity to sound": 0.5,
    "visual aura": 0.4,
    "pressure around head": 0.4,
    "neck pain": 0.4,
    "shoulder tension": 0.3,
    
    # Gastrointestinal
    "nausea": 0.5,
    "vomiting": 0.7,
    "diarrhea": 0.6,
    "bloating": 0.3,
    "indigestion": 0.4,
    "loss of appetite": 0.4,
    
    # General
    "fatigue": 0.4,
    "weakness": 0.5,
    "body aches": 0.5,
    "chills": 0.5,
    "loss of taste": 0.7,
    "loss of smell": 0.7,
    "reduced sense of smell": 0.4
}

# Risk factors that can affect condition severity
RISK_FACTORS = {
    "age_over_65": 1.5,
    "immunocompromised": 2.0,
    "pregnancy": 1.3,
    "diabetes": 1.4,
    "heart_disease": 1.5,
    "respiratory_condition": 1.6,
    "obesity": 1.3
}

# Typical condition durations (in days)
CONDITION_DURATIONS = {
    "common_cold": {"min": 7, "typical": 10, "max": 14},
    "flu": {"min": 3, "typical": 7, "max": 14},
    "covid19": {"min": 10, "typical": 14, "max": 21},
    "migraine": {"min": 0.5, "typical": 1, "max": 3},
    "sinusitis": {"min": 7, "typical": 14, "max": 21},
    "bronchitis": {"min": 10, "typical": 14, "max": 21},
    "pneumonia": {"min": 14, "typical": 21, "max": 28},
    "tension_headache": {"min": 0.5, "typical": 1, "max": 2},
    "gastritis": {"min": 2, "typical": 7, "max": 14},
    "food_poisoning": {"min": 1, "typical": 3, "max": 7}
}

# Questions to ask for each symptom for clarification
SYMPTOM_QUESTIONS = {
    "fever": [
        "What is your current temperature?",
        "How long have you had the fever?",
        "Does the fever come and go, or is it constant?"
    ],
    "cough": [
        "Is your cough dry or productive (bringing up mucus)?",
        "How frequent is the cough?",
        "Is the cough worse at night or in the morning?"
    ],
    "headache": [
        "How severe is your headache on a scale of 1-10?",
        "Where is the pain located?",
        "Are there any triggers for your headache?"
    ],
    "breathing": [
        "Do you feel short of breath at rest or only with activity?",
        "Can you take a deep breath without pain?",
        "Have you noticed any wheezing?"
    ],
    "pain": [
        "Can you rate your pain on a scale of 1-10?",
        "Is the pain constant or does it come and go?",
        "What makes the pain better or worse?"
    ],
    "gastrointestinal": [
        "Have you experienced any changes in appetite?",
        "Have you had any nausea or vomiting?",
        "Have you noticed any changes in your bowel movements?"
    ]
}

# Conditions that require immediate medical attention
EMERGENCY_SYMPTOMS = [
    "difficulty breathing",
    "severe chest pain",
    "severe headache with confusion",
    "high fever above 103°F (39.4°C)",
    "severe abdominal pain",
    "fainting or loss of consciousness",
    "sudden vision changes",
    "inability to speak or move",
    "coughing up blood",
    "severe allergic reaction"
]

# Symptom combinations that suggest specific conditions
SYMPTOM_PATTERNS = {
    "covid19": {
        "required": ["fever", "cough"],
        "supporting": ["loss of taste", "loss of smell", "fatigue"],
        "minimum_required": 2,
        "minimum_supporting": 1
    },
    "migraine": {
        "required": ["severe headache"],
        "supporting": ["sensitivity to light", "sensitivity to sound", "nausea"],
        "minimum_required": 1,
        "minimum_supporting": 2
    },
    "pneumonia": {
        "required": ["fever", "cough", "difficulty breathing"],
        "supporting": ["chest pain", "fatigue", "rapid breathing"],
        "minimum_required": 2,
        "minimum_supporting": 1
    }
}

def get_condition_probability(symptoms: List[str], condition: str) -> float:
    """
    Calculate the probability of a condition given a list of symptoms.
    Uses pattern matching and severity consideration.
    """
    if condition in SYMPTOM_PATTERNS:
        pattern = SYMPTOM_PATTERNS[condition]
        required_matches = sum(1 for s in pattern["required"] if any(s in reported for reported in symptoms))
        supporting_matches = sum(1 for s in pattern["supporting"] if any(s in reported for reported in symptoms))
        
        if (required_matches >= pattern["minimum_required"] and 
            supporting_matches >= pattern["minimum_supporting"]):
            base_prob = 0.8
        else:
            base_prob = 0.3
    else:
        condition_symptoms = set(CONDITION_SYMPTOMS[condition])
        matched_symptoms = set(symptoms) & condition_symptoms
        base_prob = len(matched_symptoms) / len(condition_symptoms)
    
    # Adjust probability based on symptom severity
    severity_factor = sum(SYMPTOM_SEVERITY.get(s, 0.5) for s in symptoms) / len(symptoms)
    return base_prob * (1 + severity_factor) / 2

def get_severity_score(symptoms: List[str]) -> float:
    """
    Calculate overall severity score based on present symptoms.
    """
    if not symptoms:
        return 0
    
    # Calculate base severity
    total_severity = sum(SYMPTOM_SEVERITY.get(s, 0.5) for s in symptoms)
    base_severity = total_severity / len(symptoms)
    
    # Check for emergency symptoms
    if any(symptom in EMERGENCY_SYMPTOMS for symptom in symptoms):
        base_severity = max(base_severity, 0.8)
    
    return base_severity

def should_escalate(symptoms: List[str]) -> bool:
    """
    Determine if the symptoms require immediate medical attention.
    """
    # Check for emergency symptoms
    if any(symptom in EMERGENCY_SYMPTOMS for symptom in symptoms):
        return True
    
    # Check severity
    if get_severity_score(symptoms) > 0.7:
        return True
    
    # Check dangerous combinations
    dangerous_combinations = [
        {"high fever", "difficulty breathing"},
        {"severe headache", "confusion"},
        {"chest pain", "shortness of breath"}
    ]
    
    symptom_set = set(symptoms)
    for combination in dangerous_combinations:
        if combination.issubset(symptom_set):
            return True
    
    return False

def get_follow_up_question(symptoms: List[str]) -> str:
    """
    Get relevant follow-up questions based on mentioned symptoms.
    """
    asked_categories = set()
    for symptom in symptoms:
        for category, questions in SYMPTOM_QUESTIONS.items():
            if category not in asked_categories and category in symptom:
                asked_categories.add(category)
                return questions[0]  # Return first unused question
    
    # If no specific questions, ask about duration
    return "How long have you been experiencing these symptoms?"

def get_condition_category(condition: str) -> str:
    """
    Get the category of a condition.
    """
    for category, subcategories in CONDITION_CATEGORIES.items():
        for subcategory, conditions in subcategories.items():
            if condition in conditions:
                return f"{category}/{subcategory}"
    return "unknown"

def get_related_conditions(condition: str) -> List[str]:
    """
    Get related conditions based on shared category and symptoms.
    """
    category = get_condition_category(condition)
    if category == "unknown":
        return []
    
    main_category = category.split('/')[0]
    related = []
    
    # Get conditions in the same category
    for subcategories in CONDITION_CATEGORIES[main_category].values():
        for related_condition in subcategories:
            if related_condition != condition:
                related.append(related_condition)
    
    return related

def get_condition_duration(condition: str) -> Dict:
    """
    Get the typical duration range for a condition.
    """
    return CONDITION_DURATIONS.get(condition, {
        "min": 7, "typical": 14, "max": 21  # Default values
    })

class ExternalDatasetIntegrator:
    """Integrates external medical datasets with our knowledge base."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.external_data = None
        self.differential_diagnoses = {}
        
    def load_ddxplus_dataset(self, path: str) -> None:
        """
        Load the DDXPlus dataset for enhanced differential diagnosis.
        
        Args:
            path: Path to the DDXPlus dataset files
        """
        try:
            # Load patient data
            patients_df = pd.read_csv(f"{path}/release_train_patients.zip")
            conditions = json.load(open(f"{path}/release_conditions.json"))
            evidences = json.load(open(f"{path}/release_evidences.json"))
            
            # Process and integrate with our knowledge base
            for condition in conditions:
                symptoms = set(condition["symptoms"].keys())
                antecedents = set(condition["antecedents"].keys())
                
                # Update our knowledge base with new condition
                if condition["condition_name"] not in CONDITION_SYMPTOMS:
                    CONDITION_SYMPTOMS[condition["condition_name"]] = list(symptoms)
                
                # Add differential diagnoses
                self.differential_diagnoses[condition["condition_name"]] = {
                    "severity": condition["severity"],
                    "icd10": condition["icd10-id"],
                    "related_conditions": []
                }
            
            # Process differential diagnoses from patient data
            for _, patient in patients_df.iterrows():
                if isinstance(patient["DIFFERENTIAL_DIAGNOSIS"], str):
                    diff_diag = eval(patient["DIFFERENTIAL_DIAGNOSIS"])
                    condition = patient["PATHOLOGY"]
                    if condition in self.differential_diagnoses:
                        for related_cond, prob in diff_diag:
                            if related_cond != condition:
                                self.differential_diagnoses[condition]["related_conditions"].append({
                                    "condition": related_cond,
                                    "probability": prob
                                })
            
            print(f"Successfully integrated DDXPlus dataset with {len(conditions)} conditions")
            
        except Exception as e:
            print(f"Error loading DDXPlus dataset: {e}")
    
    def get_enhanced_diagnosis(self, symptoms: List[str], condition: str) -> Dict:
        """
        Get enhanced diagnosis information using both our knowledge base and external data.
        
        Args:
            symptoms: List of reported symptoms
            condition: Suspected condition
            
        Returns:
            Dict containing enhanced diagnosis information
        """
        base_prob = get_condition_probability(symptoms, condition)
        
        enhanced_info = {
            "probability": base_prob,
            "severity": get_severity_score(symptoms),
            "differential_diagnoses": [],
            "confidence_score": 0.0
        }
        
        # Add differential diagnoses if available
        if condition in self.differential_diagnoses:
            diff_info = self.differential_diagnoses[condition]
            enhanced_info["severity"] = max(enhanced_info["severity"], 
                                         1.0 - (diff_info["severity"] / 5.0))
            
            # Add related conditions
            for related in diff_info["related_conditions"]:
                if related["probability"] > 0.1:  # Only include significant alternatives
                    enhanced_info["differential_diagnoses"].append({
                        "condition": related["condition"],
                        "probability": related["probability"],
                        "shared_symptoms": len(
                            set(CONDITION_SYMPTOMS.get(related["condition"], [])) &
                            set(symptoms)
                        )
                    })
            
            # Calculate confidence score based on symptom match and differential spread
            symptom_match = len(set(CONDITION_SYMPTOMS[condition]) & set(symptoms)) / \
                          len(CONDITION_SYMPTOMS[condition])
            diff_spread = sum(d["probability"] for d in enhanced_info["differential_diagnoses"])
            
            enhanced_info["confidence_score"] = (symptom_match * 0.7 + (1 - diff_spread) * 0.3)
        
        return enhanced_info

# Initialize the integrator
dataset_integrator = ExternalDatasetIntegrator()

def initialize_external_datasets(ddxplus_path: Optional[str] = None) -> None:
    """Initialize external datasets if paths are provided."""
    if ddxplus_path:
        dataset_integrator.load_ddxplus_dataset(ddxplus_path) 