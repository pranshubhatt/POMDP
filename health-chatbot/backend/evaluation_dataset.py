"""
Evaluation dataset for the medical chatbot.
Contains structured test cases with symptoms and expected diagnoses.
Used for evaluating the accuracy of the POMDP model.
"""

from typing import Dict, List, Tuple

# Define a standardized evaluation dataset
# Format: {
#   "case_id": "unique identifier",
#   "description": "description of the case",
#   "symptoms": ["symptom1", "symptom2", ...],  # Known symptoms from our database
#   "expected_diagnoses": [
#       {"disease": "disease_name", "probability_range": [min, max]},
#       ...
#   ],
#   "severity_range": [min, max],  # Expected severity score range
#   "patient_messages": ["message1", "message2", ...],  # Natural language messages
# }

EVALUATION_DATASET = [
    {
        "case_id": "common_cold_1",
        "description": "Classic common cold symptoms",
        "symptoms": ["runny_nose", "continuous_sneezing", "cough", "fatigue", "high_fever"],
        "expected_diagnoses": [
            {"disease": "common cold", "probability_range": [0.05, 0.15]},
            {"disease": "pneumonia", "probability_range": [0.03, 0.10]},
        ],
        "severity_range": [3.0, 6.0],
        "patient_messages": [
            "I have a runny nose and can't stop sneezing",
            "Now I've developed a cough and feeling tired",
            "I think I'm getting a fever too"
        ]
    },
    {
        "case_id": "diabetes_1",
        "description": "Type 2 diabetes symptoms",
        "symptoms": ["fatigue", "polyuria", "excessive_hunger", "weight_loss"],
        "expected_diagnoses": [
            {"disease": "diabetes", "probability_range": [0.05, 0.2]},
        ],
        "severity_range": [3.0, 6.0],
        "patient_messages": [
            "I've been urinating frequently and feel very thirsty all the time",
            "I keep feeling hungry but I'm actually losing weight",
            "I feel tired all the time even when I sleep well"
        ]
    },
    {
        "case_id": "tuberculosis_1",
        "description": "Pulmonary tuberculosis",
        "symptoms": ["cough", "high_fever", "fatigue", "weight_loss", "chest_pain"],
        "expected_diagnoses": [
            {"disease": "tuberculosis", "probability_range": [0.05, 0.2]},
            {"disease": "pneumonia", "probability_range": [0.03, 0.15]},
        ],
        "severity_range": [5.0, 8.0],
        "patient_messages": [
            "I've had a persistent cough for more than 3 weeks",
            "I've been having night sweats and fever",
            "I'm losing weight and feel pain in my chest when breathing"
        ]
    },
    {
        "case_id": "heart_attack_1",
        "description": "Acute myocardial infarction (heart attack)",
        "symptoms": ["chest_pain", "breathlessness", "sweating", "vomiting"],
        "expected_diagnoses": [
            {"disease": "heart attack", "probability_range": [0.1, 0.3]},
        ],
        "severity_range": [7.0, 10.0],
        "patient_messages": [
            "I'm having intense chest pain that radiates to my left arm",
            "I'm short of breath and sweating profusely",
            "I feel nauseous and just vomited"
        ]
    },
    {
        "case_id": "dengue_1",
        "description": "Dengue fever",
        "symptoms": ["high_fever", "headache", "joint_pain", "fatigue", "vomiting", "skin_rash"],
        "expected_diagnoses": [
            {"disease": "dengue", "probability_range": [0.04, 0.15]},
            {"disease": "typhoid", "probability_range": [0.03, 0.10]},
        ],
        "severity_range": [6.0, 9.0],
        "patient_messages": [
            "I have a high fever and severe headache",
            "My joints and muscles are very painful",
            "I've developed a rash and vomited several times"
        ]
    },
    {
        "case_id": "migraine_1",
        "description": "Classic migraine with aura",
        "symptoms": ["headache", "visual_disturbances", "nausea", "vomiting"],
        "expected_diagnoses": [
            {"disease": "migraine", "probability_range": [0.07, 0.25]},
        ],
        "severity_range": [4.0, 7.0],
        "patient_messages": [
            "I have a throbbing headache on one side",
            "I'm seeing zigzag lines and flashing lights",
            "I feel nauseous and sensitive to light and sound"
        ]
    },
    {
        "case_id": "malaria_1",
        "description": "Malaria symptoms",
        "symptoms": ["high_fever", "chills", "headache", "fatigue", "vomiting"],
        "expected_diagnoses": [
            {"disease": "malaria", "probability_range": [0.05, 0.15]},
            {"disease": "typhoid", "probability_range": [0.03, 0.10]},
        ],
        "severity_range": [5.0, 8.0],
        "patient_messages": [
            "I have a high fever that comes and goes",
            "I'm experiencing chills and severe headache",
            "I feel extremely tired and have vomited several times"
        ]
    },
    {
        "case_id": "urinary_tract_infection_1",
        "description": "Urinary tract infection",
        "symptoms": ["burning_micturition", "spotting__urination", "fatigue", "high_fever"],
        "expected_diagnoses": [
            {"disease": "urinary tract infection", "probability_range": [0.07, 0.2]},
        ],
        "severity_range": [3.0, 6.0],
        "patient_messages": [
            "It burns when I urinate and I feel the need to go frequently",
            "I've noticed some blood in my urine",
            "I'm feeling tired and feverish"
        ]
    },
    {
        "case_id": "hepatitis_b_1",
        "description": "Hepatitis B infection",
        "symptoms": ["fatigue", "yellowish_skin", "dark_urine", "abdominal_pain", "loss_of_appetite"],
        "expected_diagnoses": [
            {"disease": "hepatitis b", "probability_range": [0.05, 0.2]},
            {"disease": "jaundice", "probability_range": [0.04, 0.15]},
        ],
        "severity_range": [5.0, 8.0],
        "patient_messages": [
            "I've been feeling extremely tired for weeks",
            "My skin and eyes look yellow and my urine is dark",
            "I have pain in my abdomen and no appetite"
        ]
    },
    {
        "case_id": "gastroenteritis_1",
        "description": "Viral gastroenteritis",
        "symptoms": ["vomiting", "diarrhoea", "abdominal_pain", "high_fever", "dehydration"],
        "expected_diagnoses": [
            {"disease": "gastroenteritis", "probability_range": [0.05, 0.2]},
        ],
        "severity_range": [4.0, 7.0],
        "patient_messages": [
            "I've been vomiting and have diarrhea",
            "My stomach hurts and I feel feverish",
            "I think I'm getting dehydrated"
        ]
    }
]

def get_evaluation_cases() -> List[Dict]:
    """Return the evaluation dataset."""
    return EVALUATION_DATASET

def get_case_by_id(case_id: str) -> Dict:
    """Return a specific case by ID."""
    for case in EVALUATION_DATASET:
        if case["case_id"] == case_id:
            return case
    return None 