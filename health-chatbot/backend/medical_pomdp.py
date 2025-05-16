"""
Enhanced POMDP implementation for medical diagnosis using integrated knowledge base.

Mathematical Formulation:
--------------------------
The medical diagnosis problem is formulated as a Partially Observable Markov Decision Process (POMDP):

1. States (S): The set of possible diseases {d₁, d₂, ..., dₙ}
   - Hidden states representing the actual condition of the patient

2. Actions (A): Asking follow-up questions or making diagnosis recommendations
   - Questions about specific symptoms
   - Requesting clarification
   - Making a diagnosis

3. Observations (O): Symptoms reported by the patient {s₁, s₂, ..., sₘ}
   - Direct observations that depend on the underlying state

4. Transition Model T(s'|s,a): P(disease_t+1 | disease_t, action)
   - In our medical model, diseases don't transition between states
   - Represented as an identity matrix (diseases are stable over the course of interaction)

5. Observation Model O(o|s,a): P(symptom | disease)
   - Probability of observing symptom 'o' given disease 's'
   - Constructed from medical knowledge base
   - Incorporates symptom severity as a weighting factor

6. Reward Function R(s,a): Not explicitly modeled in this implementation
   - Implicitly reflects the value of correct diagnosis

7. Discount Factor γ: Not explicitly used

8. Belief State: Probability distribution over diseases
   - b(d) = P(disease=d | observed symptoms)
   - Updated using Bayes' rule after each observation

Belief Update Equation:
b'(s') = η * O(o|s',a) * ∑_s T(s'|s,a) * b(s)

where η is a normalization factor to ensure ∑_s' b'(s') = 1

In our implementation:
- b(d) is the belief state (disease_probabilities)
- O(o|s',a) is derived from the symptom-disease relationships and severity
- T(s'|s,a) is the identity matrix (diseases don't transition)
- The belief update incorporates both symptom match ratio and severity
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import os
import re

class MedicalPOMDP:
    def __init__(self, knowledge_base_path: str = "../medical_knowledge_enhanced.json"):
        """Initialize the POMDP with the enhanced knowledge base."""
        # Resolve knowledge base path
        if not os.path.isabs(knowledge_base_path):
            knowledge_base_path = os.path.join(os.path.dirname(__file__), knowledge_base_path)
            knowledge_base_path = os.path.abspath(knowledge_base_path)
        
        self.load_knowledge_base(knowledge_base_path)
        self.reset_state()
    
    def load_knowledge_base(self, path: str):
        """Load the enhanced medical knowledge base."""
        try:
            print(f"Loading knowledge base from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            
            # Debug: Print knowledge base structure
            print("\nKnowledge Base Structure:")
            print(f"Keys in knowledge base: {list(self.knowledge_base.keys())}")
            print(f"Number of symptoms in list: {len(self.knowledge_base['symptoms'])}")
            print(f"Number of diseases: {len(self.knowledge_base['diseases'])}")
            print(f"Number of severity entries: {len(self.knowledge_base['symptom_severity'])}")
            
            # Sample a disease to check structure
            sample_disease = next(iter(self.knowledge_base['diseases']))
            print(f"\nSample disease '{sample_disease}' structure:")
            print(json.dumps(self.knowledge_base['diseases'][sample_disease], indent=2))
            
            # Clean symptom keys in severity data
            cleaned_severity = {}
            for k, v in self.knowledge_base['symptom_severity'].items():
                cleaned_key = k.strip().replace(" ", "_").lower()
                cleaned_severity[cleaned_key] = float(v)  # Ensure values are float
            self.knowledge_base['symptom_severity'] = cleaned_severity
            
            # Clean symptoms list and maintain order
            self.all_symptoms = [s.strip().replace(" ", "_").lower() 
                               for s in self.knowledge_base['symptoms']]
            self.all_diseases = list(self.knowledge_base['diseases'].keys())
            
            # Clean disease symptoms
            for disease in self.all_diseases:
                self.knowledge_base['diseases'][disease]['symptoms'] = [
                    s.strip().replace(" ", "_").lower() 
                    for s in self.knowledge_base['diseases'][disease]['symptoms']
                ]
            
            # Build synonym database for symptoms
            self._build_symptom_synonyms()
            
            print(f"\nAfter cleaning:")
            print(f"Loaded {len(self.all_symptoms)} symptoms and {len(self.all_diseases)} diseases")
            print(f"Sample symptoms: {self.all_symptoms[:5]}")
            print(f"Sample diseases: {self.all_diseases[:5]}")
            print(f"Sample severity values: {dict(list(self.knowledge_base['symptom_severity'].items())[:5])}")
            
            # Initialize transition and observation matrices
            self.initialize_matrices()
        except FileNotFoundError:
            raise Exception(f"Knowledge base not found at {path}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in knowledge base: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading knowledge base: {str(e)}")
    
    def _build_symptom_synonyms(self):
        """Build a comprehensive symptom synonym dictionary for better matching."""
        self.symptom_synonyms = {
            # Basic variations
            'tired': 'fatigue',
            'exhausted': 'fatigue',
            'weakness': 'fatigue',
            'no energy': 'fatigue',
            'worn out': 'fatigue',
            'lethargic': 'fatigue',
            
            # Fever related
            'fever': 'high_fever',
            'elevated temperature': 'high_fever',
            'febrile': 'high_fever',
            'mild fever': 'mild_fever',
            'slight fever': 'mild_fever',
            'low grade fever': 'mild_fever',
            
            # Breathing related
            'breathing difficulty': 'breathlessness',
            'trouble breathing': 'breathlessness',
            'cant breathe': 'breathlessness',
            'hard to breathe': 'breathlessness',
            'shortness of breath': 'breathlessness',
            'dyspnea': 'breathlessness',
            
            # Digestive symptoms
            'throwing up': 'vomiting',
            'vomit': 'vomiting',
            'nausea': 'nausea',
            'feel sick': 'nausea',
            'queasy': 'nausea',
            'diarrhea': 'diarrhoea',
            'loose stool': 'diarrhoea',
            'constipation': 'constipation',
            'can\'t poop': 'constipation',
            'hard stool': 'constipation',
            
            # Head and sensory
            'coughing': 'cough',
            'headaches': 'headache',
            'migraine': 'headache',
            'running nose': 'runny_nose',
            'rhinorrhea': 'runny_nose',
            'sneezing': 'continuous_sneezing',
            'vision changes': 'blurred_and_distorted_vision',
            'blurry vision': 'blurred_and_distorted_vision',
            'blurry sight': 'blurred_and_distorted_vision',
            'can\'t see clearly': 'blurred_and_distorted_vision',
            'seeing things': 'visual_disturbances',
            'spots in vision': 'visual_disturbances',
            'flashing lights': 'visual_disturbances',
            
            # Pain related
            'throat pain': 'throat_irritation',
            'sore throat': 'throat_irritation',
            'stomach pain': 'stomach_pain',
            'stomach ache': 'stomach_pain',
            'abdominal pain': 'abdominal_pain',
            'belly pain': 'abdominal_pain',
            'tummy ache': 'abdominal_pain',
            'chest pain': 'chest_pain',
            'chest discomfort': 'chest_pain',
            'chest pressure': 'chest_pain',
            'joint pain': 'joint_pain',
            'arthralgia': 'joint_pain',
            'painful joints': 'joint_pain',
            'muscle pain': 'muscle_pain',
            'myalgia': 'muscle_pain',
            'sore muscles': 'muscle_pain',
            'back pain': 'back_pain',
            'backache': 'back_pain',
            'painful urination': 'burning_micturition',
            'pain when peeing': 'burning_micturition',
            'burning urination': 'burning_micturition',
            
            # Mental/neurological
            'dizzy': 'dizziness',
            'feeling dizzy': 'dizziness',
            'vertigo': 'dizziness',
            'lightheaded': 'dizziness',
            'unsteady': 'dizziness',
            'confused': 'altered_sensorium',
            'disoriented': 'altered_sensorium',
            'not thinking clearly': 'altered_sensorium',
            'depression': 'depression',
            'feeling down': 'depression',
            'no interest': 'depression',
            'feeling sad': 'depression',
            'anxious': 'anxiety',
            'worried': 'anxiety',
            'panic': 'anxiety',
            'fearful': 'anxiety',
            
            # Skin related
            'itching': 'itching',
            'itchy': 'itching',
            'pruritus': 'itching',
            'scratching': 'itching',
            'skin rash': 'skin_rash',
            'rash': 'skin_rash',
            'eruption': 'skin_rash',
            'hives': 'nodal_skin_eruptions',
            'skin bumps': 'nodal_skin_eruptions',
            'yellow skin': 'yellowish_skin',
            'jaundice': 'yellowish_skin',
            'skin turning yellow': 'yellowish_skin',
            
            # Miscellaneous
            'cold': 'chills',
            'feeling cold': 'chills',
            'shiver': 'shivering',
            'shivering': 'shivering',
            'blood in urine': 'spotting__urination',
            'blood when peeing': 'spotting__urination',
            'hematuria': 'spotting__urination',
            'weight loss': 'weight_loss',
            'losing weight': 'weight_loss',
            'gaining weight': 'weight_gain',
            'weight gain': 'weight_gain',
            'swollen': 'swelling',
            'edema': 'swelling',
            'puffiness': 'swelling',
            'can\'t sleep': 'insomnia',
            'trouble sleeping': 'insomnia',
            'wake up at night': 'insomnia',
            'no appetite': 'loss_of_appetite',
            'don\'t feel like eating': 'loss_of_appetite',
            'always hungry': 'excessive_hunger',
            'hungry all the time': 'excessive_hunger',
            'blood in stool': 'bloody_stool',
            'rectal bleeding': 'bloody_stool',
            'bloody poop': 'bloody_stool',
            'urinating frequently': 'polyuria',
            'peeing a lot': 'polyuria',
            'frequent urination': 'polyuria',
            'dark urine': 'dark_urine',
            'cola colored urine': 'dark_urine',
            'frequent colds': 'cough',
            'belly swelling': 'abdominal_pain',
            'lumpy throat': 'patches_in_throat',
            'throat patches': 'patches_in_throat',
            'white spots in throat': 'patches_in_throat'
        }
        
        # Build reverse index from symptoms to potential text matches
        self.symptom_regexes = {}
        for symptom in self.all_symptoms:
            # Create variants of the symptom name for matching
            base_term = symptom.replace('_', ' ')
            self.symptom_regexes[symptom] = [
                re.compile(r'\b' + re.escape(base_term) + r'\b', re.IGNORECASE),
                re.compile(r'\b' + re.escape(symptom) + r'\b', re.IGNORECASE)
            ]
        
        # Add synonym regexes
        for synonym, target in self.symptom_synonyms.items():
            if target in self.symptom_regexes:
                self.symptom_regexes[target].append(
                    re.compile(r'\b' + re.escape(synonym) + r'\b', re.IGNORECASE)
                )
    
    def initialize_matrices(self):
        """Initialize POMDP matrices based on knowledge base.
        
        Creates:
        1. Transition matrix: P(disease_t+1 | disease_t, action)
        2. Observation matrix: P(symptom | disease)
        """
        print("Initializing POMDP matrices...")
        
        n_diseases = len(self.all_diseases)
        n_symptoms = len(self.all_symptoms)
        
        print(f"Matrix dimensions: Diseases={n_diseases}, Symptoms={n_symptoms}")
        
        # Initialize transition matrix: P(disease_t+1 | disease_t, action)
        # In our model, diseases don't transition (identity matrix)
        self.transition_matrix = np.eye(n_diseases)
        
        # Initialize observation matrix: P(symptom | disease)
        self.observation_matrix = np.zeros((n_diseases, n_symptoms))
        
        # Create symptom index mapping for faster lookup
        self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(self.all_symptoms)}
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.all_diseases)}
        
        # Fill observation matrix based on knowledge base
        for disease, idx in self.disease_to_idx.items():
            disease_symptoms = set(self.knowledge_base['diseases'][disease]['symptoms'])
            for symptom in disease_symptoms:
                if symptom in self.symptom_to_idx:
                    j = self.symptom_to_idx[symptom]
                    # Use symptom severity as the observation probability
                    severity = self.knowledge_base['symptom_severity'].get(symptom, 0.5)
                    self.observation_matrix[idx, j] = severity
        
        # Normalize observation probabilities
        row_sums = self.observation_matrix.sum(axis=1, keepdims=True)
        self.observation_matrix = np.divide(self.observation_matrix, row_sums, 
                                         where=row_sums != 0)
        
        print("POMDP matrices initialized successfully")
    
    def reset_state(self):
        """Reset the POMDP state for a new session."""
        self.belief_state = {
            'disease_probabilities': {},  # Current belief over diseases
            'observed_symptoms': set(),   # Symptoms mentioned by user
            'severity_score': 0.0,        # Overall severity assessment
            'conversation_history': [],   # Track conversation flow
            'suggested_questions': []     # Follow-up questions to ask
        }
        
        # Initialize uniform distribution over diseases
        total_diseases = len(self.all_diseases)
        for disease in self.all_diseases:
            self.belief_state['disease_probabilities'][disease] = 1.0 / total_diseases
    
    async def extract_symptoms(self, user_message: str) -> List[str]:
        """Extract symptoms from user message using enhanced regex and context matching.
        
        This method implements a multi-stage approach to symptom extraction:
        1. Direct matching using regex patterns
        2. Synonym matching for common expressions
        3. Contextual matching for symptoms mentioned in context
        """
        print(f"Extracting symptoms from message: {user_message}")
        
        if not user_message.strip():
            print("Empty message, no symptoms to extract")
            return []
        
        # Convert message to lowercase for case-insensitive matching
        message_lower = user_message.lower()
        
        # Phase 1: Direct symptom matching using regex
        extracted_symptoms = set()
        for symptom, regexes in self.symptom_regexes.items():
            for regex in regexes:
                if regex.search(message_lower):
                    extracted_symptoms.add(symptom)
                    break
        
        # Phase 2: Look for common symptom phrases (regex may have missed)
        for symptom_phrase, canonical_symptom in self.symptom_synonyms.items():
            if symptom_phrase.lower() in message_lower and canonical_symptom not in extracted_symptoms:
                if canonical_symptom in self.all_symptoms:
                    extracted_symptoms.add(canonical_symptom)
        
        # Phase 3: Handle negation (e.g., "I don't have a fever")
        negation_patterns = [
            r"don't have (a |an |)([a-zA-Z\s]+)",
            r"no ([a-zA-Z\s]+)", 
            r"not experiencing ([a-zA-Z\s]+)",
            r"haven't had (a |an |)([a-zA-Z\s]+)",
            r"without ([a-zA-Z\s]+)"
        ]
        
        for pattern in negation_patterns:
            for match in re.finditer(pattern, message_lower):
                negated_term = match.group(2) if pattern.count('(') > 1 else match.group(1)
                # Check if the negated term matches any of our symptoms
                negated_symptoms = set()
                for symptom in extracted_symptoms:
                    symptom_term = symptom.replace('_', ' ')
                    if symptom_term in negated_term or symptom in negated_term:
                        negated_symptoms.add(symptom)
                
                # Remove negated symptoms
                extracted_symptoms = extracted_symptoms - negated_symptoms
        
        # Convert to list for return
        extracted_list = list(extracted_symptoms)
        
        if extracted_list:
            print(f"Extracted symptoms: {extracted_list}")
        else:
            print("No symptoms extracted from message")
            
        return extracted_list
    
    def update_belief_state(self, new_symptoms: List[str]):
        """Update belief state based on observed symptoms using Bayesian updating.
        
        This implements the core of the POMDP belief update equation:
        b'(s') = η * O(o|s',a) * ∑_s T(s'|s,a) * b(s)
        
        Where:
        - b(s) is the current belief state (disease_probabilities)
        - O(o|s',a) is derived from symptom-disease relationships
        - T(s'|s,a) is the identity matrix (diseases don't transition)
        - η is a normalization factor
        """
        print(f"\nUpdating belief state with new symptoms: {new_symptoms}")
        
        # Clean and normalize new symptoms
        new_symptoms = [s.strip().replace(" ", "_").lower() for s in new_symptoms]
        
        # Add new symptoms to observed set
        self.belief_state['observed_symptoms'].update(new_symptoms)
        print(f"Total observed symptoms: {self.belief_state['observed_symptoms']}")
        
        # Calculate new disease probabilities
        new_probabilities = {}
        total_score = 0
        
        # Get severity of observed symptoms
        observed_severities = {
            symptom: self.knowledge_base['symptom_severity'].get(symptom, 0.5)
            for symptom in self.belief_state['observed_symptoms']
        }
        print(f"Symptom severities: {observed_severities}")
        
        # Track top matches for debugging
        disease_scores = []
        
        for disease in self.all_diseases:
            disease_symptoms = set(self.knowledge_base['diseases'][disease]['symptoms'])
            matched_symptoms = disease_symptoms.intersection(self.belief_state['observed_symptoms'])
            unmatched_symptoms = self.belief_state['observed_symptoms'] - disease_symptoms
            
            # Calculate base score from symptom matches
            if matched_symptoms:
                # Calculate match ratio (percentage of disease symptoms matched)
                match_ratio = len(matched_symptoms) / len(disease_symptoms)
                
                # Calculate coverage ratio (percentage of observed symptoms explained)
                coverage_ratio = len(matched_symptoms) / len(self.belief_state['observed_symptoms'])
                
                # Calculate average severity of matched symptoms
                severity = sum(observed_severities[s] for s in matched_symptoms) / len(matched_symptoms)
                
                # Weight for recency of symptoms (new symptoms are more important)
                new_symptom_match = len(set(new_symptoms).intersection(matched_symptoms))
                recency_weight = 1.0
                if new_symptom_match > 0:
                    recency_weight = 1.2  # Boost for diseases matching recent symptoms
                
                # Combine scores with weights (implements Bayesian belief update)
                score = (
                    match_ratio * 0.4 +      # How many of the disease's symptoms are present
                    coverage_ratio * 0.4 +    # How many of the observed symptoms match this disease
                    severity * 0.2            # How severe are the matching symptoms
                ) * recency_weight
                
                # Apply penalty for unmatched symptoms (symptoms that don't fit the disease)
                penalty = len(unmatched_symptoms) / (len(self.belief_state['observed_symptoms']) + 1)
                score *= (1 - penalty * 0.2)  # Small penalty for unmatched symptoms
                
                # Store the score
                new_probabilities[disease] = max(score, 0.001)  # Ensure non-zero probability
                total_score += new_probabilities[disease]
                
                # Store details for debugging
                disease_scores.append({
                    'disease': disease,
                    'matched_symptoms': list(matched_symptoms),
                    'unmatched_symptoms': list(unmatched_symptoms),
                    'match_ratio': match_ratio,
                    'coverage_ratio': coverage_ratio,
                    'severity': severity,
                    'recency_weight': recency_weight,
                    'final_score': score
                })
            else:
                # Assign very small probability if no symptoms match
                new_probabilities[disease] = 0.001
                total_score += 0.001
        
        # Sort and print top matches for debugging
        disease_scores.sort(key=lambda x: x['final_score'], reverse=True)
        print("\nTop 3 disease matches before normalization:")
        for score in disease_scores[:3]:
            print(f"\nDisease: {score['disease']}")
            print(f"Matched symptoms: {score['matched_symptoms']}")
            print(f"Match ratio: {score['match_ratio']:.3f}")
            print(f"Coverage ratio: {score['coverage_ratio']:.3f}")
            print(f"Severity: {score['severity']:.3f}")
            print(f"Recency weight: {score['recency_weight']:.1f}")
            print(f"Final score: {score['final_score']:.3f}")
        
        # Normalize probabilities (the η factor in Bayesian update)
        if total_score > 0:
            for disease in new_probabilities:
                new_probabilities[disease] /= total_score
        
        self.belief_state['disease_probabilities'] = new_probabilities
        
        # ENHANCED SEVERITY CALCULATION
        # Define critical symptoms that should increase severity
        critical_symptoms = {
            "chest_pain": 9.0,
            "breathlessness": 8.5,
            "high_fever": 7.5,
            "loss_of_consciousness": 10.0,
            "vomiting": 6.5,
            "headache": 5.0,  # Could be severe depending on context
            "visual_disturbances": 7.0,  # Important for stroke/neurological issues
            "paralysis": 9.5,
            "altered_sensorium": 9.0,
            "yellow_urine": 7.0,  # Important for kidney/liver issues
            "yellowish_skin": 7.5,
            "internal_itching": 6.0,
            "swelling_of_stomach": 7.5,
            "distention_of_abdomen": 7.5,
            "polyuria": 6.5,  # Increased for diabetes recognition
            "blurred_and_distorted_vision": 7.0,  # Increased for diabetes/neurological issues
            "irregular_sugar_level": 7.0,  # Important diabetes symptom
            "excessive_hunger": 6.0,  # Important diabetes symptom
            "weight_loss": 6.0,  # Can indicate serious metabolic issues
            "patches_in_throat": 6.5,  # Can indicate serious infection
            "high_fever": 7.5,
            "extra_marital_contacts": 7.0,  # HIV risk indicator
            "fatigue": 4.5,  # Increased as it's present in many serious conditions
        }
        
        # Define critical symptom combinations that indicate high severity
        critical_combinations = [
            {"chest_pain", "breathlessness"},  # Potential heart attack
            {"high_fever", "stiff_neck"},  # Potential meningitis
            {"high_fever", "skin_rash", "joint_pain"},  # Potential severe infection/dengue
            {"headache", "visual_disturbances"},  # Potential neurological issue
            {"yellowish_skin", "dark_urine"},  # Potential liver failure
            {"headache", "dizziness", "vomiting"},  # Potential serious neurological issue
            {"polyuria", "excessive_hunger", "weight_loss"},  # Classic diabetes triad
            {"polyuria", "fatigue", "blurred_and_distorted_vision"},  # Diabetes with complications
            {"visual_disturbances", "headache", "nausea"},  # Classic migraine with aura
            {"headache", "vomiting", "sensitivity_to_light_and_sound"}  # Severe migraine/potential meningitis
        ]
        
        # Define high-severity diseases
        high_severity_diseases = {
            "heart attack": 9.0,
            "stroke": 9.5,
            "tuberculosis": 8.0,
            "pneumonia": 7.5,
            "hepatitis": 7.5,
            "hepatitis b": 7.5, 
            "hepatitis c": 7.5,
            "hepatitis d": 7.5,
            "hepatitis e": 7.5,
            "alcoholic hepatitis": 7.5,
            "jaundice": 7.0,
            "malaria": 7.5,
            "dengue": 8.0,
            "typhoid": 7.5,
            "hypoglycemia": 8.0,
            "hyperthyroidism": 7.0,
            "hypoglycemia": 8.0,
            "chronic cholestasis": 7.0,
            "paralysis (brain hemorrhage)": 9.5,
            "migraine": 6.5,  # Increased severity for migraine
            "diabetes": 7.0,  # Increased severity for diabetes
            "cervical spondylosis": 6.0,
            "dimorphic hemmorhoids(piles)": 5.5,
            "varicose veins": 5.5,
            "hypothyroidism": 6.5,
            "urinary tract infection": 6.0
        }
        
        # Mapping of symptoms to related conditions for special handling
        symptom_to_condition_mapping = {
            "polyuria": ["diabetes"],
            "excessive_hunger": ["diabetes"],
            "weight_loss": ["diabetes", "hyperthyroidism", "tuberculosis"],
            "fatigue": ["diabetes", "anemia", "hypothyroidism"],
            "visual_disturbances": ["migraine", "stroke", "hypertension"],
            "headache": ["migraine", "hypertension", "sinusitis"],
            "chest_pain": ["heart attack", "angina"],
            "breathlessness": ["heart attack", "pneumonia", "bronchial asthma"],
            "vomiting": ["migraine", "jaundice", "typhoid"],
            "high_fever": ["malaria", "typhoid", "dengue"]
        }

        if self.belief_state['observed_symptoms']:
            # Special condition detection for certain symptom combinations
            detected_conditions = set()
            
            # Check for diabetes symptoms
            diabetes_symptoms = {"polyuria", "excessive_hunger", "weight_loss", "fatigue", "blurred_and_distorted_vision", "irregular_sugar_level"}
            if len(diabetes_symptoms.intersection(self.belief_state['observed_symptoms'])) >= 2:
                detected_conditions.add("diabetes")
            
            # Check for migraine symptoms
            migraine_symptoms = {"headache", "visual_disturbances", "nausea", "vomiting", "sensitivity_to_light_and_sound"}
            if len(migraine_symptoms.intersection(self.belief_state['observed_symptoms'])) >= 2:
                detected_conditions.add("migraine")
                
            # Check for heart attack symptoms
            heart_attack_symptoms = {"chest_pain", "breathlessness", "sweating", "vomiting"}
            if len(heart_attack_symptoms.intersection(self.belief_state['observed_symptoms'])) >= 2:
                detected_conditions.add("heart attack")
                
            # Calculate base severity from observed symptoms
            severity_scores = [
                self.knowledge_base['symptom_severity'].get(s, 0.5) 
                for s in self.belief_state['observed_symptoms']
            ]
            base_severity = sum(severity_scores) / len(severity_scores)
            
            # Apply condition-based severity boost
            for condition in detected_conditions:
                if condition in high_severity_diseases:
                    # More significant boost for detected conditions
                    condition_severity = high_severity_diseases[condition] / 10
                    base_severity = max(base_severity, condition_severity * 8.0 + base_severity * 0.2)
            
            # Check for critical symptoms and apply their higher severity value
            for symptom in self.belief_state['observed_symptoms']:
                if symptom in critical_symptoms:
                    # Add extra weight to critical symptoms, don't just average
                    base_severity = max(base_severity, critical_symptoms[symptom] / 10 * 7 + base_severity * 0.3)
            
            # Check for critical combinations
            for combo in critical_combinations:
                if combo.issubset(self.belief_state['observed_symptoms']):
                    base_severity = max(base_severity, 8.0)  # Significant increase for dangerous combinations
            
            # Consider top disease severity
            top_diseases = sorted(self.belief_state['disease_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            disease_severity_factor = 0
            for disease, prob in top_diseases[:3]:  # Consider top 3 diseases
                if disease in high_severity_diseases and prob > 0.05:  # Only if reasonable probability
                    # Weight by probability and severity of disease
                    disease_severity_factor = max(
                        disease_severity_factor,
                        prob * (high_severity_diseases[disease] / 10 * 0.7 + base_severity * 0.3)
                    )
            
            # Combine base and disease-based severity 
            if disease_severity_factor > 0:
                base_severity = base_severity * 0.6 + disease_severity_factor * 0.4
            
            # Scale the final severity to ensure appropriate ranges (1-10)
            # Symptoms alone shouldn't usually go below 3 or above 9
            final_severity = max(3.0, min(9.5, base_severity))
            
            # For recognized severe conditions with good symptom matches, ensure minimum severity
            for disease, prob in top_diseases[:2]:  # Only top 2 diseases
                if disease in high_severity_diseases and prob > 0.1:  # Strong match
                    min_severity = high_severity_diseases[disease] / 10 * 6.0  # 60% of the disease severity
                    final_severity = max(final_severity, min_severity)
            
            self.belief_state['severity_score'] = final_severity
        
        print(f"\nFinal belief state: {self.get_state_summary()}")
    
    def generate_response(self) -> Dict:
        """Generate response based on current belief state."""
        # Sort diseases by probability
        sorted_diseases = sorted(
            self.belief_state['disease_probabilities'].items(),
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top 3 diseases with probabilities
        top_diseases = {d: round(p, 4) for d, p in sorted_diseases[:3]}
        
        response = {
            'message': '',
            'diseases': top_diseases,
            'severity': round(self.belief_state['severity_score'], 2),
            'suggested_questions': [],
            'precautions': []
        }
        
        # If no symptoms were observed, provide a generic response
        if not self.belief_state['observed_symptoms']:
            response['message'] = "Please describe your symptoms so I can help assess your condition."
            return response
        
        # ADDED: List of common diseases with corresponding symptoms
        common_diseases = {
            "common cold": ["high_fever", "mild_fever", "fatigue", "cough", "runny_nose", "sneezing", "continuous_sneezing", "throat_irritation", "headache"],
            "viral fever": ["high_fever", "mild_fever", "fatigue", "headache", "body_pain", "watering_from_eyes"],
            "influenza": ["high_fever", "mild_fever", "fatigue", "cough", "headache", "body_pain", "muscle_pain", "throat_irritation"],
            "sinusitis": ["high_fever", "mild_fever", "headache", "watering_from_eyes", "runny_nose", "cough"]
        }
        
        # ADDED: Check if only one or few common symptoms are present (like just fever)
        if len(self.belief_state['observed_symptoms']) <= 2:
            # Check for very common symptoms like fever
            if "high_fever" in self.belief_state['observed_symptoms'] or "mild_fever" in self.belief_state['observed_symptoms']:
                # For just fever, suggest common viral conditions instead of serious diseases
                response['message'] = f"Based on your symptoms ({', '.join(self.belief_state['observed_symptoms'])}), "
                response['message'] += "you may have a viral fever or common cold. "
                
                if self.belief_state['severity_score'] > 7.0:
                    response['message'] += "\nIf your fever is very high or persists for more than 3 days, please consult a healthcare provider."
                elif self.belief_state['severity_score'] > 5.0:
                    response['message'] += "\nPlease rest, stay hydrated, and monitor your symptoms. Consult a healthcare provider if symptoms worsen."
                
                # Provide reasonable disease suggestions for fever
                response['diseases'] = {
                    "viral fever": 0.45,
                    "common cold": 0.25,
                    "influenza": 0.20
                }
                
                # Add follow-up questions to determine more specific conditions
                response['suggested_questions'] = [
                    "Do you have any other symptoms like cough or sore throat?",
                    "How long have you had the fever?",
                    "Have you been in contact with sick people recently?"
                ]
                
                # Add reasonable precautions
                response['precautions'] = [
                    "Rest and stay hydrated",
                    "Take over-the-counter fever reducers as directed",
                    "Monitor your temperature regularly",
                    "Seek medical attention if fever persists more than 3 days"
                ]
                
                return response
        
        # Get most likely disease
        if sorted_diseases:
            top_disease, top_probability = sorted_diseases[0]
            
            # ADDED: Check if any common diseases match the symptoms better
            for disease, symptoms in common_diseases.items():
                symptom_match = len(set(symptoms).intersection(self.belief_state['observed_symptoms']))
                if symptom_match >= 2:  # If 2+ symptoms match a common disease
                    # Prioritize common diseases for common symptoms
                    matched_ratio = symptom_match / len(symptoms)
                    if matched_ratio > 0.3:  # At least 30% match with common disease
                        # Check if this common disease is more applicable than the top disease
                        if top_disease not in common_diseases.keys() and top_probability < 0.25:
                            top_disease = disease
                            break
            
            # Get observed symptoms for the message
            symptoms_text = ", ".join(self.belief_state['observed_symptoms'])
            response['message'] = f"Based on your symptoms ({symptoms_text}), "
            
            # MODIFIED: Add disease information with modified confidence level and thresholds
            conf_level = ""
            # Only suggest serious conditions when confidence is high and symptoms match well
            serious_conditions = ["aids", "tuberculosis", "hepatitis", "heart attack", "stroke"]
            
            if top_disease in serious_conditions and len(self.belief_state['observed_symptoms']) < 3:
                # Avoid suggesting serious conditions with insufficient symptoms
                conf_level = "might"
                # Try to substitute with a more common disease if available
                alt_diseases = [d for d, p in sorted_diseases[1:4] if d not in serious_conditions]
                if alt_diseases:
                    top_disease = alt_diseases[0]
            elif top_probability > 0.2:
                conf_level = "most likely"
            elif top_probability > 0.1:
                conf_level = "possibly"
            else:
                conf_level = "might"
                
            response['message'] += f"you {conf_level} have {top_disease}."
            
            # Add severity assessment based on improved scaling
            if self.belief_state['severity_score'] > 7.0:
                response['message'] += "\nYour symptoms indicate a potentially serious condition. Please seek immediate medical attention."
            elif self.belief_state['severity_score'] > 5.0:
                response['message'] += "\nYour symptoms warrant medical attention. Please consult a healthcare provider."
            elif self.belief_state['severity_score'] > 3.0:
                response['message'] += "\nPlease monitor your symptoms and consult a healthcare provider if they worsen."
            
            # Add precautions from top disease or from observed symptoms
            for symptom in self.belief_state['observed_symptoms']:
                if symptom in self.knowledge_base['symptom_precautions']:
                    response['precautions'].extend(
                        prec for prec in self.knowledge_base['symptom_precautions'][symptom] 
                        if prec not in response['precautions']
                    )
            
            # Generate follow-up questions based on top disease symptoms
            disease_symptoms = self.knowledge_base['diseases'][top_disease]['symptoms']
            missing_symptoms = set(disease_symptoms) - self.belief_state['observed_symptoms']
            if missing_symptoms:
                response['suggested_questions'] = [
                    f"Are you experiencing {symptom.replace('_', ' ')}?" 
                    for symptom in list(missing_symptoms)[:3]
                ]
        
        return response

    def get_state_summary(self) -> Dict:
        """Get a summary of the current POMDP state."""
        return {
            'observed_symptoms': list(self.belief_state['observed_symptoms']),
            'top_diseases': dict(sorted(
                self.belief_state['disease_probabilities'].items(),
                key=lambda x: x[1], reverse=True
            )[:5]),
            'severity_score': self.belief_state['severity_score'],
            'conversation_length': len(self.belief_state['conversation_history'])
        }