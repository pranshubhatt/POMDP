# PulseAI: Advanced Medical Diagnostic System with POMDP

PulseAI is an intelligent medical diagnostic assistant that combines Partially Observable Markov Decision Process (POMDP) with natural language understanding for accurate symptom assessment and diagnosis suggestions.

## Features

- **Advanced POMDP Decision Model**: Handles uncertainty in patient inputs for accurate medical diagnosis
- **Bayesian Belief Network**: Dynamically updates diagnosis probabilities as new symptoms are reported
- **Robust Symptom Extraction**: Uses sophisticated NLP techniques to identify symptoms from natural language
- **Comprehensive Medical Knowledge Base**: Integrated database of diseases, symptoms, and severity factors
- **Severity Assessment System**: Monitors symptom severity and recommends appropriate medical attention
- **Futuristic User Interface**: Professional, responsive design with real-time updates and animations
- **Multi-Session Support**: Maintains separate chat sessions for multiple users

## Architecture

### Backend Components

1. **Medical Knowledge Base**
   - Structured symptom-condition relationships
   - Severity scoring system
   - Comprehensive symptom synonym matching

2. **POMDP Engine**
   - Mathematical Bayesian belief state tracking
   - Symptom combination detection
   - Condition-severity correlations
   - Natural language pattern matching

3. **FastAPI Server**
   - RESTful API endpoints
   - Session management
   - Error handling
   - CORS support

### Frontend Components

React application with TypeScript and Tailwind CSS featuring:
- Modern glassmorphism UI design
- Animated components and transitions
- Responsive layout for all devices
- Real-time diagnostic visualizations

## Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd health-chatbot
   ```

2. **Install Backend Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Run the Backend**
   ```bash
   uvicorn main:app --reload
   ```

4. **Install Frontend Dependencies**
   ```bash
   cd ../frontend
   npm install
   ```

5. **Run the Frontend**
   ```bash
   npm run dev
   ```

## API Endpoints

### POST /chat
Process a chat message and get diagnosis information.

Request:
```json
{
    "session_id": "string",
    "message": "string"
}
```

Response:
```json
{
    "message": "string",
    "diseases": {
        "disease_name": float
    },
    "severity": float,
    "suggested_questions": ["string"],
    "precautions": ["string"]
}
```

### DELETE /session/{session_id}
End a chat session and cleanup resources.

## Research Applications

PulseAI is designed for medical research and educational purposes:
- Demonstrates POMDP applications in healthcare
- Provides a framework for testing medical diagnostic algorithms
- Offers visualization of Bayesian belief updates in real-time
- Supports research into symptom extraction from natural language

## Disclaimer

PulseAI is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. 