export interface Message {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
    diseases?: Record<string, number>;
    severity?: number;
    suggested_questions?: string[];
    precautions?: string[];
}

export interface ChatResponse {
    message: string;
    diseases: Record<string, number>;
    severity: number;
    suggested_questions: string[];
    precautions: string[];
}

export interface ChatRequest {
    session_id: string;
    message: string;
} 