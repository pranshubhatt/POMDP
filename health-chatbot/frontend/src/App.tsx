import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { Message, ChatResponse } from './types';
import './animations.css';

const API_URL = 'http://localhost:8000';

function App() {
    const [messages, setMessages] = useState<Message[]>([{
        id: uuidv4(),
        text: 'Hello! I\'m PulseAI, your medical diagnostic assistant. Please describe your symptoms so I can help assess your condition.',
        sender: 'bot',
        timestamp: new Date(),
    }]);
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState(uuidv4());
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (text: string) => {
        // Add user message
        const userMessage: Message = {
            id: uuidv4(),
            text,
            sender: 'user',
            timestamp: new Date(),
        };
        setMessages((prev) => [...prev, userMessage]);
        setLoading(true);

        try {
            const response = await axios.post<ChatResponse>(`${API_URL}/chat`, {
                session_id: sessionId,
                message: text,
            });

            // Add bot message
            const botMessage: Message = {
                id: uuidv4(),
                text: response.data.message,
                sender: 'bot',
                timestamp: new Date(),
                diseases: response.data.diseases,
                severity: response.data.severity,
                suggested_questions: response.data.suggested_questions,
                precautions: response.data.precautions,
            };
            setMessages((prev) => [...prev, botMessage]);
        } catch (error) {
            console.error('Error sending message:', error);
            const errorMessage: Message = {
                id: uuidv4(),
                text: 'Sorry, there was an error processing your message. Please try again.',
                sender: 'bot',
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    // Function to reset the chat session
    const handleResetSession = async () => {
        try {
            // Try to delete the current session
            await axios.delete(`${API_URL}/session/${sessionId}`);
        } catch (error) {
            console.error('Error ending session:', error);
        }

        // Generate a new session ID
        const newSessionId = uuidv4();
        setSessionId(newSessionId);
        
        // Reset messages
        setMessages([{
            id: uuidv4(),
            text: 'Hello! I\'m PulseAI, your medical diagnostic assistant. Please describe your symptoms so I can help assess your condition.',
            sender: 'bot',
            timestamp: new Date(),
        }]);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div className="animate-pulse-slow absolute top-10 left-10 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"></div>
            <div className="animate-pulse-slow absolute bottom-10 right-10 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"></div>
            
            <div className="max-w-3xl mx-auto w-full">
                <div className="backdrop-blur-lg bg-white/10 rounded-3xl shadow-2xl overflow-hidden border border-gray-200/20">
                    <div className="relative">
                        {/* Header with glassmorphism effect */}
                        <div className="bg-gradient-to-r from-blue-600/80 to-indigo-600/80 backdrop-blur-md p-6 flex justify-between items-center border-b border-white/10">
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-white/20 rounded-full">
                                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
                                    </svg>
                                </div>
                                <h1 className="text-2xl font-bold text-white">PulseAI</h1>
                            </div>
                            <button 
                                onClick={handleResetSession}
                                className="transition duration-300 ease-in-out transform hover:scale-105 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg flex items-center space-x-2 border border-white/20"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                                </svg>
                                <span>New Chat</span>
                            </button>
                        </div>
                        
                        {/* Chat container */}
                        <div className="h-[600px] bg-gradient-to-b from-gray-900/70 to-gray-800/70 overflow-hidden flex flex-col">
                            {/* Message area */}
                            <div className="flex-1 overflow-y-auto px-6 py-4 scroll-smooth message-container">
                                <div className="space-y-4 animate-fade-in">
                                    {messages.map((message) => (
                                        <div key={message.id} className="message-animation">
                                            <ChatMessage message={message} />
                                        </div>
                                    ))}
                                    <div ref={messagesEndRef} />
                                </div>
                            </div>
                            
                            {/* Input area with glassmorphism effect */}
                            <div className="border-t border-white/10 bg-gray-800/50 backdrop-blur-md p-4">
                                <ChatInput
                                    onSendMessage={handleSendMessage}
                                    disabled={loading}
                                />
                                {loading && (
                                    <div className="flex justify-center mt-2">
                                        <div className="typing-indicator">
                                            <span></span>
                                            <span></span>
                                            <span></span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div className="text-center mt-4 text-gray-400 text-sm">
                    <p>© 2023 PulseAI. For research purposes only. Not a substitute for professional medical advice.</p>
                </div>
            </div>
        </div>
    );
}

export default App; 