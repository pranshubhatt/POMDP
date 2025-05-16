import React from 'react';
import { Message } from '../types';

interface ChatMessageProps {
    message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
    const isBot = message.sender === 'bot';
    
    // Format diseases for display
    const formatDiseases = (diseases?: Record<string, number>) => {
        if (!diseases || Object.keys(diseases).length === 0) return null;
        
        return (
            <div className="mt-3 bg-black/20 rounded-lg p-3 border border-white/10">
                <p className="font-semibold text-white/90 text-sm">Potential conditions:</p>
                <div className="mt-2 space-y-2">
                    {Object.entries(diseases)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 3)
                        .map(([disease, probability]) => {
                            // Calculate percentage and width for progress bar
                            const percentage = (probability * 100).toFixed(1);
                            const width = `${Math.max(5, Math.min(100, Number(percentage)))}%`;
                            
                            return (
                                <div key={disease} className="relative">
                                    <div className="flex justify-between items-center mb-1">
                                        <p className="text-sm capitalize font-medium text-white/80">{disease}</p>
                                        <span className="text-xs text-white/70">{percentage}%</span>
                                    </div>
                                    <div className="w-full bg-gray-700/50 rounded-full h-2">
                                        <div 
                                            className="h-2 rounded-full bg-gradient-to-r from-blue-400 to-indigo-500"
                                            style={{ width }}
                                        ></div>
                                    </div>
                                </div>
                            );
                        })}
                </div>
            </div>
        );
    };

    // Format severity
    const getSeverityLabel = (severity?: number) => {
        if (severity === undefined) return null;
        
        let label = 'Low';
        let colorClass = 'from-green-400 to-green-500';
        let textClass = 'text-green-300';
        
        if (severity > 7) {
            label = 'High';
            colorClass = 'from-red-500 to-red-600';
            textClass = 'text-red-300';
        } else if (severity > 5) {
            label = 'Moderate';
            colorClass = 'from-yellow-400 to-yellow-500';
            textClass = 'text-yellow-300';
        } else if (severity > 3) {
            label = 'Low';
            colorClass = 'from-green-400 to-green-500';
            textClass = 'text-green-300';
        } else {
            label = 'Very Low';
            colorClass = 'from-blue-400 to-blue-500';
            textClass = 'text-blue-300';
        }
        
        return (
            <div className="mt-3 flex items-center space-x-2">
                <div className="flex-shrink-0">
                    <div className="relative h-8 w-8 rounded-full bg-black/30 flex items-center justify-center">
                        <div className={`absolute inset-0 rounded-full bg-gradient-to-r ${colorClass} opacity-20 animate-pulse`}></div>
                        <span className={`text-xs font-bold ${textClass}`}>{severity.toFixed(1)}</span>
                    </div>
                </div>
                <div>
                    <span className="font-semibold text-white/90">Severity: </span>
                    <span className={`${textClass} font-semibold`}>{label}</span>
                </div>
            </div>
        );
    };

    // Format suggested questions
    const formatSuggestedQuestions = (questions?: string[]) => {
        if (!questions || questions.length === 0) return null;
        
        return (
            <div className="mt-3 bg-white/5 rounded-lg p-3 border border-white/10">
                <p className="font-semibold text-white/90 text-sm mb-2">Follow-up questions:</p>
                <div className="space-y-1">
                    {questions.map((question, index) => (
                        <div key={index} className="flex items-start">
                            <div className="flex-shrink-0 mt-1 mr-2">
                                <div className="h-2 w-2 rounded-full bg-blue-400"></div>
                            </div>
                            <p className="text-sm text-white/80">{question}</p>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    // Format precautions
    const formatPrecautions = (precautions?: string[]) => {
        if (!precautions || precautions.length === 0) return null;
        
        return (
            <div className="mt-3 bg-red-900/20 rounded-lg p-3 border border-red-500/20">
                <p className="font-semibold text-red-300 text-sm flex items-center">
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                    </svg>
                    Precautions:
                </p>
                <div className="mt-2 space-y-1">
                    {precautions.map((precaution, index) => (
                        <div key={index} className="flex items-start">
                            <div className="flex-shrink-0 mt-1 mr-2">
                                <div className="h-2 w-2 rounded-full bg-red-400"></div>
                            </div>
                            <p className="text-sm text-white/80">{precaution}</p>
                        </div>
                    ))}
                </div>
            </div>
        );
    };
    
    return (
        <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
            {isBot && (
                <div className="flex-shrink-0 mr-3">
                    <div className="h-9 w-9 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-glow">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
                        </svg>
                    </div>
                </div>
            )}
            
            <div
                className={`max-w-[85%] rounded-xl p-4 ${
                    isBot
                        ? 'bg-gradient-to-br from-gray-800/90 to-gray-900/90 text-white border border-white/10 shadow-md'
                        : 'bg-gradient-to-br from-blue-500 to-blue-600 text-white border border-blue-400/30 shadow-md'
                }`}
            >
                <p className="text-sm leading-relaxed whitespace-pre-line">{message.text}</p>
                
                {isBot && (
                    <div className="mt-2">
                        {getSeverityLabel(message.severity)}
                        {formatDiseases(message.diseases)}
                        {formatSuggestedQuestions(message.suggested_questions)}
                        {formatPrecautions(message.precautions)}
                        
                        <div className="mt-3 flex justify-between items-center border-t border-white/10 pt-2">
                            <span className="text-xs text-white/50">
                                {new Date(message.timestamp).toLocaleTimeString()}
                            </span>
                            {message.diseases && Object.keys(message.diseases).length > 0 && (
                                <span className="text-xs bg-white/10 rounded-full px-2 py-0.5 text-white/60">
                                    Medical Analysis
                                </span>
                            )}
                        </div>
                    </div>
                )}
                
                {!isBot && (
                    <div className="mt-2 flex justify-end">
                        <span className="text-xs text-white/70">
                            {new Date(message.timestamp).toLocaleTimeString()}
                        </span>
                    </div>
                )}
            </div>
            
            {!isBot && (
                <div className="flex-shrink-0 ml-3">
                    <div className="h-9 w-9 rounded-full bg-gradient-to-r from-green-500 to-green-600 flex items-center justify-center text-white shadow-glow">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                        </svg>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChatMessage; 