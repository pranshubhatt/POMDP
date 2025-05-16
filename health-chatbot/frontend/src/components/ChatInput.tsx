import React, { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
    onSendMessage: (message: string) => void;
    disabled?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled }) => {
    const [message, setMessage] = useState('');
    const inputRef = useRef<HTMLInputElement>(null);
    
    // Focus input on component mount
    useEffect(() => {
        if (inputRef.current) {
            inputRef.current.focus();
        }
    }, []);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (message.trim() && !disabled) {
            onSendMessage(message);
            setMessage('');
        }
    };

    // Handle suggested message click
    const handleSuggestedMessage = (text: string) => {
        if (!disabled) {
            onSendMessage(text);
        }
    };

    // Some example symptom suggestions
    const suggestions = [
        "I have a headache",
        "I feel tired",
        "I have a fever",
        "My chest hurts"
    ];

    return (
        <div className="space-y-3">
            {/* Quick suggestions */}
            <div className="flex flex-wrap gap-2 mb-2">
                {suggestions.map((suggestion, index) => (
                    <button
                        key={index}
                        onClick={() => handleSuggestedMessage(suggestion)}
                        disabled={disabled}
                        className="text-xs px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-100 rounded-full border border-blue-500/30 transition-all duration-200 hover:scale-105 transform"
                    >
                        {suggestion}
                    </button>
                ))}
            </div>
            
            <form onSubmit={handleSubmit} className="relative">
                <input
                    ref={inputRef}
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Describe your symptoms..."
                    className="w-full p-4 pr-16 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-white placeholder-gray-400 shadow-inner"
                    disabled={disabled}
                />
                <button
                    type="submit"
                    disabled={disabled || !message.trim()}
                    className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-lg ${
                        disabled || !message.trim()
                            ? 'bg-gray-600/30 cursor-not-allowed'
                            : 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 pulse-on-hover'
                    } transition-all duration-300 ease-in-out`}
                >
                    <svg
                        className="w-5 h-5 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M14 5l7 7m0 0l-7 7m7-7H3"
                        />
                    </svg>
                </button>
            </form>
        </div>
    );
};

export default ChatInput; 