/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            animation: {
                'pulse-slow': 'pulse-slow 6s ease-in-out infinite',
                'fade-in-up': 'fade-in-up 0.3s ease-out forwards',
                'fade-in': 'fade-in 0.5s ease-out',
            },
            keyframes: {
                'pulse-slow': {
                    '0%, 100%': { opacity: '0.2', transform: 'scale(1)' },
                    '50%': { opacity: '0.3', transform: 'scale(1.05)' },
                },
                'fade-in-up': {
                    'from': { opacity: '0', transform: 'translateY(10px)' },
                    'to': { opacity: '1', transform: 'translateY(0)' },
                },
                'fade-in': {
                    'from': { opacity: '0' },
                    'to': { opacity: '1' },
                },
            },
            boxShadow: {
                'glow': '0 0 10px rgba(59, 130, 246, 0.5)',
            },
            backgroundImage: {
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
            },
        },
    },
    plugins: [],
} 