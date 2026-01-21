/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#09090b', // Deep Zinc/Graphite
        surface: '#121212',
        text: '#e4e4e7', // Off-white

        // Deep Teal & Crimson Palette
        accent: '#0d9488', // Teal 600
        'accent-glow': '#2dd4bf', // Teal 400
        'accent-dim': 'rgba(13, 148, 136, 0.1)',
        'bubble-user': '#115e59', // Teal 800
        'bubble-ai': '#18181b', // Zinc 900
        
        danger: '#be123c', // Rose 700
        'danger-glow': '#f43f5e', // Rose 500
      },
      fontFamily: {
        sans: ['Inter', 'SF Pro Display', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'breathing': 'breathing 5s ease-in-out infinite',
        
        // --- BLACK HOLE PHYSICS ---
        'spin-slow': 'spin 12s linear infinite',
        'spin-reverse-slow': 'spin-reverse 15s linear infinite',
        'spin-reverse': 'spin-reverse 1s linear infinite', // Бързо въртене за активност
        'pulse-fast': 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        breathing: {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.2' },
          '50%': { transform: 'scale(1.05)', opacity: '0.5' },
        },
        // --- NEW KEYFRAMES ---
        'spin-reverse': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(-360deg)' },
        }
      }
    },
  },
  plugins: [],
}
