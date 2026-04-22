/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'gd-bg': '#0b0f1a',
        'gd-panel': '#141a2a',
        'gd-accent': '#39a0ff',
        'gd-soft': '#2a3249',
      },
    },
  },
  plugins: [],
};
