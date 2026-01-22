/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme colors
        background: '#0d1117',
        surface: '#161b22',
        border: '#30363d',
        // Trading colors
        bullish: '#26a69a',
        bearish: '#ef5350',
        accent: '#58a6ff',
        // Text colors
        primary: '#c9d1d9',
        secondary: '#8b949e',
        muted: '#6e7681',
      },
    },
  },
  plugins: [],
}
