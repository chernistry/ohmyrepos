import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: '#1e1e1e',
                surface: '#252526',
                border: '#3e3e3e',
                primary: '#d4d4d4',
                secondary: '#808080',
                accent: '#569cd6',
                'vs-blue': '#569cd6',
                'vs-light-blue': '#9cdcfe',
                'vs-cyan': '#4ec9b0',
                'vs-yellow': '#dcdcaa',
                'vs-orange': '#ce9178',
                'vs-green': '#7ca668',
                'vs-purple': '#c586c0',
                'vs-red': '#f44747',
            },
            fontFamily: {
                sans: ['var(--font-inter)'],
                mono: ['var(--font-jetbrains-mono)'],
            },
            animation: {
                'spin-slow': 'spin 3s linear infinite',
            },
        },
    },
    plugins: [],
};
export default config;
