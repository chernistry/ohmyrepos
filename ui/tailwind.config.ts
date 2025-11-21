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
                background: '#0a0a0a',
                surface: '#111111',
                border: '#333333',
                primary: '#EDEDED',
                secondary: '#A1A1A1',
                accent: '#3B82F6',
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
