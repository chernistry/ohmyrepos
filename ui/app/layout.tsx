import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains-mono" });

export const metadata: Metadata = {
  title: "Oh My Repos",
  description: "Semantic search for GitHub repositories",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body suppressHydrationWarning className={`${inter.variable} ${jetbrainsMono.variable} bg-background text-primary flex h-screen overflow-hidden selection:bg-accent/30`}>
        <Sidebar />
        <main className="flex-1 relative overflow-y-auto scrollbar-hide">
          {children}
        </main>
        <Toaster position="bottom-right" theme="dark" />
      </body>
    </html>
  );
}
