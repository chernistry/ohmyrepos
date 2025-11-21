import { useState, useRef } from 'react';
import { toast } from 'sonner';

export interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const abortControllerRef = useRef<AbortController | null>(null);

    const sendMessage = async (content: string) => {
        if (!content.trim()) return;

        const userMessage: Message = { role: 'user', content };
        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        abortControllerRef.current = new AbortController();

        try {
            // Add placeholder for assistant message
            setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: [...messages, userMessage] }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) throw new Error('Failed to send message');
            if (!response.body) throw new Error('No response body');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                assistantMessage += chunk;

                setMessages(prev => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1] = { role: 'assistant', content: assistantMessage };
                    return newMessages;
                });
            }
        } catch (error: any) {
            if (error.name !== 'AbortError') {
                toast.error('Failed to send message');
                console.error(error);
            }
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    };

    const stop = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setIsLoading(false);
        }
    };

    const clear = () => {
        setMessages([]);
    };

    return { messages, sendMessage, isLoading, stop, clear };
}
