'use client';

import React, { useState, useRef, useEffect, SVGProps } from 'react';
import ReactMarkdown from 'react-markdown';

// --- Types and Constants ---

type MessageType = {
    id: number;
    role: 'user' | 'assistant' | 'loading';
    content: string;
};

type IconProps = SVGProps<SVGSVGElement>;

const LOCAL_API_URL = 'http://127.0.0.1:8000/query';

// --- Icon Components ---

const BotIcon = (props: IconProps) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-bot-custom">
        <path d="M12 8V4H8"/>
        <rect width="16" height="12" x="4" y="8" rx="2"/>
        <path d="M2 14h2"/>
        <path d="M20 14h2"/>
        <path d="M15 13v2"/>
        <path d="M9 13v2"/>
    </svg>
);

const UserIcon = (props: IconProps) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-user-round-icon lucide-user-round">
        <circle cx="12" cy="8" r="5"/>
        <path d="M20 21a8 8 0 0 0-16 0"/>
    </svg>
);

const SendIcon = (props: IconProps) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-send">
        <path d="m22 2-7 20-4-9-9-4Z"/>
        <path d="M22 2 11 13"/>
    </svg>
);

// --- Typing Indicator Component ---

const TypingIndicator: React.FC = () => (
    <div className="flex items-center space-x-1 p-3 rounded-2xl bg-gray-100 dark:bg-gray-800 rounded-tl-sm shadow-sm text-sm">
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse-slow delay-0"></div>
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse-slow delay-200"></div>
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse-slow delay-400"></div>
        <style jsx global>{`
            @keyframes pulse-slow {
                0%, 100% { opacity: 0.5; }
                50% { opacity: 1; }
            }
            .animate-pulse-slow {
                animation: pulse-slow 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            .delay-0 { animation-delay: 0s; }
            .delay-200 { animation-delay: 0.2s; }
            .delay-400 { animation-delay: 0.4s; }
        `}</style>
    </div>
);

// --- Message Component ---

const Message: React.FC<Pick<MessageType, 'content' | 'role'>> = ({ content, role }) => {
    const isUser = role === 'user';
    const isLoading = role === 'loading';

    type CustomCodeProps = React.ComponentProps<'code'> & { inline?: boolean };

    if (isLoading) {
        return (
            <div className={`flex w-full justify-start mb-4`}>
                <div className="flex items-start space-x-2.5">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full text-lg font-mono bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200">
                        <BotIcon className="w-5 h-5" />
                    </div>
                    <TypingIndicator />
                </div>
            </div>
        );
    }

    return (
        <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            <div className={`flex max-w-lg ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-2.5`}>
                <div className={`flex h-8 w-8 items-center justify-center rounded-full text-lg font-mono bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200`}>
                    {isUser 
                        ? <UserIcon className="w-5 h-5" />
                        : <BotIcon className="w-5 h-5" />
                    }
                </div>
                
                <div className={`
                    p-3 rounded-2xl max-w-xs sm:max-w-sm lg:max-w-md 
                    shadow-md text-sm transition-colors duration-200
                    ${isUser
                        ? 'bg-blue-600 text-white rounded-tr-sm whitespace-pre-wrap'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100 rounded-tl-sm'
                    }
                `}>
                    {isUser ? (
                        content
                    ) : (
                        <ReactMarkdown
                            components={{
                                p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                                a: ({ children, href }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-400 underline">{children}</a>,
                                ul: ({ children }) => <ul className="list-disc ml-5 mb-2 space-y-1">{children}</ul>,
                                ol: ({ children }) => <ol className="list-decimal ml-5 mb-2 space-y-1">{children}</ol>,
                                code: (props: CustomCodeProps) => {
                                    const { inline, children } = props;
                                    if (inline) {
                                        return <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-red-600 dark:text-red-400 text-[0.8rem] font-mono">{children}</code>;
                                    }
                                    return (
                                        <pre className="bg-gray-900 text-gray-100 p-3 my-2 rounded-lg overflow-x-auto text-[0.75rem] font-mono">
                                            <code>{children}</code>
                                        </pre>
                                    );
                                },
                                table: ({ children }) => <div className="overflow-x-auto my-3"><table className="w-full text-left border-collapse my-2 text-xs table-auto border border-gray-400 dark:border-gray-600 rounded-lg overflow-hidden">{children}</table></div>,
                                thead: ({ children }) => <thead className="bg-gray-200 dark:bg-gray-700">{children}</thead>,
                                th: ({ children }) => <th className="border border-gray-300 dark:border-gray-600 p-2 font-semibold text-gray-800 dark:text-gray-100">{children}</th>,
                                td: ({ children }) => <td className="border border-gray-300 dark:border-gray-600 p-2">{children}</td>,
                                hr: () => <hr className="my-4 border-t-2 border-gray-400 dark:border-gray-600" />,
                                //h1: ({ children }) => <h1 className="text-base font-bold mt-4 mb-2">{children}</h1>,
                                //h2: ({ children }) => <h2 className="text-base font-bold mt-3 mb-1">{children}</h2>,
                                // Unutar ReactMarkdown components objekta:
                                h1: ({ children }) => (
                                    <h1 className="text-base font-bold mt-4 mb-2 first:mt-0">
                                        {children}
                                    </h1>
                                ),
                                h2: ({ children }) => (
                                    <h2 className="text-base font-bold mt-3 mb-1 first:mt-0">
                                        {children}
                                    </h2>
                                ),
                                blockquote: ({ children }) => <blockquote className="border-l-4 border-blue-400 pl-3 italic text-gray-600 dark:text-gray-400 my-2">{children}</blockquote>,
                            }}
                        >
                            {content}
                        </ReactMarkdown>
                    )}
                </div>
            </div>
        </div>
    );
};

// --- API Function ---

async function callLocalAPI(userQuery: string) {
    const payload = {
        query: userQuery 
    };

    const fetchOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    };

    try {
        const response = await fetch(LOCAL_API_URL, fetchOptions);

        if (!response.ok) {
            const errorBody = await response.json();
            const detail = errorBody.detail || 'Unknown server error';
            return { text: `[Error: HR API Status ${response.status}. Detail: ${detail}]` };
        }

        const result = await response.json();
        const responseText = result.response; 

        if (responseText) {
            return { text: responseText };
        } else {
            return { text: "[Error: The server returned an empty 'response' field.]" };
        }

    } catch (error) {
        console.error("Local API call failed:", error);
        return { text: "[Error: Could not connect to the HR API service. Is the Docker container running on port 8000?]" };
    }
}

// --- Main Chat Application Component (Fixed for iframe) ---
export default function App() {
    const initialMessage: MessageType = {
    id: 1,
    role: 'assistant',
    content: "# Dobrodošli u Hrstud AI\n\nJa sam vaš AI asistent, spreman odgovoriti na pitanja vezana za fakultet Hrvatskih studija.\n\n---\n\n# Welcome to Hrstud AI\n\nI am your AI assistant, ready to answer questions related to the Faculty of Croatian Studies.",
    };

    const [messages, setMessages] = useState<MessageType[]>([initialMessage]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSend = async () => {
        if (input.trim() === '' || isLoading) return;

        const userQuery = input.trim();
        const newUserMessage: MessageType = {
            id: Date.now(),
            role: 'user',
            content: userQuery,
        };
        
        setMessages(prev => [
            ...prev, 
            newUserMessage,
            { id: Date.now() + 1, role: 'loading', content: '...' }
        ]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await callLocalAPI(userQuery);

            setMessages(prev => {
                const updatedMessages = prev.filter(msg => msg.role !== 'loading');
                return [
                    ...updatedMessages,
                    {
                        id: Date.now() + 2,
                        role: 'assistant',
                        content: response.text,
                    }
                ];
            });
        } catch (error) {
            console.error("Chat failed:", error);
            setMessages(prev => {
                const updatedMessages = prev.filter(msg => msg.role !== 'loading');
                return [
                    ...updatedMessages,
                    {
                        id: Date.now() + 2,
                        role: 'assistant',
                        content: "[Error: An internal application error occurred in the frontend.]",
                    }
                ];
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            {/* CSS Reset for iframe compatibility */}
            <style>{`
                html, body, #root {
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    width: 100%;
                    overflow: hidden;
                }
            `}</style>
            
            <div className="relative w-full h-full bg-white dark:bg-gray-950 flex flex-col overflow-hidden">
                
                {/* Header */}
                <header className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 sticky top-0 z-10">
                    <div className="flex items-center space-x-3">
                        <BotIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                            Hrstud AI
                        </h1>
                    </div>
                    <div className="flex items-center space-x-2">
                        <span className={`h-2 w-2 rounded-full ${isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-green-600'}`}></span>
                        <span className={`text-xs font-medium ${isLoading ? 'text-yellow-600 dark:text-yellow-400' : 'text-green-600 dark:text-green-400'}`}>
                            {isLoading ? 'Thinking...' : 'Online'}
                        </span>
                    </div>
                </header>

                {/* Message Area */}
                <main className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
                    {messages.map((msg) => (
                        <Message key={msg.id} content={msg.content} role={msg.role as 'user' | 'assistant'} />
                    ))}
                    <div ref={messagesEndRef} />
                </main>

                {/* Input Area */}
                <footer className="p-4 sm:p-6 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
                    <div className="flex items-end space-x-3">
                        <textarea
                            className="flex-1 min-h-[40px] max-h-40 p-3 text-sm rounded-xl border border-gray-300 dark:border-gray-700 
                                     focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none 
                                     bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors duration-200
                                     disabled:opacity-75 disabled:cursor-not-allowed"
                            placeholder={isLoading ? "Odgovor AI..." : "Pitajte AI nešto..."}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            rows={1}
                            disabled={isLoading}
                        />
                        <button
                            onClick={handleSend}
                            disabled={input.trim() === '' || isLoading}
                            className="p-3 bg-blue-600 text-white rounded-xl shadow-md hover:bg-blue-700 
                                     transition-all duration-300 disabled:bg-gray-300 disabled:text-gray-500 
                                     dark:disabled:bg-gray-700 dark:disabled:text-gray-500 flex-shrink-0"
                            aria-label="Send message"
                        >
                            <SendIcon className="w-5 h-5" />
                        </button>
                    </div>
                </footer>
            </div>
        </>
    );
};
