import { useState, useEffect } from 'react';
import './App.css';

interface Message {
  sender: 'user' | 'ai';
  text: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
}

function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Load chats from Local Storage
  useEffect(() => {
    const savedSessions = localStorage.getItem('cricket_chats');
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions);
      setSessions(parsed);
      if (parsed.length > 0) setActiveSessionId(parsed[0].id);
    } else {
      createNewChat();
    }
  }, []);

  // Save chats to Local Storage
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('cricket_chats', JSON.stringify(sessions));
    }
  }, [sessions]);

  const createNewChat = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: `Match Analysis ${sessions.length + 1}`,
      messages: []
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
  };

  const activeSession = sessions.find(s => s.id === activeSessionId);

  const sendMessage = async () => {
    if (!input.trim() || !activeSessionId) return;

    const userMessage: Message = { sender: 'user', text: input };
    const chatHistory = activeSession?.messages.slice(-6) || [];

    const updatedSessions = sessions.map(s => {
      if (s.id === activeSessionId) return { ...s, messages: [...s.messages, userMessage] };
      return s;
    });
    setSessions(updatedSessions);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.text, history: chatHistory }),
      });

      const data = await response.json();
      const aiMessage: Message = { sender: 'ai', text: data.answer };

      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) return { ...s, messages: [...s.messages, aiMessage] };
        return s;
      }));

    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <div className="sidebar">
        <button className="new-chat-btn" onClick={createNewChat}>
          + NEW MATCH INQUIRY
        </button>
        <div className="session-list">
          {sessions.map(session => (
            <div 
              key={session.id} 
              className={`session-item ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(session.id)}
            >
              🏏 {session.title}
            </div>
          ))}
        </div>
      </div>

      {/* MAIN CHAT */}
      <div className="main-chat">
        <header className="scoreboard-header">
          <div className="header-content">
            <h1>LIVE: AI CRICKET ANALYST</h1>
            <p className="subtitle">BALL-BY-BALL DATA INSIGHTS & MATCH SUMMARIES</p>
          </div>
        </header>

        <div className="chat-window">
          {activeSession?.messages.length === 0 && (
            <div className="empty-state">
              <h2>Welcome to the Commentary Box! 🎙️</h2>
              <p>Ask me about recent match results, then ask follow-up questions!</p>
            </div>
          )}
          
          {activeSession?.messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.sender}`}>
              <div className="message-label">
                {msg.sender === 'user' ? 'YOU' : 'AI EXPERT'}
              </div>
              <div className={`message-bubble ${msg.sender}`}>
                {msg.text}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message-wrapper ai">
               <div className="message-label">AI EXPERT</div>
               <div className="message-bubble ai loading">
                 <em>Reviewing the footage... 📺</em>
               </div>
            </div>
          )}
        </div>

        <div className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask a question, then try a follow-up like 'Who was man of the match?'..."
            disabled={isLoading}
          />
          <button className="send-btn" onClick={sendMessage} disabled={isLoading || !input.trim()}>
            BOWL
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;