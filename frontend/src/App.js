import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import SystemMonitor from './SystemMonitor';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // �q�����ܼ�Ū����� API �����}
  // �b���a�}�o�ɡA���|�O http://localhost:8000
  // �b���p�� Vercel ��A���|�۰��ܦ��z�b Render �W����ݺ��}
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setInput('');

    try {
      const full_strategy_context = [...messages, userMessage]
        .map(m => `${m.sender}: ${m.text}`)
        .join('\n');

      // *** �ץ��I ***
      // �ϥ� API_URL �ܼƨӲզX���㪺�ШD���}
      const res = await axios.post(`${API_URL}/api/backtest`, { 
        strategy: full_strategy_context
      });

      if (res.data.question) {
        setMessages(prev => [...prev, { sender: 'ai', text: res.data.question, type: 'question' }]);
      } else {
        setMessages(prev => [...prev, { sender: 'ai', data: res.data, type: 'result' }]);
      }
    } catch (err) {
      const errorMessage = err.response ? err.response.data.detail : '�L�k�s�����ݦ��A���C';
      setMessages(prev => [...prev, { sender: 'ai', text: errorMessage, type: 'error' }]);
    }

    setIsLoading(false);
  };

  const formatKey = (key) => {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const renderMessage = (msg, index) => {
    if (msg.sender === 'user') {
      return <div key={index} className="message user-message">{msg.text}</div>;
    }

    if (msg.type === 'question') {
      return <div key={index} className="message ai-message">{msg.text}</div>;
    }

    if (msg.type === 'error') {
      return <div key={index} className="message ai-message error-message"><strong>���~:</strong> {msg.text}</div>;
    }

    if (msg.type === 'result' && msg.data) {
      const { stats, tickers } = msg.data;
      return (
        <div key={index} className="message ai-message result-message">
          <h3>�^�����G for {tickers.join(', ')}</h3>
          <div className="results-grid">
            {Object.entries(stats).map(([key, value]) => {
              if (['_strategy', '_equity_curve', '_trades'].includes(key)) return null;
              return (
                <div key={key} className="result-item">
                  <span className="result-key">{formatKey(key)}</span>
                  <span className="result-value">{value}</span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="App">
      <SystemMonitor />
      <div className="chat-container">
        <div className="message-list">
          {messages.map(renderMessage)}
          {isLoading && <div className="message ai-message typing-indicator"><span></span><span></span><span></span></div>}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="message-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="�дy�z�z���������..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>�ǰe</button>
        </form>
      </div>
    </div>
  );
}

export default App;
