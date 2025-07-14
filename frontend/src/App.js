import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import SystemMonitor from './SystemMonitor';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // 從環境變數讀取後端 API 的網址
  // 在本地開發時，它會是 http://localhost:8000
  // 在部署到 Vercel 後，它會自動變成您在 Render 上的後端網址
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

      // *** 修正點 ***
      // 使用 API_URL 變數來組合完整的請求網址
      const res = await axios.post(`${API_URL}/api/backtest`, { 
        strategy: full_strategy_context
      });

      if (res.data.question) {
        setMessages(prev => [...prev, { sender: 'ai', text: res.data.question, type: 'question' }]);
      } else {
        setMessages(prev => [...prev, { sender: 'ai', data: res.data, type: 'result' }]);
      }
    } catch (err) {
      const errorMessage = err.response ? err.response.data.detail : '無法連接到後端伺服器。';
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
      return <div key={index} className="message ai-message error-message"><strong>錯誤:</strong> {msg.text}</div>;
    }

    if (msg.type === 'result' && msg.data) {
      const { stats, tickers } = msg.data;
      return (
        <div key={index} className="message ai-message result-message">
          <h3>回測結果 for {tickers.join(', ')}</h3>
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
            placeholder="請描述您的交易策略..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>傳送</button>
        </form>
      </div>
    </div>
  );
}

export default App;
