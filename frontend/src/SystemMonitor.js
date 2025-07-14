import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SystemMonitor = () => {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');

  // 從環境變數讀取後端 API 的網址
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // *** 修正點 ***
        // 使用 API_URL 變數來組合完整的請求網址
        const res = await axios.get(`${API_URL}/api/system_stats`);
        setStats(res.data);
        setError('');
      } catch (err) {
        setError('無法獲取系統狀態。');
        console.error(err);
      }
    };

    const intervalId = setInterval(fetchStats, 5000); // 每 5 秒獲取一次狀態

    return () => clearInterval(intervalId); // 組件卸載時清除定時器
  }, [API_URL]); // 將 API_URL 加入依賴項

  if (error) {
    return <div className="system-monitor error">{error}</div>;
  }

  if (!stats) {
    return <div className="system-monitor">正在載入系統狀態...</div>;
  }

  return (
    <div className="system-monitor">
      <div className="stat-item">CPU: <span>{stats.cpu_usage}</span></div>
      <div className="stat-item">記憶體: <span>{stats.memory_usage}</span></div>
      {stats.gpus && stats.gpus.map(gpu => (
        <div key={gpu.id} className="gpu-stat-item">
          <div className="stat-item">GPU {gpu.id} ({gpu.name}): <span>{gpu.load}</span></div>
          <div className="stat-item">GPU 記憶體: <span>{gpu.memoryUtil} ({gpu.memoryUsed}/{gpu.memoryTotal})</span></div>
        </div>
      ))}
      {stats.gpus && stats.gpus.length === 0 && <div className="stat-item">未偵測到 GPU。</div>}
    </div>
  );
};

export default SystemMonitor;
