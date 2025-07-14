import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SystemMonitor = () => {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');

  // �q�����ܼ�Ū����� API �����}
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // *** �ץ��I ***
        // �ϥ� API_URL �ܼƨӲզX���㪺�ШD���}
        const res = await axios.get(`${API_URL}/api/system_stats`);
        setStats(res.data);
        setError('');
      } catch (err) {
        setError('�L�k����t�Ϊ��A�C');
        console.error(err);
      }
    };

    const intervalId = setInterval(fetchStats, 5000); // �C 5 ������@�����A

    return () => clearInterval(intervalId); // �ե�����ɲM���w�ɾ�
  }, [API_URL]); // �N API_URL �[�J�̿ඵ

  if (error) {
    return <div className="system-monitor error">{error}</div>;
  }

  if (!stats) {
    return <div className="system-monitor">���b���J�t�Ϊ��A...</div>;
  }

  return (
    <div className="system-monitor">
      <div className="stat-item">CPU: <span>{stats.cpu_usage}</span></div>
      <div className="stat-item">�O����: <span>{stats.memory_usage}</span></div>
      {stats.gpus && stats.gpus.map(gpu => (
        <div key={gpu.id} className="gpu-stat-item">
          <div className="stat-item">GPU {gpu.id} ({gpu.name}): <span>{gpu.load}</span></div>
          <div className="stat-item">GPU �O����: <span>{gpu.memoryUtil} ({gpu.memoryUsed}/{gpu.memoryTotal})</span></div>
        </div>
      ))}
      {stats.gpus && stats.gpus.length === 0 && <div className="stat-item">�������� GPU�C</div>}
    </div>
  );
};

export default SystemMonitor;
