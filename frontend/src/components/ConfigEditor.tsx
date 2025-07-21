import React, { useState } from 'react';

const ConfigEditor: React.FC = () => {
  const [config, setConfig] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const loadConfig = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const res = await fetch('/api/config');
      if (!res.ok) throw new Error('Failed to load config');
      const data = await res.json();
      setConfig(data.content || '');
      setSuccess(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const res = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: config }),
      });
      if (!res.ok) throw new Error('Failed to save config');
      setSuccess(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-muted p-6 rounded shadow max-w-2xl mx-auto mt-8">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold">Config Editor</h2>
        <button
          className="px-3 py-1 bg-accent text-white rounded hover:bg-accent/80"
          onClick={loadConfig}
          disabled={loading}
        >
          Load
        </button>
      </div>
      <textarea
        className="w-full h-64 p-2 bg-background text-foreground border border-muted rounded resize-none mb-2"
        value={config}
        onChange={e => setConfig(e.target.value)}
        disabled={loading}
      />
      <div className="flex items-center gap-2">
        <button
          className="px-4 py-2 bg-accent text-white rounded hover:bg-accent/80"
          onClick={saveConfig}
          disabled={loading}
        >
          Save
        </button>
        {loading && <span className="text-xs text-muted">Loading...</span>}
        {error && <span className="text-xs text-red-400">{error}</span>}
        {success && <span className="text-xs text-green-400">Success!</span>}
      </div>
    </div>
  );
};

export default ConfigEditor; 