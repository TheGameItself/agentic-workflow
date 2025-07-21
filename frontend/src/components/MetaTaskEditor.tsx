import React, { useState } from 'react';
import { TaskNode } from './TaskTree';

interface MetaTaskEditorProps {
  task: TaskNode;
  onSave: (updated: TaskNode) => void;
  onCancel: () => void;
}

const MetaTaskEditor: React.FC<MetaTaskEditorProps> = ({ task, onSave, onCancel }) => {
  const [title, setTitle] = useState(task.title);
  const [status, setStatus] = useState(task.status);
  const [progress, setProgress] = useState(task.progress || 0);
  const [notes, setNotes] = useState('');

  const handleSave = () => {
    onSave({ ...task, title, status, progress });
  };

  return (
    <div className="bg-muted rounded p-4 max-w-md mx-auto">
      <h3 className="font-semibold mb-2">Edit Meta/Partial Task</h3>
      <div className="mb-2">
        <label className="block text-sm mb-1">Title</label>
        <input
          className="w-full p-2 rounded bg-background text-foreground border border-muted"
          value={title}
          onChange={e => setTitle(e.target.value)}
        />
      </div>
      <div className="mb-2">
        <label className="block text-sm mb-1">Status</label>
        <select
          className="w-full p-2 rounded bg-background text-foreground border border-muted"
          value={status}
          onChange={e => setStatus(e.target.value)}
        >
          <option value="pending">Pending</option>
          <option value="in_progress">In Progress</option>
          <option value="partial">Partial</option>
          <option value="completed">Completed</option>
          <option value="blocked">Blocked</option>
        </select>
      </div>
      <div className="mb-2">
        <label className="block text-sm mb-1">Progress (%)</label>
        <input
          type="number"
          min={0}
          max={100}
          className="w-full p-2 rounded bg-background text-foreground border border-muted"
          value={progress}
          onChange={e => setProgress(Number(e.target.value))}
        />
      </div>
      <div className="mb-2">
        <label className="block text-sm mb-1">Notes</label>
        <textarea
          className="w-full p-2 rounded bg-background text-foreground border border-muted"
          value={notes}
          onChange={e => setNotes(e.target.value)}
        />
      </div>
      <div className="flex gap-2 mt-4">
        <button
          className="px-4 py-2 bg-accent text-background rounded hover:bg-accent/80"
          onClick={handleSave}
        >
          Save
        </button>
        <button
          className="px-4 py-2 bg-muted text-foreground rounded border border-muted hover:bg-muted/80"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default MetaTaskEditor; 