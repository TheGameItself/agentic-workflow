import React from 'react';

export interface TaskNode {
  id: number;
  title: string;
  status: string;
  isMeta?: boolean;
  progress?: number;
  children?: TaskNode[];
}

interface TaskTreeProps {
  tasks: TaskNode[];
  onSelect?: (task: TaskNode) => void;
  onEdit?: (task: TaskNode) => void;
}

const TaskTree: React.FC<TaskTreeProps> = ({ tasks, onSelect, onEdit }) => {
  const renderNode = (node: TaskNode, depth = 0) => (
    <div key={node.id} className={`pl-${depth * 4} py-1 flex items-center`}> 
      <div
        className={`flex-1 cursor-pointer ${node.isMeta ? 'italic text-accent' : ''}`}
        onClick={() => onSelect && onSelect(node)}
      >
        {node.title}
        {node.progress !== undefined && (
          <span className="ml-2 text-xs text-muted">({node.progress}%)</span>
        )}
        {node.status === 'partial' && (
          <span className="ml-2 px-2 py-0.5 bg-muted text-xs rounded">Partial</span>
        )}
      </div>
      {onEdit && (
        <button
          className="ml-2 px-2 py-0.5 bg-accent text-background rounded text-xs hover:bg-accent/80"
          onClick={() => onEdit(node)}
        >
          Edit
        </button>
      )}
      {node.children && node.children.length > 0 && (
        <div className="ml-4">
          {node.children.map(child => renderNode(child, depth + 1))}
        </div>
      )}
    </div>
  );

  return (
    <div className="bg-muted rounded p-4">
      <h3 className="font-semibold mb-2">Task Tree</h3>
      {tasks.length === 0 ? (
        <div className="text-muted">No tasks found.</div>
      ) : (
        tasks.map(task => renderNode(task))
      )}
    </div>
  );
};

export default TaskTree; 