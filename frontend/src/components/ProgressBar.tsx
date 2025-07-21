import React from 'react';

interface ProgressBarProps {
  percentage: number;
  label?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ percentage, label }) => {
  return (
    <div className="w-full">
      {label && <div className="mb-1 text-xs text-muted">{label}</div>}
      <div className="h-3 bg-muted rounded overflow-hidden">
        <div
          className="h-full bg-accent transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
        />
      </div>
      <div className="text-xs text-muted mt-1 text-right">{percentage}%</div>
    </div>
  );
};

export default ProgressBar; 