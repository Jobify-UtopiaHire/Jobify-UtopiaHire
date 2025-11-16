//
// CREATE NEW FILE: src/components/skills/ReportCard.tsx
//
import React from 'react';

interface ReportCardProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const ReportCard: React.FC<ReportCardProps> = ({ title, icon, children, className = '' }) => {
  return (
    <div
      className={`glass-effect rounded-2xl border border-border bg-card/80 p-6 shadow-xl ${className}`}
    >
      <div className="mb-4 flex items-center gap-3">
        <span className="text-2xl">{icon}</span>
        <h3 className="text-xl font-semibold text-foreground">{title}</h3>
      </div>
      <div className="space-y-4 text-muted-foreground">{children}</div>
    </div>
  );
};