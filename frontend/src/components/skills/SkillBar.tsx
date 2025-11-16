//
// CREATE NEW FILE: src/components/skills/SkillBar.tsx
//
import React from 'react';

interface SkillBarProps {
  skill: string;
  you: number; // Proficiency 0-5
  required: number; // Proficiency 0-5
  isMustHave?: boolean;
}

export const SkillBar: React.FC<SkillBarProps> = ({ skill, you, required, isMustHave = false }) => {
  const max = Math.max(you, required, 5);
  const youWidth = (you / max) * 100;
  const reqWidth = (required / max) * 100;
  
  const gap = required - you;
  const isMatch = gap <= 0;
  const isGap = gap > 0;

  return (
    <div className="w-full">
      <div className="mb-1.5 flex justify-between items-center text-sm font-medium">
        <span className="text-foreground">
          {skill}
          {isMustHave && <span className="ml-2 text-xs font-semibold text-red-400">(Must-Have)</span>}
        </span>
        <span className={`font-bold ${isMatch ? 'text-green-400' : isGap ? 'text-yellow-400' : 'text-gray-400'}`}>
          {isMatch ? '✓ Match' : isGap ? `Gap: ${gap}` : 'Not Required'}
        </span>
      </div>
      
      <div className="relative h-5 w-full rounded-full bg-muted/50 overflow-hidden border border-border">
        {/* Required Line "marker" */}
        {required > 0 && (
          <div
            className="absolute top-0 left-0 h-full border-r-2 border-dashed border-red-400"
            style={{ width: `${reqWidth}%` }}
            title={`Required: ${required}/5`}
          >
            <div className="absolute -top-1 right-0 text-xs text-red-400 translate-x-1/2">▼</div>
          </div>
        )}
        
        {/* Your Bar */}
        <div 
          className={`h-full rounded-full ${isMatch ? 'background-gradient' : 'bg-yellow-500'}`}
          style={{ width: `${youWidth}%` }}
          title={`You: ${you}/5`}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs text-muted-foreground">
        <span>You: {you}/5</span>
        <span>Required: {required}/5</span>
      </div>
    </div>
  );
};