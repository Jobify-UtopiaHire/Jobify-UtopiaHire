import React from 'react';

interface CVStrengthIndicatorProps {
  atsScore?: number;
  overallScore?: number;
  suggestionsCount?: number;
  appliedCount?: number;
}

const CVStrengthIndicator: React.FC<CVStrengthIndicatorProps> = ({ 
  atsScore = 0, 
  overallScore = 0, 
  suggestionsCount = 0,
  appliedCount = 0
}) => {
  const avgScore = atsScore && overallScore ? Math.round((atsScore + overallScore) / 2) : 0;
  const completionRate = suggestionsCount > 0 ? Math.round((appliedCount / suggestionsCount) * 100) : 0;

  const getStrengthLevel = (score: number) => {
    if (score >= 80) return { label: 'Excellent', color: 'bg-green-500', textColor: 'text-green-700' };
    if (score >= 60) return { label: 'Good', color: 'bg-blue-500', textColor: 'text-blue-700' };
    if (score >= 40) return { label: 'Fair', color: 'bg-yellow-500', textColor: 'text-yellow-700' };
    return { label: 'Needs Work', color: 'bg-red-500', textColor: 'text-red-700' };
  };

  const strength = getStrengthLevel(avgScore);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-300">CV Strength</span>
        <span className={`text-sm font-semibold ${strength.textColor}`}>{strength.label}</span>
      </div>
      
      <div className="w-full bg-slate-700 rounded-full h-2.5">
        <div 
          className={`h-2.5 rounded-full ${strength.color} transition-all duration-500`}
          style={{ width: `${avgScore}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-3 bg-slate-800/60 rounded-lg">
          <div className="text-xs text-slate-400 mb-1">Average Score</div>
          <div className="text-2xl font-bold text-slate-100">{avgScore}</div>
        </div>
        <div className="p-3 bg-slate-800/60 rounded-lg">
          <div className="text-xs text-slate-400 mb-1">Improvements</div>
          <div className="text-2xl font-bold text-slate-100">{appliedCount}/{suggestionsCount}</div>
        </div>
      </div>

      {completionRate < 100 && suggestionsCount > 0 && (
        <div className="mt-4 p-3 bg-indigo-900/30 border border-indigo-700/40 rounded-lg">
          <p className="text-xs text-indigo-300">
            Apply {suggestionsCount - appliedCount} more suggestion{suggestionsCount - appliedCount !== 1 ? 's' : ''} to maximize your CV's potential!
          </p>
        </div>
      )}
    </div>
  );
};

export default CVStrengthIndicator;
