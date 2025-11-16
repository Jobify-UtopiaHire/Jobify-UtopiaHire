import React from "react";

interface OverallScoreProps {
  score?: number;
}

const OverallScore: React.FC<OverallScoreProps> = ({ score }) => {
  const displayScore = score !== undefined ? score : null;
  const scoreColor = displayScore !== null 
    ? displayScore >= 80 ? 'text-blue-400' 
    : displayScore >= 60 ? 'text-indigo-400' 
    : 'text-purple-400'
    : 'text-slate-400';

  const getScoreLabel = (score: number | null) => {
    if (score === null) return 'Not Available';
    if (score >= 80) return 'Outstanding';
    if (score >= 60) return 'Strong';
    if (score >= 40) return 'Average';
    return 'Needs Work';
  };

  const categories = [
    { name: 'Content Quality', description: 'Relevance and impact of your experience' },
    { name: 'Formatting', description: 'Layout, structure, and readability' },
    { name: 'Keyword Optimization', description: 'Industry-relevant terms and skills' },
    { name: 'Completeness', description: 'All essential sections included' },
    { name: 'Grammar & Spelling', description: 'Professional language use' }
  ];

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-indigo-300">Overall CV Score</h1>
      <p className="text-base mb-6 text-slate-300">This score reflects the overall quality and effectiveness of your CV, including formatting, content, and impact.</p>
      
      <div className="flex flex-col items-center justify-center mb-6 p-6 bg-slate-800/60 backdrop-blur-sm rounded-lg border border-slate-700">
        <div className="flex items-center mb-2">
          <span className={`text-6xl font-extrabold ${scoreColor}`}>
            {displayScore !== null ? displayScore : '--'}
          </span>
          <span className="ml-4 text-xl text-slate-400">/ 100</span>
        </div>
        <span className={`text-lg font-semibold ${scoreColor}`}>
          {getScoreLabel(displayScore)}
        </span>
      </div>

      <div className="mt-6">
        <h2 className="text-xl font-semibold text-slate-100 mb-3">Score Categories</h2>
        <div className="space-y-3">
          {categories.map((category, idx) => (
            <div key={idx} className="p-4 bg-indigo-900/30 backdrop-blur-sm rounded-lg border border-indigo-700/40">
              <div className="font-semibold text-slate-100">{category.name}</div>
              <div className="text-sm text-slate-400 mt-1">{category.description}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 p-4 bg-blue-900/30 backdrop-blur-sm rounded-lg border border-blue-700/40">
        <p className="text-sm text-slate-300">
          <strong>Pro Tip:</strong> Apply the AI-suggested changes to improve your score across all categories!
        </p>
      </div>
    </div>
  );
};

export default OverallScore;
