import React from "react";

interface ATSScoreProps {
  score?: number;
}

const ATSScore: React.FC<ATSScoreProps> = ({ score }) => {
  const displayScore = score !== undefined ? score : null;
  const scoreColor = displayScore !== null 
    ? displayScore >= 80 ? 'text-green-400' 
    : displayScore >= 60 ? 'text-yellow-400' 
    : 'text-red-400'
    : 'text-slate-400';

  const getScoreLabel = (score: number | null) => {
    if (score === null) return 'Not Available';
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Needs Improvement';
  };

  const tips = [
    "Use keywords from the job description",
    "Include quantifiable achievements",
    "Use standard section headings",
    "Avoid images, charts, and special formatting",
    "Use a simple, clean layout",
    "Include relevant certifications"
  ];

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-green-300 to-emerald-300">ATS Score</h1>
      <p className="text-base mb-6 text-slate-300">Your CV's ATS (Applicant Tracking System) score helps you understand how well your CV will perform in automated screening systems.</p>
      
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
        <h2 className="text-xl font-semibold text-slate-100 mb-3">Tips to Improve Your ATS Score</h2>
        <ul className="space-y-2">
          {tips.map((tip, idx) => (
            <li key={idx} className="flex items-start">
              <span className="text-green-400 mr-2">✓</span>
              <span className="text-slate-300">{tip}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ATSScore;
