/**
 * AnalysisTab Component
 * Displays overview, critical issues, and score breakdown
 */
import { AlertTriangle, XCircle } from "lucide-react";
import type { Analysis } from "../../../types";

interface AnalysisTabProps {
  analysis: Analysis;
}

export default function AnalysisTab({ analysis }: AnalysisTabProps) {
  const criticalIssues = (analysis as any).critical_issues || [];
  const jobMatch = analysis.job_match_analysis || {
    relevance_score: 0,
  };

  const getScoreColor = (score: number): string => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    if (score >= 40) return "text-orange-600";
    return "text-red-600";
  };

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <div className="backdrop-blur-xl bg-black/40 rounded-lg shadow-sm border border-gray-700 p-6">
        <h2 className="text-xl font-bold mb-4 text-white">Executive Summary</h2>
        <p className="text-[#D1D5DB] leading-relaxed">
          {(analysis as any).summary || "No summary available"}
        </p>
      </div>

      {/* Critical Issues */}
      {criticalIssues.length > 0 && (
        <div className="backdrop-blur-xl bg-red-950/30 border-2 border-red-700 rounded-lg p-6">
          <h3 className="font-bold text-red-400 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" />
            Critical Issues ({criticalIssues.length})
          </h3>
          <ul className="space-y-2">
            {criticalIssues.map((issue: string, idx: number) => (
              <li key={idx} className="flex items-start gap-2 text-red-300">
                <XCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <span>{issue}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Score Breakdown */}
      <div className="backdrop-blur-xl bg-black/40 rounded-lg shadow-sm border border-gray-700 p-6">
        <h3 className="font-bold mb-4 text-white">Score Breakdown</h3>
        <div className="space-y-4">
          {[
            { label: "Overall Quality", score: analysis.overall_score || 0 },
            { label: "ATS Compatibility", score: analysis.ats_score || 0 },
            {
              label: "Readability",
              score: (analysis as any).readability_score || 0,
            },
            { label: "Job Match", score: jobMatch.relevance_score || 0 },
          ].map((item, idx) => (
            <div key={idx}>
              <div className="flex justify-between mb-2">
                <span className="font-medium text-[#D1D5DB]">{item.label}</span>
                <span className={`font-bold ${getScoreColor(item.score)}`}>
                  {item.score}/100
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all ${
                    item.score >= 80
                      ? "bg-green-500"
                      : item.score >= 60
                      ? "bg-yellow-500"
                      : item.score >= 40
                      ? "bg-orange-500"
                      : "bg-red-500"
                  }`}
                  style={{ width: `${item.score}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
