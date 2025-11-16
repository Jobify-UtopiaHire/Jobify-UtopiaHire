/**
 * StatisticsTab Component
 * Displays section analysis and other statistics
 */
import { CheckCircle, FileText } from "lucide-react";
import type { Analysis, SectionAnalysis } from "../../../types";

interface StatisticsTabProps {
  analysis: Analysis;
}

interface ExtendedSectionAnalysis extends SectionAnalysis {
  name?: string;
  quality_score?: number;
  feedback?: string;
}

export default function StatisticsTab({ analysis }: StatisticsTabProps) {
  const sectionAnalysis = (analysis.section_analysis ||
    []) as ExtendedSectionAnalysis[];

  const getScoreColor = (score: number): string => {
    if (score >= 8) return "text-green-600";
    if (score >= 6) return "text-yellow-600";
    if (score >= 4) return "text-orange-600";
    return "text-red-600";
  };

  if (sectionAnalysis.length === 0) {
    return (
      <div className="text-center py-12 text-[#D1D5DB]">
        <FileText className="w-12 h-12 mx-auto mb-3 text-gray-500" />
        <p>No section analysis available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {sectionAnalysis.map((section, idx) => (
        <div
          key={idx}
          className="backdrop-blur-xl bg-black/40 rounded-lg shadow-sm border border-gray-700 p-5"
        >
          <div className="flex items-start justify-between mb-3">
            <div>
              <h3 className="font-bold text-lg text-white">
                {section.name || section.section}
              </h3>
              {section.quality_score !== undefined && (
                <div className="flex items-center gap-2 mt-1">
                  <div className="text-sm text-[#D1D5DB]">Quality Score:</div>
                  <div
                    className={`text-lg font-bold ${getScoreColor(
                      section.quality_score
                    )}`}
                  >
                    {section.quality_score}/10
                  </div>
                </div>
              )}
            </div>
          </div>

          {section.feedback && (
            <div className="mb-3 p-3 backdrop-blur-xl bg-blue-950/30 border border-blue-700 rounded">
              <p className="text-sm text-blue-300">{section.feedback}</p>
            </div>
          )}

          {section.suggestions && section.suggestions.length > 0 && (
            <div>
              <h4 className="font-semibold text-sm text-[#D1D5DB] mb-2">
                Suggestions:
              </h4>
              <ul className="space-y-1">
                {section.suggestions.map((suggestion, sidx) => (
                  <li
                    key={sidx}
                    className="flex items-start gap-2 text-sm text-[#D1D5DB]"
                  >
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
