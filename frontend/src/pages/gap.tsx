import React, { useState, useEffect } from "react";
import { toast } from "sonner";
import { FullAnalysisResponse, GapItem } from "../types";
import { ReportCard } from "../components/skills/ReportCard";
import { SkillBar } from "../components/skills/SkillBar";
import { useSharedCV } from "../hooks/useSharedCV";
import {
  UploadCloud,
  FileText,
  CheckCircle,
  AlertTriangle,
  ArrowRight,
  Brain,
  BookOpen,
  Edit3,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

// Get the API URL from environment or default
const API_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// localStorage keys for Skills Gap Analyzer
const STORAGE_KEYS = {
  JOB_TITLE: "gap_job_title",
  JOB_DESCRIPTION: "gap_job_description",
};

export default function SkillsGapAnalyzer() {
  const { cvFile, saveCV } = useSharedCV();
  const [jobTitle, setJobTitle] = useState("");
  const [jobDescription, setJobDescription] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<FullAnalysisResponse | null>(null);

  const [showAllMissing, setShowAllMissing] = useState(false);
  const [showAllMatched, setShowAllMatched] = useState(false);

  // Load data from localStorage on mount
  useEffect(() => {
    const savedJobTitle = localStorage.getItem(STORAGE_KEYS.JOB_TITLE);
    const savedJobDescription = localStorage.getItem(
      STORAGE_KEYS.JOB_DESCRIPTION
    );

    if (savedJobTitle) {
      setJobTitle(savedJobTitle);
    }
    if (savedJobDescription) {
      setJobDescription(savedJobDescription);
    }
  }, []);

  // Save job title to localStorage whenever it changes
  useEffect(() => {
    if (jobTitle) {
      localStorage.setItem(STORAGE_KEYS.JOB_TITLE, jobTitle);
    }
  }, [jobTitle]);

  // Save job description to localStorage whenever it changes
  useEffect(() => {
    if (jobDescription) {
      localStorage.setItem(STORAGE_KEYS.JOB_DESCRIPTION, jobDescription);
    }
  }, [jobDescription]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type !== "application/pdf") {
        toast.error("Invalid file type. Please upload a PDF.");
        e.target.value = "";
      } else {
        // Gap analyzer doesn't upload to backend, so no server filename
        saveCV(file);
        toast.success(`File "${file.name}" selected.`);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cvFile) {
      toast.error("Please upload your CV as a PDF.");
      return;
    }
    if (!jobDescription) {
      toast.error("Please paste the job description.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setReport(null);

    const formData = new FormData();
    formData.append("cv_file", cvFile);
    formData.append("job_description", jobDescription);
    formData.append("job_title", jobTitle || "Target Role");

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(
          errData.detail || `HTTP error! Status: ${response.status}`
        );
      }

      const data: FullAnalysisResponse = await response.json();
      setReport(data);
      toast.success("Analysis complete!");
    } catch (err: any) {
      console.error("Analysis failed:", err);
      setError(err.message || "An unknown error occurred.");
      toast.error(err.message || "Analysis failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderGapList = (
    title: string,
    icon: React.ReactNode,
    list: GapItem[],
    isExpanded: boolean,
    setIsExpanded: (value: boolean) => void,
    initialLimit: number = 5
  ) => {
    if (!list || list.length === 0) return null;

    const displayList = isExpanded ? list : list.slice(0, initialLimit);
    const hasMore = list.length > initialLimit;

    return (
      <ReportCard title={title} icon={icon}>
        <div className="space-y-5">
          {displayList.map((item) => (
            <SkillBar
              key={item.skill}
              skill={item.skill}
              you={item.proficiency_you}
              required={item.proficiency_req}
              isMustHave={item.is_must_have}
            />
          ))}
        </div>

        {hasMore && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="mt-4 w-full flex items-center justify-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors py-2 rounded-lg hover:bg-muted/50"
          >
            {isExpanded ? (
              <>
                <ChevronUp size={16} />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown size={16} />
                Show {list.length - initialLimit} More
              </>
            )}
          </button>
        )}
      </ReportCard>
    );
  };

  const gapSkills =
    report?.job_skill_profile.filter((s: GapItem) => s.gap > 0) || [];
  const matchSkills =
    report?.job_skill_profile.filter((s: GapItem) => s.gap <= 0) || [];

  return (
    <div className="container py-16 sm:py-24">
      <div className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-gradient">
          Skills Gap Analyzer
        </h1>
        <p className="mt-4 text-lg text-[#D1D5DB]">
          Upload your CV and paste a job description to see where you match,
          what’s missing, and get an AI-powered plan to close the gap.
        </p>
      </div>

      {/* --- Form Section --- */}
      <form
        onSubmit={handleSubmit}
        className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-2"
      >
        <div className="space-y-8">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              1. Upload Your CV (PDF)
            </label>
            <div className="relative flex items-center justify-center w-full h-32 px-4 py-3 rounded-lg border-2 border-dashed border-border bg-background/50 transition-all duration-200 hover:border-[var(--gradient-1)]">
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="text-center">
                {cvFile ? (
                  <div className="flex items-center gap-2 text-green-500">
                    <FileText size={20} />
                    <span className="font-medium">{cvFile.name}</span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2 text-muted-foreground">
                    <UploadCloud size={24} />
                    <span className="font-medium">
                      Click to upload or drag & drop
                    </span>
                    <span className="text-xs">PDF only</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <label
              htmlFor="jobTitle"
              className="text-sm font-medium text-foreground"
            >
              2. Job Title (Optional)
            </label>
            <input
              id="jobTitle"
              type="text"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              placeholder="E.g., Senior Financial Analyst"
              className="input-field glass-effect"
            />
          </div>
        </div>

        <div className="space-y-2 flex flex-col">
          <label
            htmlFor="jobDescription"
            className="text-sm font-medium text-foreground"
          >
            3. Paste Job Description
          </label>
          <textarea
            id="jobDescription"
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the full job description here..."
            className="input-field glass-effect min-h-[220px] flex-grow resize-none"
          />
        </div>

        <div className="md:col-span-2 flex justify-center mt-4">
          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full max-w-xs"
          >
            {isLoading ? (
              <svg
                className="animate-spin h-5 w-5 text-white mx-auto"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
            ) : (
              "Analyze My Skills Gap"
            )}
          </button>
        </div>
      </form>

      {/* --- Error State --- */}
      {error && (
        <div className="mt-16 text-center">
          <div className="inline-flex items-center gap-2 rounded-lg bg-red-500/10 p-4 text-red-400">
            <AlertTriangle />
            <span className="font-medium">Error: {error}</span>
          </div>
        </div>
      )}

      {/* --- Report Section --- */}
      {report && (
        <div className="mt-24 space-y-8">
          <ReportCard
            title="AI Summary & Overall Score"
            icon={<Brain size={24} />}
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 text-center">
              <div className="rounded-lg bg-muted/50 p-4">
                <div className="text-3xl font-bold text-gradient">
                  {report.quantitative_summary.overall_score.toFixed(0)}%
                </div>
                <div className="text-sm font-medium">Keywords</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-4">
                <div className="text-3xl font-bold text-gradient">
                  {report.ai_scores.coverage.toFixed(0)}%
                </div>
                <div className="text-sm font-medium">Coverage</div>
              </div>
              <div className="rounded-lg bg-muted/50 p-4">
                <div className="text-3xl font-bold text-gradient">
                  {report.ai_scores.depth.toFixed(0)}%
                </div>
                <div className="text-sm font-medium">Depth</div>
              </div>
            </div>
            <p className="text-base text-foreground">{report.ai_summary}</p>
          </ReportCard>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderGapList(
              "Missing Skills (Your Gaps)",
              <AlertTriangle className="text-yellow-400" />,
              gapSkills,
              showAllMissing,
              setShowAllMissing
            )}
            {renderGapList(
              "Matched Skills (Your Strengths)",
              <CheckCircle className="text-green-400" />,
              matchSkills,
              showAllMatched,
              setShowAllMatched
            )}
          </div>

          <ReportCard
            title="Priority Action Plan"
            icon={<ArrowRight className="text-blue-400" />}
          >
            <ul className="list-inside list-disc space-y-3">
              {report.priority_actions.map((action: any, i: number) => (
                <li key={i}>
                  <strong className="text-foreground">{action.action}</strong>
                  <span className="text-xs font-mono p-1 bg-muted/50 rounded ml-2">
                    {action.time_estimate}
                  </span>
                  <p className="pl-5 text-sm">{action.why}</p>
                </li>
              ))}
            </ul>
          </ReportCard>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <ReportCard
              title="Suggested Learning Paths"
              icon={<BookOpen className="text-cyan-400" />}
            >
              {report.learning_paths.map((path: any, i: number) => (
                <div
                  key={i}
                  className="border-b border-border pb-3 mb-3 last:border-b-0"
                >
                  <div className="font-semibold text-foreground">
                    {path.skill}
                  </div>
                  <div>{path.path_title}</div>
                  <div className="text-sm font-medium text-gradient">
                    {path.platform}
                  </div>
                </div>
              ))}
            </ReportCard>

            <ReportCard
              title="AI Resume Edits"
              icon={<Edit3 className="text-purple-400" />}
            >
              {report.resume_edits.map((edit: any, i: number) => (
                <div
                  key={i}
                  className="border-b border-border pb-3 mb-3 last:border-b-0"
                >
                  <div className="text-xs font-semibold uppercase text-red-400">
                    BEFORE
                  </div>
                  <p className="italic opacity-80">"{edit.before}"</p>
                  <div className="text-xs font-semibold uppercase text-green-400 mt-2">
                    AFTER
                  </div>
                  <p className="font-medium text-foreground">"{edit.after}"</p>
                </div>
              ))}
            </ReportCard>
          </div>
        </div>
      )}
    </div>
  );
}
