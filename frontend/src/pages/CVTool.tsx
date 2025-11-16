/**
 * Main App Component
 * Lightweight orchestrator for CV analysis application
 */
import { useState, useEffect } from "react";
import { Sparkles, FileText, Briefcase } from "lucide-react";
import FileUpload from "../components/upload/FileUpload";
import AnalysisContainer from "../components/analysis/AnalysisContainer";
import { Button } from "../components/ui/button";
import LoadingSpinner from "../components/ui/LoadingSpinner";
import ErrorMessage from "../components/ui/ErrorMessage";
import ProgressBar from "../components/ui/ProgressBar";
import { analyzeStructuredCV, validateCVFile, applySuggestion } from "../api";
import { useSharedCV } from "../hooks/useSharedCV";
import type {
  Analysis,
  StructuredCV,
  OriginalFile,
  FieldSuggestion,
} from "../types";

// localStorage keys for CVTool
const STORAGE_KEYS = {
  JOB_DESCRIPTION: "cvtool_job_description",
};

export default function EnhancedCVAnalyzer() {
  const { cvFile, saveCV, clearSharedCV } = useSharedCV();
  const [jobDesc, setJobDesc] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [structuredCV, setStructuredCV] = useState<StructuredCV | null>(null);
  const [cvText, setCvText] = useState("");
  const [originalFile, setOriginalFile] = useState<OriginalFile | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load job description from localStorage on mount
  useEffect(() => {
    const savedJobDesc = localStorage.getItem(STORAGE_KEYS.JOB_DESCRIPTION);
    if (savedJobDesc) {
      setJobDesc(savedJobDesc);
    }
  }, []);

  // Save job description to localStorage whenever it changes
  useEffect(() => {
    if (jobDesc) {
      localStorage.setItem(STORAGE_KEYS.JOB_DESCRIPTION, jobDesc);
    }
  }, [jobDesc]);

  const handleFileSelect = (file: File) => {
    const validation = validateCVFile(file);
    if (!validation.valid) {
      setError(validation.error || "Invalid file");
      return;
    }
    // CVTool doesn't upload to backend, so no server filename
    saveCV(file);
    setError(null);
  };

  const handleFileRemove = () => {
    clearSharedCV();
  };

  const handleAnalyze = async () => {
    if (!cvFile) {
      setError("Please upload a CV");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await analyzeStructuredCV(cvFile, jobDesc);

      // Store structured CV data
      setStructuredCV(data.structured_cv || {});

      // Keep original text for backward compatibility
      const extractedText = data.structured_cv?.original_text || "";
      setCvText(extractedText);

      // Store original file data if available
      if (data.original_file) {
        setOriginalFile(data.original_file);
      }

      const analysisData = data.gemini_analysis?.analysis || {};
      setAnalysis({
        overall_score: analysisData.overall_score || 0,
        ats_score: analysisData.ats_score || 0,
        ...analysisData,
        metadata: data.file_info || {},
        structured: data.structured_cv || {},
      });
    } catch (err) {
      setError(
        (err as Error).message ||
          "Failed to analyze CV. Make sure backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleApplySuggestion = async (
    suggestion: FieldSuggestion
  ): Promise<StructuredCV> => {
    if (!structuredCV) {
      console.error("No structured CV data available");
      throw new Error("No structured CV data available");
    }

    console.log("🔄 Applying suggestion in App.tsx:", suggestion);
    console.log("📋 Current structuredCV before apply:", structuredCV);

    try {
      const result = await applySuggestion(structuredCV, suggestion);

      console.log("✅ Received updated CV from backend:", result.updated_cv);

      // Update the structured CV with the applied suggestion
      setStructuredCV(result.updated_cv);

      // Update analysis to reflect the change
      setAnalysis((prev) =>
        prev
          ? {
              ...prev,
              structured: result.updated_cv,
            }
          : null
      );

      console.log("✅ State updated with new CV data");

      return result.updated_cv;
    } catch (err) {
      console.error("Failed to apply suggestion:", err);
      throw err;
    }
  };

  const handleNewAnalysis = () => {
    setAnalysis(null);
    setStructuredCV(null);
    // Don't clear jobDesc - keep it for next analysis
    setCvText("");
    setOriginalFile(null);
    setError(null);
  };

  // Show loading screen while analyzing
  if (loading) {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center p-8">
        <div className="max-w-2xl w-full">
          <div className="backdrop-blur-xl bg-black/40 rounded-2xl shadow-2xl p-12">
            {/* Animated Icon */}
            <div className="flex justify-center mb-8">
              <div className="relative">
                <Sparkles className="w-20 h-20 text-blue-600 animate-pulse" />
                <div className="absolute inset-0 w-20 h-20 bg-blue-400 rounded-full blur-xl opacity-50 animate-ping" />
              </div>
            </div>

            {/* Title */}
            <h2 className="text-3xl font-bold text-center text-white mb-4">
              Analyzing Your CV
            </h2>
            <p className="text-center text-[#D1D5DB] mb-8">
              Our AI is carefully reviewing your resume...
            </p>

            {/* Progress Bar */}
            <ProgressBar isActive={loading} />

            {/* Tips Section */}
            <div className="mt-8 backdrop-blur-xl p-4 rounded">
              <p className="text-sm text-[#D1D5DB]">
                <strong className="text-white">💡 Did you know?</strong> We
                analyze your CV for ATS compatibility, keyword optimization, and
                industry best practices.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show analysis view if analysis exists
  if (analysis) {
    return (
      <AnalysisContainer
        analysis={analysis}
        cvFile={cvFile}
        cvText={cvText}
        structuredCV={structuredCV}
        originalFile={originalFile}
        onNewAnalysis={handleNewAnalysis}
        onApplySuggestion={handleApplySuggestion}
      />
    );
  }

  // Show upload view
  return (
    <div className="min-h-screen bg-transparent p-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Sparkles className="w-10 h-10 font-bold tracking-tight text-gradient text-violet-700" />
            <h1 className="text-4xl font-bold tracking-tight text-gradient">
              AI CV Analyzer Pro
            </h1>
          </div>
          <p className="text-[#D1D5DB] text-lg">
            Upload your CV in any format → Get comprehensive AI analysis →
            Export to PDF
          </p>
        </div>

        {error && (
          <div className="mb-6">
            <ErrorMessage message={error} onClose={() => setError(null)} />
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-8">
          {/* CV Upload */}
          <div className="backdrop-blur-xl bg-black/40 rounded-xl shadow-lg p-8 ">
            <div className="flex items-center gap-3 mb-6">
              <FileText className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-semibold">Your CV</h2>
            </div>

            <FileUpload
              file={cvFile}
              onFileSelect={handleFileSelect}
              onFileRemove={handleFileRemove}
              onUploadingChange={setUploading}
            />

            <div className="mt-4 backdrop-blur-xl  rounded-lg p-4">
              <p className="text-sm text-[#D1D5DB]">
                <strong className="text-white">Supported formats:</strong> We
                extract text from all major CV formats including Canva exports,
                LaTeX resumes, and standard office documents.
              </p>
            </div>
          </div>

          {/* Job Description */}
          <div className="backdrop-blur-xl bg-black/40 rounded-xl shadow-lg p-8">
            <div className="flex items-center gap-3 mb-6">
              <Briefcase className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-semibold">
                Target Job{" "}
                <span className="text-sm text-gray-500 font-normal">
                  (Optional)
                </span>
              </h2>
            </div>

            <textarea
              className="backdrop-blur-xl bg-transparent w-full h-56 p-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Paste the full job description here (optional)...&#10;&#10;Leave empty to analyze your CV for general best practices and ATS compatibility.&#10;&#10;Or include: requirements, responsibilities, desired skills, and qualifications for job-specific analysis."
              value={jobDesc}
              onChange={(e) => setJobDesc(e.target.value)}
            />

            <div className="mt-4 backdrop-blur-xl rounded-lg p-4">
              <p className="text-sm text-[#D1D5DB]">
                <strong className="text-white">Note:</strong> Job description is
                optional. Without it, we'll analyze your CV for general quality,
                formatting, and ATS compatibility.
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 text-center">
          <div className="has-[button:disabled]:cursor-not-allowed inline-block">
            <Button
              onClick={handleAnalyze}
              disabled={loading || !cvFile || uploading}
              variant="outline"
              size="lg"
              className="transform hover:scale-105 disabled:transform-none"
            >
              {loading ? (
                <LoadingSpinner message="Analyzing Your CV..." />
              ) : (
                "Analyze CV with AI"
              )}
            </Button>
          </div>

          {!cvFile ? (
            <p className="mt-4 text-sm text-[#D1D5DB]">
              Please upload your CV to continue
            </p>
          ) : !jobDesc ? (
            <p className="mt-4 text-sm text-[#D1D5DB] italic">
              Job description is optional - you can analyze your CV for general
              best practices
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
