/**
 * FieldSuggestions Component
 * Displays field-targeted suggestions with Apply buttons for structured CV
 */
import { useState } from "react";
import {
  CheckCircle,
  AlertCircle,
  Target,
  Check,
  Undo2,
  Download,
  Code2,
} from "lucide-react";
import { exportStructuredCVToPdf } from "../../utils/exportPdf";
import { downloadLatexCode } from "../../utils/exportLatex";
import { getLatexCode } from "../../api";
import type { FieldSuggestion, StructuredCV } from "../../types";

interface FieldSuggestionsProps {
  suggestions: FieldSuggestion[];
  onApplySuggestion: (suggestion: FieldSuggestion) => Promise<StructuredCV>;
  structuredCV: StructuredCV | null;
}

export default function FieldSuggestions({
  suggestions,
  onApplySuggestion,
  structuredCV,
}: FieldSuggestionsProps) {
  const [appliedSuggestions, setAppliedSuggestions] = useState<Set<number>>(
    new Set()
  );
  const [applyingId, setApplyingId] = useState<number | null>(null);
  const [revertingId, setRevertingId] = useState<number | null>(null);
  const [prevValues, setPrevValues] = useState<{ [key: number]: string }>({}); // suggestionId -> previous value for undo
  const [exportingPdf, setExportingPdf] = useState(false);
  const [gettingLatex, setGettingLatex] = useState(false);
  const [cachedLatexCode, setCachedLatexCode] = useState<string | null>(null); // Cache LaTeX code to avoid redundant API calls
  const [userInputValues, setUserInputValues] = useState<{
    [key: number]: string;
  }>({}); // suggestionId -> user input value

  const getSeverityBadge = (severity: string): string => {
    const colors: { [key: string]: string } = {
      critical: "bg-red-900/50 text-red-300 border-red-700",
      high: "bg-orange-900/50 text-orange-300 border-orange-700",
      medium: "bg-yellow-900/50 text-yellow-300 border-yellow-700",
      low: "bg-blue-900/50 text-blue-300 border-blue-700",
    };
    return colors[severity] || colors.medium;
  };

  const getFieldPathDisplay = (fieldPath: (string | number)[]): string => {
    if (!fieldPath || fieldPath.length === 0) return "Unknown field";

    // Convert field path to readable format
    // e.g., ["experience", 0, "description"] -> "Experience #1: Description"
    const parts: string[] = [];

    for (let i = 0; i < fieldPath.length; i++) {
      const part = fieldPath[i];

      if (typeof part === "number" || !isNaN(Number(part))) {
        parts.push(`#${parseInt(String(part)) + 1}`);
      } else {
        parts.push(
          String(part).charAt(0).toUpperCase() + String(part).slice(1)
        );
      }
    }

    return parts.join(" → ");
  };

  // Tiny language detector: fr/ar/en heuristics (kept for validation)
  const detectLanguage = (text: string = ""): string => {
    if (!text) return "unknown";
    // Arabic script range
    if (/[\u0600-\u06FF]/.test(text)) return "ar";
    // French indicators: accents and common words
    const frenchAccents = /[àâäéèêëîïôöùûüçœÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒ]/;
    const frenchWords =
      /( le | la | les | des | et | pour | avec | Présent|Septembre|Juin|Août|Mars|Février|École|Ingénieur)/i;
    if (frenchAccents.test(text) || frenchWords.test(text)) return "fr";
    return "en";
  };

  // Check if a suggestion requires user input (contains placeholders or current is "Not provided")
  const requiresUserInput = (suggestion: FieldSuggestion): boolean => {
    const current = String(suggestion.originalValue || "");
    const improved = String(suggestion.improvedValue || "");

    // If current is "Not provided", it definitely needs user input
    if (current === "Not provided" || current.trim() === "") {
      console.log(
        `✅ Field ${suggestion.targetField} needs input - current is "${current}"`
      );
      return true;
    }

    // Check for common placeholder patterns
    const placeholderPatterns = [
      /YYYY/i, // Year placeholders
      /XXXX/i, // Generic placeholders
      /\bX%/i, // Percentage placeholders (e.g., "by X%")
      /\bN\b/i, // Number placeholders (e.g., "N users", "N projects")
      /\[.*?\]/, // Bracketed placeholders
      /e\.g\.,/i, // "e.g.," suggestions
      /\(example/i, // "(example...)"
      /\(if quantifiable\)/i, // "(if quantifiable)" indicator
      /\{.*?\}/, // Curly brace placeholders
      /<.*?>/, // Angle bracket placeholders
      /Month YYYY/i, // Date format suggestions
      /your\s+\w+/i, // "your something"
      /specific\s+\w+/i, // "specific something"
      /add\s+\w+/i, // "add something"
      /\d+%\s*\(if/i, // Pattern like "20% (if known)"
    ];

    const hasPlaceholder = placeholderPatterns.some((pattern) =>
      pattern.test(improved)
    );
    if (hasPlaceholder) {
      console.log(
        `✅ Field ${suggestion.targetField} needs input - improved value has placeholder: "${improved}"`
      );
    } else {
      console.log(
        `❌ Field ${suggestion.targetField} does NOT need input - current: "${current}", improved: "${improved}"`
      );
    }

    return hasPlaceholder;
  };

  const handleApply = async (suggestion: FieldSuggestion) => {
    setApplyingId(suggestion.suggestionId);

    try {
      // Get the value to apply - either user input or the improved value
      const valueToApply =
        userInputValues[suggestion.suggestionId] || suggestion.improvedValue;

      // Validate that user has provided input if required
      if (
        requiresUserInput(suggestion) &&
        !userInputValues[suggestion.suggestionId]
      ) {
        alert(
          "Please provide the required information in the text field before applying."
        );
        setApplyingId(null);
        return;
      }

      // Language preservation: warn if mismatch
      const origLang = detectLanguage(suggestion.originalValue);
      const improvedLang = detectLanguage(valueToApply);
      if (
        origLang !== "unknown" &&
        improvedLang !== "unknown" &&
        origLang !== improvedLang
      ) {
        const proceed = window.confirm(
          `The improved text appears to be ${improvedLang.toUpperCase()} while the original is ${origLang.toUpperCase()}. Apply anyway and keep the CV language consistent?`
        );
        if (!proceed) {
          setApplyingId(null);
          return;
        }
      }

      // Keep previous value for undo
      setPrevValues((prev) => ({
        ...prev,
        [suggestion.suggestionId]: suggestion.originalValue,
      }));

      // Create modified suggestion with user input value
      const modifiedSuggestion: FieldSuggestion = {
        ...suggestion,
        improvedValue: valueToApply,
      };

      await onApplySuggestion(modifiedSuggestion);

      // Mark as applied and clear user input
      setAppliedSuggestions(
        (prev) => new Set([...prev, suggestion.suggestionId])
      );

      // Clear the user input value after successful application
      setUserInputValues((prev) => {
        const next = { ...prev };
        delete next[suggestion.suggestionId];
        return next;
      });
    } catch (error) {
      console.error("Failed to apply suggestion:", error);
      alert("Failed to apply suggestion. Please try again.");
    } finally {
      setApplyingId(null);
    }
  };

  const handleUndo = async (suggestion: FieldSuggestion) => {
    setRevertingId(suggestion.suggestionId);
    try {
      const previous =
        prevValues[suggestion.suggestionId] ?? suggestion.originalValue;
      const revertSuggestion: FieldSuggestion = {
        ...suggestion,
        improvedValue: previous,
      };
      await onApplySuggestion(revertSuggestion);
      setAppliedSuggestions((prev) => {
        const next = new Set(prev);
        next.delete(suggestion.suggestionId);
        return next;
      });
    } catch (e) {
      console.error("Failed to revert suggestion:", e);
      alert("Failed to revert change.");
    } finally {
      setRevertingId(null);
    }
  };

  if (!suggestions || suggestions.length === 0) {
    return (
      <div className="text-center py-12 text-[#D1D5DB]">
        <AlertCircle className="w-12 h-12 mx-auto mb-3 text-gray-500" />
        <p>No field suggestions available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="backdrop-blur-xl bg-blue-950/30 border border-blue-700 rounded-lg p-4 mb-6">
        <div className="flex items-start gap-3">
          <Target className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-300">
            <p className="font-semibold mb-1">Field-Targeted Suggestions</p>
            <p>
              Click "Apply" to automatically update the specific field in your
              CV. Changes are made directly to the structured data.
            </p>
          </div>
        </div>
      </div>

      {suggestions.map((sugg) => {
        const isApplied = appliedSuggestions.has(sugg.suggestionId);
        const isApplying = applyingId === sugg.suggestionId;
        const needsInput = requiresUserInput(sugg);

        return (
          <div
            key={sugg.suggestionId}
            className={`backdrop-blur-xl rounded-lg shadow-sm border-2 p-5 transition-all ${
              isApplied
                ? "border-green-700 bg-green-950/30"
                : "border-gray-700 bg-black/40"
            }`}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold border ${getSeverityBadge(
                    sugg.severity
                  )}`}
                >
                  {sugg.severity?.toUpperCase()}
                </span>
                <span className="text-xs text-[#D1D5DB] font-medium">
                  {sugg.issue_type}
                </span>
              </div>

              {isApplied ? (
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1 text-green-400 font-semibold text-sm">
                    <CheckCircle className="w-4 h-4" />
                    Applied
                  </div>
                  <button
                    onClick={() => handleUndo(sugg)}
                    disabled={revertingId === sugg.suggestionId}
                    className={`flex items-center gap-1 px-2 py-1 rounded-md text-sm border ${
                      revertingId === sugg.suggestionId
                        ? "bg-gray-700 text-gray-400 border-gray-600"
                        : "border-gray-600 text-[#D1D5DB] hover:bg-white/10"
                    }`}
                    title="Revert to previous value"
                  >
                    <Undo2 className="w-4 h-4" /> Undo
                  </button>
                </div>
              ) : null}
            </div>

            {/* Field Location */}
            <div className="mb-3 text-sm">
              <span className="font-semibold text-[#D1D5DB]">
                Target Field:
              </span>
              <span className="ml-2 text-blue-400 font-medium">
                {getFieldPathDisplay(sugg.fieldPath)}
              </span>
              {sugg.fieldId && (
                <span className="ml-2 text-gray-500 text-xs">
                  ({sugg.fieldId})
                </span>
              )}
            </div>

            {/* Problem Description */}
            <div className="mb-3 text-sm text-[#D1D5DB]">
              <strong>Problem:</strong>{" "}
              {typeof sugg.problem === "string"
                ? sugg.problem
                : JSON.stringify(sugg.problem)}
            </div>

            {/* Original Value */}
            <div className="bg-red-950/30 border border-red-700 rounded p-3 mb-3">
              <div className="text-xs font-semibold text-red-400 mb-1">
                Current:
              </div>
              <div className="text-sm text-gray-200 italic">
                "
                {typeof sugg.originalValue === "string"
                  ? sugg.originalValue
                  : JSON.stringify(sugg.originalValue)}
                "
              </div>
            </div>

            {/* Improved Value */}
            <div className="bg-green-950/30 border border-green-700 rounded p-3 mb-3">
              <div className="text-xs font-semibold text-green-400 mb-1">
                {needsInput
                  ? "Suggested Format / Your Input Required:"
                  : "Improved:"}
              </div>
              {needsInput ? (
                <div className="space-y-2">
                  <div className="text-xs text-gray-300 italic mb-2 p-2 bg-black/40 rounded border border-gray-600">
                    <strong>Example format:</strong>{" "}
                    {typeof sugg.improvedValue === "string"
                      ? sugg.improvedValue
                      : JSON.stringify(sugg.improvedValue)}
                  </div>
                  <textarea
                    value={userInputValues[sugg.suggestionId] || ""}
                    onChange={(e) =>
                      setUserInputValues((prev) => ({
                        ...prev,
                        [sugg.suggestionId]: e.target.value,
                      }))
                    }
                    placeholder="Enter your information here (e.g., October 2024, https://github.com/username, etc.)"
                    className="w-full p-3 border-2 border-green-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent font-mono resize-y min-h-[80px] bg-gray-900 text-white placeholder-gray-500"
                    rows={3}
                    disabled={isApplied}
                  />
                  <div className="text-xs text-gray-400">
                    💡 Tip: You can resize this box by dragging the bottom-right
                    corner
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-200 font-medium break-words">
                  "
                  {typeof sugg.improvedValue === "string"
                    ? sugg.improvedValue
                    : JSON.stringify(sugg.improvedValue)}
                  "
                </div>
              )}
            </div>

            {/* Explanation */}
            {sugg.explanation && (
              <div className="text-xs text-gray-400 italic border-l-2 border-gray-600 pl-3 mb-3">
                {typeof sugg.explanation === "string"
                  ? sugg.explanation
                  : JSON.stringify(sugg.explanation)}
              </div>
            )}

            {/* Apply Button */}
            <div className="flex justify-end">
              {needsInput ? (
                <div className="flex flex-col items-end gap-2 w-full">
                  <div className="backdrop-blur-xl bg-blue-950/30 border border-blue-700 rounded-lg px-3 py-2 text-xs text-blue-300 w-full">
                    <strong>💡 Action Required:</strong> Please fill in the text
                    area above with the correct information, then click Apply.
                  </div>
                  <button
                    onClick={() => handleApply(sugg)}
                    disabled={
                      isApplied ||
                      isApplying ||
                      !userInputValues[sugg.suggestionId]?.trim()
                    }
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
                      isApplied
                        ? "bg-green-600 text-white cursor-default"
                        : isApplying
                        ? "bg-blue-500 text-white cursor-wait"
                        : userInputValues[sugg.suggestionId]?.trim()
                        ? "bg-blue-600 text-white hover:bg-blue-700 active:scale-95"
                        : "bg-gray-700 text-gray-500 cursor-not-allowed"
                    }`}
                  >
                    {isApplying ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                        Applying...
                      </>
                    ) : isApplied ? (
                      <>
                        <Check className="w-4 h-4" />
                        Applied
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-4 h-4" />
                        Apply with My Input
                      </>
                    )}
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => handleApply(sugg)}
                  disabled={isApplied || isApplying}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
                    isApplied
                      ? "bg-green-600 text-white cursor-default"
                      : isApplying
                      ? "bg-blue-500 text-white cursor-wait"
                      : "bg-blue-600 text-white hover:bg-blue-700 active:scale-95"
                  }`}
                >
                  {isApplying ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                      Applying...
                    </>
                  ) : isApplied ? (
                    <>
                      <Check className="w-4 h-4" />
                      Applied
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      Apply Suggestion
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        );
      })}
      {/* Export PDF and LaTeX Code buttons placed at the bottom of Apply Changes panel */}
      <div className="mt-6 flex gap-3 justify-end">
        <button
          onClick={async () => {
            setGettingLatex(true);
            try {
              // Use cached code if available, otherwise fetch from API
              let latexCode = cachedLatexCode;
              if (!latexCode) {
                latexCode = await getLatexCode(structuredCV || {});
                setCachedLatexCode(latexCode);
              }
              // Pass cached code to avoid redundant API call
              await downloadLatexCode(structuredCV || {}, latexCode);
            } finally {
              setGettingLatex(false);
            }
          }}
          disabled={!structuredCV || gettingLatex || exportingPdf}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
            structuredCV && !gettingLatex && !exportingPdf
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "bg-gray-700 text-gray-500 cursor-not-allowed"
          }`}
        >
          {gettingLatex ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
              Getting LaTeX...
            </>
          ) : (
            <>
              <Code2 className="w-4 h-4" />
              Get LaTeX Code
            </>
          )}
        </button>

        <button
          onClick={async () => {
            setExportingPdf(true);
            try {
              // First ensure we have LaTeX code cached
              let latexCode = cachedLatexCode;
              if (!latexCode) {
                // Only fetch if not cached
                latexCode = await getLatexCode(structuredCV || {});
                setCachedLatexCode(latexCode);
              }
              // PDF export will call backend to compile (not Gemini)
              await exportStructuredCVToPdf(structuredCV || {});
            } finally {
              setExportingPdf(false);
            }
          }}
          disabled={!structuredCV || exportingPdf || gettingLatex}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition ${
            structuredCV && !exportingPdf && !gettingLatex
              ? "bg-green-600 text-white hover:bg-green-700"
              : "bg-gray-700 text-gray-500 cursor-not-allowed"
          }`}
        >
          {exportingPdf ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
              Generating PDF...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Export PDF
            </>
          )}
        </button>
      </div>
    </div>
  );
}
