import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';

interface Suggestion {
  targetField?: string;
  fieldPath?: string[];
  improvedValue?: string;
  text?: string;
  originalValue?: string;
  reason?: string;
}

interface ApplyChangesProps {
  suggestions: Suggestion[];
  onApply: (suggestion: Suggestion, index: number) => void;
  onApplyAll: () => void;
  appliedChanges: Set<number>;
}

const ApplyChanges: React.FC<ApplyChangesProps> = ({ suggestions, onApply, onApplyAll, appliedChanges }) => {
  if (!suggestions || suggestions.length === 0) {
    return (
      <Card className="bg-slate-900/50 backdrop-blur-md border-slate-700 shadow-lg">
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">No suggestions available. Please analyze your CV first.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="bg-slate-900/50 backdrop-blur-md border-slate-700 shadow-lg">
        <CardHeader className="border-b border-slate-800">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-slate-100">AI Suggested Changes</CardTitle>
              <CardDescription>Review and apply changes to improve your CV</CardDescription>
            </div>
            <Button
              onClick={onApplyAll}
              className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:via-purple-500 hover:to-pink-500"
            >
              Apply All Changes
            </Button>
          </div>
        </CardHeader>
        <CardContent className="mt-6">
          <div className="space-y-4">
            {suggestions.map((suggestion, index) => {
              const isApplied = appliedChanges.has(index);
              return (
                <div
                  key={index}
                  className={`p-5 rounded-lg border ${
                    isApplied
                      ? 'border-green-500 bg-green-900/20'
                      : 'border-slate-700 bg-slate-800/60'
                  } transition-all`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-semibold text-indigo-300">
                          {suggestion?.targetField || suggestion?.fieldPath?.join(' / ') || 'Field'}
                        </span>
                        {isApplied && (
                          <span className="flex items-center gap-1 text-xs text-green-400">
                            <FaCheckCircle /> Applied
                          </span>
                        )}
                      </div>
                      {suggestion?.reason && (
                        <p className="text-xs text-slate-400 mb-2">{suggestion.reason}</p>
                      )}
                    </div>
                    {!isApplied && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => onApply(suggestion, index)}
                        className="border-indigo-500 text-indigo-300 hover:bg-indigo-500/20"
                      >
                        Apply
                      </Button>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {suggestion?.originalValue && (
                      <div className="p-3 rounded bg-red-900/20 border border-red-700/40">
                        <div className="text-xs font-semibold text-red-300 mb-1 flex items-center gap-1">
                          <FaTimesCircle /> Original
                        </div>
                        <p className="text-sm text-slate-300 whitespace-pre-wrap">
                          {suggestion.originalValue}
                        </p>
                      </div>
                    )}
                    <div className="p-3 rounded bg-green-900/20 border border-green-700/40">
                      <div className="text-xs font-semibold text-green-300 mb-1 flex items-center gap-1">
                        <FaCheckCircle /> Improved
                      </div>
                      <p className="text-sm text-slate-300 whitespace-pre-wrap">
                        {suggestion?.improvedValue || suggestion?.text}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ApplyChanges;
