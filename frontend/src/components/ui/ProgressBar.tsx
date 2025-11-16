/**
 * ProgressBar Component
 * Simple progress bar with percentage completion
 */
import { useState, useEffect } from "react";

interface ProgressBarProps {
  isActive: boolean;
}

export default function ProgressBar({ isActive }: ProgressBarProps) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setProgress(0);
      return;
    }

    // Simulate progress from 0 to 90% quickly, then slowly to 100%
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev < 90) {
          return prev + Math.random() * 30;
        } else if (prev < 99) {
          return prev + Math.random() * 2;
        }
        return 99;
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isActive]);

  // Complete the bar when loading finishes
  useEffect(() => {
    if (!isActive && progress > 0) {
      setProgress(100);
      const timer = setTimeout(() => setProgress(0), 500);
      return () => clearTimeout(timer);
    }
  }, [isActive, progress]);

  return (
    <div className="w-full space-y-2">
      <div className="w-full bg-gray-300 dark:bg-gray-600 rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full transition-all duration-300 ease-out"
          style={{
            width: `${Math.min(progress, 100)}%`,
          }}
        />
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
        {Math.round(Math.min(progress, 100))}% complete
      </p>
    </div>
  );
}
