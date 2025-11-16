/**
 * LoadingSpinner Component
 * Simple loading spinner with optional message
 */

interface LoadingSpinnerProps {
  message?: string;
}

export default function LoadingSpinner({ message }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
      {message && <span>{message}</span>}
    </div>
  );
}
