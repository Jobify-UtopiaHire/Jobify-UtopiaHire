/**
 * ErrorMessage Component
 * Displays error messages with dismiss functionality
 */
import { X } from 'lucide-react';

interface ErrorMessageProps {
  message: string;
  onClose: () => void;
}

export default function ErrorMessage({ message, onClose }: ErrorMessageProps) {
  return (
    <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 flex items-start justify-between">
      <div className="flex-1">
        <p className="text-red-800 font-medium">{message}</p>
      </div>
      <button
        onClick={onClose}
        className="text-red-600 hover:text-red-800 ml-4"
        aria-label="Dismiss error"
      >
        <X className="w-5 h-5" />
      </button>
    </div>
  );
}
