/**
 * FileUpload Component
 * Handles CV file upload with drag-and-drop support
 */
import React, { useState, useEffect } from "react";
import { Upload, CheckCircle, X } from "lucide-react";

interface FileUploadProps {
  file: File | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  onUploadingChange?: (uploading: boolean) => void;
}

export default function FileUpload({
  file,
  onFileSelect,
  onFileRemove,
  onUploadingChange,
}: FileUploadProps) {
  const [uploading, setUploading] = useState(false);

  // Simulate upload completion after file is selected
  useEffect(() => {
    if (file && uploading) {
      const timer = setTimeout(() => {
        setUploading(false);
        onUploadingChange?.(false);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [file, uploading, onUploadingChange]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setUploading(true);
      onUploadingChange?.(true);
      onFileSelect(selectedFile);
    }
  };

  return (
    <div>
      <label
        className={`flex flex-col items-center justify-center w-full h-48 border-3 border-dashed rounded-xl cursor-pointer transition-all group ${
          file && !uploading
            ? "border-green-500 bg-green-50 dark:bg-green-950/20"
            : "border-gray-300 hover:backdrop-blur-xl hover:border-blue-400"
        }`}
      >
        <div className="flex flex-col items-center justify-center pt-7">
          {!uploading ? (
            <>
              <Upload className="w-12 h-12 mb-4 text-gray-400 group-hover:text-blue-500 transition" />
              <p className="mb-2 text-base font-semibold text-gray-700">
                Drop your CV here or click to upload
              </p>
              <p className="text-sm text-gray-500">
                PDF, DOCX, DOC, TXT, ODT, LaTeX, HTML, RTF
              </p>
              <p className="text-xs text-gray-400 mt-2">MAX. 10MB</p>
            </>
          ) : (
            <>
              <Upload className="w-12 h-12 mb-4 text-blue-400 animate-bounce" />
              <p className="mb-2 text-base font-semibold text-blue-700">
                Uploading...
              </p>
              <p className="text-sm text-gray-500">Processing your CV</p>
            </>
          )}
        </div>
        <input
          type="file"
          className="hidden"
          accept=".pdf,.docx,.doc,.txt,.odt,.tex,.html,.rtf"
          onChange={handleFileChange}
          disabled={uploading}
        />
      </label>

      {file && !uploading && (
        <div className="mt-4 p-4 bg-green-50 dark:bg-green-950/20 border-2 border-green-300 dark:border-green-700 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
            <div>
              <div className="font-semibold text-green-900 dark:text-green-100">
                {file.name}
              </div>
              <div className="text-sm text-green-700 dark:text-green-300">
                {(file.size / 1024).toFixed(1)} KB
              </div>
            </div>
          </div>
          <button
            onClick={onFileRemove}
            className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
            aria-label="Remove file"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
}
