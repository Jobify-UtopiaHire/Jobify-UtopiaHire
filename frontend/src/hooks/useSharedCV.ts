import { useState, useEffect } from 'react';

// Shared storage key for CV across all tools
const SHARED_CV_STORAGE = {
  CV_FILE_NAME: 'shared_cv_file_name',
  CV_FILE_DATA: 'shared_cv_file_data',
  CV_FILE_UPLOADED: 'shared_cv_file_uploaded',
  CV_SERVER_FILENAME: 'shared_cv_server_filename', // Server-generated filename
};

export function useSharedCV() {
  const [cvFile, setCvFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState('');
  const [serverFilename, setServerFilename] = useState(''); // Server-generated filename

  // Load CV from shared localStorage on mount
  useEffect(() => {
    const savedFileName = localStorage.getItem(SHARED_CV_STORAGE.CV_FILE_NAME);
    const savedFileData = localStorage.getItem(SHARED_CV_STORAGE.CV_FILE_DATA);
    const savedFileUploaded = localStorage.getItem(SHARED_CV_STORAGE.CV_FILE_UPLOADED);
    const savedServerFilename = localStorage.getItem(SHARED_CV_STORAGE.CV_SERVER_FILENAME);

    if (savedFileName && savedFileData && savedFileUploaded === 'true') {
      try {
        // Convert base64 back to file
        const byteString = atob(savedFileData.split(',')[1]);
        const mimeString = savedFileData
          .split(',')[0]
          .split(':')[1]
          .split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
          ia[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([ab], { type: mimeString });
        const restoredFile = new File([blob], savedFileName, {
          type: mimeString,
        });

        setCvFile(restoredFile);
        setFileName(savedFileName);
        setServerFilename(savedServerFilename || savedFileName); // Use server filename if available
      } catch (error) {
        console.error('Error restoring CV from localStorage:', error);
        clearSharedCV();
      }
    }
  }, []);

  // Save CV file to shared localStorage
  const saveCV = (file: File, serverFilenameParam?: string) => {
    setCvFile(file);
    setFileName(file.name);
    
    // Save server filename if provided
    if (serverFilenameParam) {
      setServerFilename(serverFilenameParam);
      localStorage.setItem(SHARED_CV_STORAGE.CV_SERVER_FILENAME, serverFilenameParam);
    }

    // Save file to localStorage if it's under 5MB
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB in bytes
    if (file.size < MAX_FILE_SIZE) {
      const reader = new FileReader();
      reader.onload = () => {
        const base64Data = reader.result as string;
        localStorage.setItem(SHARED_CV_STORAGE.CV_FILE_NAME, file.name);
        localStorage.setItem(SHARED_CV_STORAGE.CV_FILE_DATA, base64Data);
        localStorage.setItem(SHARED_CV_STORAGE.CV_FILE_UPLOADED, 'true');
      };
      reader.readAsDataURL(file);
    } else {
      // File is too large for localStorage, just save the filename
      localStorage.setItem(SHARED_CV_STORAGE.CV_FILE_NAME, file.name);
      localStorage.setItem(SHARED_CV_STORAGE.CV_FILE_UPLOADED, 'true');
    }
  };

  // Clear CV from shared localStorage
  const clearSharedCV = () => {
    setCvFile(null);
    setFileName('');
    setServerFilename('');
    localStorage.removeItem(SHARED_CV_STORAGE.CV_FILE_NAME);
    localStorage.removeItem(SHARED_CV_STORAGE.CV_FILE_DATA);
    localStorage.removeItem(SHARED_CV_STORAGE.CV_FILE_UPLOADED);
    localStorage.removeItem(SHARED_CV_STORAGE.CV_SERVER_FILENAME);
  };

  return {
    cvFile,
    fileName,
    serverFilename, // Export server filename
    saveCV,
    clearSharedCV,
    setCvFile,
  };
}
