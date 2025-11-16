import { useState, useEffect, useContext } from "react";
import {
  Upload,
  ArrowLeft,
  FileText,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { AuthContext } from "@/AuthContext";
import { useSharedCV } from "../hooks/useSharedCV";

interface UploadViewProps {
  onUploadComplete: (
    cvFilename: string,
    jobDescription: string,
    candidateEmail: string,
    candidateName: string,
    jobTitle: string
  ) => void;
  onBack: () => void;
  loading: boolean;
}

const TOKEN_SERVER_URL = "http://localhost:8000";

export default function UploadView({
  onUploadComplete,
  onBack,
  loading,
}: UploadViewProps) {
  const { t } = useTranslation();
  const { saveCV: saveCVToShared } = useSharedCV();
  const [fileName, setFileName] = useState("");
  const [uploaded, setUploaded] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [candidateEmail, setCandidateEmail] = useState("");
  const [candidateName, setCandidateName] = useState("");
  const [jobTitle, setJobTitle] = useState("");

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Validate file type
      const allowedTypes = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      ];
      if (!allowedTypes.includes(selectedFile.type)) {
        toast.error(
          t("upload.invalidFileType") ||
            "Invalid file type. Please upload a PDF or Word document."
        );
        return;
      }

      // Validate file size (max 10MB)
      if (selectedFile.size > 10 * 1024 * 1024) {
        toast.error(
          t("upload.fileTooLarge") || "File is too large. Maximum size is 10MB."
        );
        return;
      }

      setFile(selectedFile);
      setFileName(selectedFile.name);
      setUploading(true);

      try {
        const formData = new FormData();
        formData.append("cv", selectedFile);

        const token = localStorage.getItem("token");
        const headers: HeadersInit = {};
        if (token) {
          headers["Authorization"] = `Bearer ${token}`;
        }

        const response = await fetch(`${TOKEN_SERVER_URL}/api/upload-cv`, {
          method: "POST",
          headers: headers,
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Upload failed");
        }

        const data = await response.json();
        console.log("File uploaded:", data);

        // Save to shared CV storage with server filename
        saveCVToShared(selectedFile, data.filename);

        setUploaded(true);
        toast.success(t("upload.uploadSuccess"));
      } catch (error) {
        console.error("Error uploading file:", error);
        toast.error(
          t("upload.uploadError") || "Failed to upload file. Please try again."
        );
        setUploaded(false);
        setFileName("");
        setFile(null);
      } finally {
        setUploading(false);
      }
    }
  };

  // Prefill candidate email from logged-in user (if available) but allow edits
  const auth = useContext(AuthContext);
  useEffect(() => {
    const email = auth?.user?.email;
    if (email && !candidateEmail) {
      setCandidateEmail(email);
    }
    // only when auth user email changes
  }, [auth?.user?.email]);

  const handleContinue = () => {
    if (!jobDescription.trim()) {
      toast.error("Please enter a job description");
      return;
    }

    if (!candidateEmail.trim()) {
      toast.error("Please enter your email address");
      return;
    }

    if (!candidateName.trim()) {
      toast.error("Please enter your name");
      return;
    }

    if (!jobTitle.trim()) {
      toast.error("Please enter the job title");
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(candidateEmail)) {
      toast.error("Please enter a valid email address");
      return;
    }

    if (uploaded && file && fileName) {
      onUploadComplete(
        fileName,
        jobDescription,
        candidateEmail,
        candidateName,
        jobTitle
      );
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-8 relative">
      {/* Back button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={onBack}
        disabled={uploading || loading}
        className="absolute top-8 left-8 bg-white/20 hover:bg-white/30 backdrop-blur-md rounded-full w-12 h-12"
      >
        <ArrowLeft className="w-6 h-6" />
      </Button>

      <Card className="w-full max-w-2xl shadow-2xl">
        <CardHeader className="text-center pb-4">
          <div className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
            <FileText className="w-10 h-10" />
          </div>
          <CardTitle className="text-3xl font-bold">
            {t("upload.title")}
          </CardTitle>
          <CardDescription className="text-base mt-2">
            {t("upload.subtitle")}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Upload Area */}
          <div
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
              uploaded
                ? "border-green-500 bg-green-50 dark:bg-green-950/20"
                : "border-gray-300 bg-gray-50 dark:bg-gray-900/20"
            }`}
          >
            <Input
              id="cv-upload"
              type="file"
              accept=".pdf,.doc,.docx"
              onChange={handleFileChange}
              disabled={uploading || loading}
              className="hidden"
            />

            {!uploaded ? (
              <div className="space-y-4">
                <Upload
                  className={`w-12 h-12 mx-auto ${
                    uploading ? "text-blue-400 animate-bounce" : "text-gray-400"
                  }`}
                />
                <div>
                  <p className="text-base font-medium text-gray-700 dark:text-gray-300 mb-2">
                    {uploading
                      ? t("upload.uploading") || "Uploading..."
                      : t("upload.dragDrop")}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {t("upload.fileTypes")}
                  </p>
                </div>
                <div className="flex items-center justify-center">
                  <Label htmlFor="cv-upload">
                    <Button
                      type="button"
                      disabled={uploading || loading}
                      onClick={() =>
                        document.getElementById("cv-upload")?.click()
                      }
                      className="hover:bg-white center"
                    >
                      {t("upload.browseFiles")}
                    </Button>
                  </Label>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto" />
                <div>
                  <p className="text-base font-semibold text-green-600 dark:text-green-400 mb-2">
                    {t("upload.uploadSuccess")}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {fileName}
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setUploaded(false);
                      setFileName("");
                      setFile(null);
                    }}
                    disabled={loading}
                    className="border-purple-600 hover:bg-purple-600 hover:text-white"
                  >
                    {t("upload.uploadDifferent")}
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Note */}
          <div className="bg-amber-50 dark:bg-amber-950/20 border border-amber-300 dark:border-amber-700 rounded-lg p-4">
            <div className="flex gap-3">
              <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-500 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-amber-800 dark:text-amber-200">
                <strong>Note:</strong> {t("upload.note")}
              </p>
            </div>
          </div>

          {/* Job Description Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Job Description
            </label>
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job description for the role you're interviewing for..."
              disabled={loading || uploading}
              className="w-full h-32 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed resize-none"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Paste the job description to help personalize your interview
              experience
            </p>
          </div>

          {/* Job Title Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Job Title
            </label>
            <Input
              type="text"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              placeholder="e.g., Senior Frontend Developer"
              disabled={loading || uploading}
              className="w-full"
            />
          </div>

          {/* Candidate Name Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Your Full Name
            </label>
            <Input
              type="text"
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              placeholder="e.g., John Doe"
              disabled={loading || uploading}
              className="w-full"
            />
          </div>

          {/* Candidate Email Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Your Email Address
            </label>
            <Input
              type="email"
              value={candidateEmail}
              onChange={(e) => setCandidateEmail(e.target.value)}
              placeholder="e.g., john.doe@example.com"
              disabled={loading || uploading}
              className="w-full"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Your interview report will be sent to this email address
            </p>
          </div>

          {/* Continue Button */}
          <Button
            onClick={handleContinue}
            disabled={
              !uploaded ||
              loading ||
              uploading ||
              !jobDescription.trim() ||
              !candidateEmail.trim() ||
              !candidateName.trim() ||
              !jobTitle.trim()
            }
            className="w-full h-12 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white hover:text-gray-900 transition-all"
          >
            {loading ? t("upload.starting") : t("upload.continue")}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
