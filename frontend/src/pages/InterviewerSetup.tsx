import { useNavigate } from "react-router-dom";
import { useServices } from "../ServiceContext";
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
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
const TOKEN_SERVER_URL = "http://localhost:8000";

// localStorage keys - only for interview-specific data
const STORAGE_KEYS = {
  JOB_DESCRIPTION: "interviewer_job_description",
  JOB_TITLE: "interviewer_job_title",
  CANDIDATE_NAME: "interviewer_candidate_name",
  CANDIDATE_EMAIL: "interviewer_candidate_email",
  LANGUAGE: "interviewer_language",
};

export default function InterviewerSetup() {
  const navigate = useNavigate();
  const { setCvFile, setInterviewConfig, addHistory } = useServices();
  const [loading, setLoading] = useState(false);
  const { cvFile, serverFilename, saveCV, clearSharedCV } = useSharedCV();
  const [uploading, setUploading] = useState(false);

  const handleUploadComplete = async (
    filename: string,
    jobDesc: string,
    email: string,
    name: string,
    title: string,
    language: string
  ) => {
    try {
      setLoading(true);

      // Store all the interview data in ServiceContext
      // Use server filename (from upload response) instead of original filename
      setInterviewConfig({
        jobTitle: title,
        description: jobDesc,
        company: "", // Can be extracted from job description if needed
        language: language,
        seniority: "Mid",
        experience: "",
        // Store additional fields - use server filename
        cvFilename: filename,
        candidateEmail: email,
        candidateName: name,
      } as any);

      // Record history
      addHistory({
        type: "interview",
        title: title,
        meta: {
          company: "",
          language: language,
          seniority: "Mid",
        },
      });

      // Navigate to interview session
      navigate("/interviewer/session");
      toast.info("Navigated to interview session");
    } catch (error) {
      console.error("Error processing upload:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate("/dashboard");
    toast.info("Returned to home Dashboard");
  };
  const { t } = useTranslation();
  const [jobDescription, setJobDescription] = useState("");
  const [candidateEmail, setCandidateEmail] = useState("");
  const [candidateName, setCandidateName] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("");

  // Load data from localStorage on mount
  useEffect(() => {
    const savedJobDescription = localStorage.getItem(
      STORAGE_KEYS.JOB_DESCRIPTION
    );
    const savedJobTitle = localStorage.getItem(STORAGE_KEYS.JOB_TITLE);
    const savedCandidateName = localStorage.getItem(
      STORAGE_KEYS.CANDIDATE_NAME
    );
    const savedCandidateEmail = localStorage.getItem(
      STORAGE_KEYS.CANDIDATE_EMAIL
    );
    const savedLanguage = localStorage.getItem(STORAGE_KEYS.LANGUAGE);

    if (savedJobDescription) setJobDescription(savedJobDescription);
    if (savedJobTitle) setJobTitle(savedJobTitle);
    if (savedCandidateName) setCandidateName(savedCandidateName);
    if (savedCandidateEmail) setCandidateEmail(savedCandidateEmail);
    if (savedLanguage) setSelectedLanguage(savedLanguage);
  }, []);

  // Save text inputs to localStorage whenever they change
  useEffect(() => {
    if (jobDescription) {
      localStorage.setItem(STORAGE_KEYS.JOB_DESCRIPTION, jobDescription);
    }
  }, [jobDescription]);

  useEffect(() => {
    if (jobTitle) {
      localStorage.setItem(STORAGE_KEYS.JOB_TITLE, jobTitle);
    }
  }, [jobTitle]);

  useEffect(() => {
    if (candidateName) {
      localStorage.setItem(STORAGE_KEYS.CANDIDATE_NAME, candidateName);
    }
  }, [candidateName]);

  useEffect(() => {
    if (candidateEmail) {
      localStorage.setItem(STORAGE_KEYS.CANDIDATE_EMAIL, candidateEmail);
    }
  }, [candidateEmail]);

  useEffect(() => {
    if (selectedLanguage) {
      localStorage.setItem(STORAGE_KEYS.LANGUAGE, selectedLanguage);
    }
  }, [selectedLanguage]);

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
        saveCV(selectedFile, data.filename);
        toast.success(t("upload.uploadSuccess"));
      } catch (error) {
        console.error("Error uploading file:", error);
        toast.error(
          t("upload.uploadError") || "Failed to upload file. Please try again."
        );
      } finally {
        setUploading(false);
      }
    }
  };

  // Prefill candidate email from logged-in user and prefill name from account.
  // Email is taken from the account and made read-only when available. Name is
  // filled from the account but stays editable so the user can change it.
  const auth = useContext(AuthContext);
  useEffect(() => {
    const email = auth?.user?.email;
    if (email) {
      // Always prefer account email when available (overrides any saved value)
      setCandidateEmail(email);
    }
  }, [auth?.user?.email]);

  useEffect(() => {
    const name = auth?.user?.name;
    // Only prefill name if there isn't already a candidateName (allow user edits)
    if (name && !candidateName) {
      setCandidateName(name);
    }
  }, [auth?.user?.name, candidateName]);

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

    if (!selectedLanguage) {
      toast.error("Please select a language");
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(candidateEmail)) {
      toast.error("Please enter a valid email address");
      return;
    }

    if (cvFile) {
      // Use server filename instead of original filename
      handleUploadComplete(
        serverFilename || cvFile.name,
        jobDescription,
        candidateEmail,
        candidateName,
        jobTitle,
        selectedLanguage
      );
    }
  };
  return (
    <div className="min-h-screen flex items-center justify-center p-8 relative">
      {/* Back button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={handleBack}
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
              cvFile
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

            {!cvFile ? (
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
                    {cvFile?.name}
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => {
                      clearSharedCV();
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
            {/* Make the input read-only when an account email is available. If
                there's no account email (not logged in) it stays editable. */}
            <Input
              type="email"
              value={candidateEmail}
              onChange={(e) => setCandidateEmail(e.target.value)}
              placeholder="e.g., john.doe@example.com"
              disabled={loading || uploading}
              readOnly={!!auth?.user?.email}
              className={`w-full ${
                auth?.user?.email
                  ? "bg-gray-100 dark:bg-gray-800 cursor-not-allowed"
                  : ""
              }`}
            />
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Your interview report will be sent to this email address
            </p>
          </div>

          {/* Language Selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Interview Language
            </label>
            <Select
              value={selectedLanguage}
              onValueChange={setSelectedLanguage}
              disabled={loading || uploading}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a language" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value="English">English</SelectItem>
                  <SelectItem value="French">French</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              The interview will be conducted in this language
            </p>
          </div>

          {/* Continue Button */}
          <div className="has-[button:disabled]:cursor-not-allowed">
            <Button
              onClick={handleContinue}
              disabled={
                !cvFile ||
                loading ||
                uploading ||
                !jobDescription.trim() ||
                !candidateEmail.trim() ||
                !candidateName.trim() ||
                !jobTitle.trim() ||
                !selectedLanguage
              }
              className="w-full h-12 text-lg font-bold disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white hover:text-gray-900 transition-all"
            >
              {loading ? t("upload.starting") : t("upload.continue")}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
