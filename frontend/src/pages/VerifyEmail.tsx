import { useEffect, useState, useContext, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { toast } from "sonner";
import API from "@/api";
import { AuthContext } from "@/AuthContext";

export default function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const auth = useContext(AuthContext);
  const [status, setStatus] = useState<"loading" | "success" | "error">(
    "loading"
  );
  const [message, setMessage] = useState("Verifying your email...");
  const hasVerified = useRef(false);

  useEffect(() => {
    // Prevent running multiple times
    if (hasVerified.current) return;
    hasVerified.current = true;

    const verifyEmail = async () => {
      const token = searchParams.get("token");

      if (!token) {
        setStatus("error");
        setMessage("Invalid verification link. Please check your email.");
        toast.error("Invalid verification link");
        return;
      }

      try {
        // Verify email
        const res = await API.post("/auth/verify-email", { token });

        // Store tokens
        localStorage.setItem("token", res.data.token);
        localStorage.setItem("refreshToken", res.data.refreshToken);

        // Update auth context with user data
        if (auth && (auth as any).setUser) {
          (auth as any).setUser(res.data.user);
        }

        // Set success state
        setStatus("success");
        setMessage("Your email has been verified successfully!");
        toast.success("Email verified! Redirecting to dashboard...");

        // Redirect to dashboard after 1.5 seconds
        setTimeout(() => {
          navigate("/dashboard", { replace: true });
        }, 1500);
      } catch (error: any) {
        console.error("Verification error:", error);
        setStatus("error");
        const errorMessage =
          error.response?.data?.detail ||
          error.message ||
          "Failed to verify email. The link may have expired.";
        setMessage(errorMessage);
        toast.error(errorMessage);
      }
    };

    verifyEmail();
  }, [searchParams, navigate, auth]);

  return (
    <div className="min-h-screen flex items-center justify-center p-8">
      <Card className="w-full max-w-md shadow-2xl">
        <CardHeader className="text-center pb-4">
          <div className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
            {status === "loading" && (
              <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
            )}
            {status === "success" && (
              <CheckCircle className="w-10 h-10 text-green-500" />
            )}
            {status === "error" && (
              <AlertCircle className="w-10 h-10 text-red-500" />
            )}
          </div>
          <CardTitle className="text-3xl font-bold">
            {status === "loading" && "Verifying Email..."}
            {status === "success" && "Email Verified!"}
            {status === "error" && "Verification Failed"}
          </CardTitle>
          <CardDescription className="text-base mt-2">
            {message}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {status === "success" && (
            <div className="bg-green-50 dark:bg-green-950/20 border border-green-300 dark:border-green-700 rounded-lg p-4">
              <p className="text-sm text-green-800 dark:text-green-200">
                You will be redirected to your dashboard shortly...
              </p>
            </div>
          )}

          {status === "error" && (
            <div className="space-y-4">
              <div className="bg-red-50 dark:bg-red-950/20 border border-red-300 dark:border-red-700 rounded-lg p-4">
                <p className="text-sm text-red-800 dark:text-red-200">
                  If you continue to experience issues, please try registering
                  again or contact support.
                </p>
              </div>
              <Button
                onClick={() => navigate("/")}
                className="w-full"
                variant="outline"
              >
                Return to Home
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
