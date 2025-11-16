import React, { useContext, useState, useEffect, FormEvent } from "react";
import { AuthContext } from "../AuthContext";
import { FaGoogle, FaGithub, FaTimes } from "react-icons/fa";
import { Loader2 } from "lucide-react";

export default function AuthModal() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("AuthModal must be used within AuthProvider");
  }

  const {
    authModal,
    closeAuthModal,
    openAuthModal,
    login,
    register,
    oauthSignIn,
  } = context;
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (authModal.open) {
      setTimeout(() => setIsVisible(true), 10);
      // Lock background scroll while modal is open
      const prev = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = prev;
      };
    } else {
      setIsVisible(false);
    }
  }, [authModal.open]);

  if (!authModal.open) return null;

  const isRegister = authModal.view === "register";

  const checks = {
    lower: /[a-z]/.test(password),
    upper: /[A-Z]/.test(password),
    digit: /\d/.test(password),
    symbol: /[^A-Za-z0-9]/.test(password),
    length: password.length >= 8,
  };
  const isStrong = Object.values(checks).every(Boolean);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (isRegister) {
        if (!isStrong) {
          setError(
            "Password must be at least 8 characters and include lowercase (a-z), uppercase (A-Z), a number (0-9), and a symbol."
          );
          setLoading(false);
          return;
        }
        await register(name, email, password);
      } else {
        await login(email, password);
      }
      closeAuthModal();
      // Reset form
      setEmail("");
      setPassword("");
      setName("");
    } catch (err: any) {
      console.error("Auth error:", err);
      console.error("Error response:", err?.response?.data);
      setError(
        err?.response?.data?.detail ||
          err?.response?.data?.message ||
          err?.message ||
          "Auth failed"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center transition-all duration-300 ${
        isVisible ? "opacity-100" : "opacity-0"
      }`}
      role="dialog"
      aria-modal="true"
      onClick={closeAuthModal}
    >
      {/* Backdrop with blur */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-md"></div>

      {/* Modal */}
      <div
        className={`relative w-full max-w-md mx-4 bg-white rounded-3xl shadow-2xl transform transition-all duration-300 ${
          isVisible ? "scale-100 translate-y-0" : "scale-95 translate-y-4"
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Gradient header */}
        <div className="absolute top-0 left-0 right-0 h-2 bg-gradient-to-r from-primary-600 via-primary-700 to-primary-800 rounded-t-3xl"></div>

        <div className="p-6 sm:p-8 max-h-[85vh] overflow-y-auto">
          {/* Close button */}
          <button
            onClick={closeAuthModal}
            disabled={loading}
            className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FaTimes />
          </button>

          {/* Title */}
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-2">
              {isRegister ? "Create your account" : "Welcome back"}
            </h3>
            <p className="text-gray-600">
              {isRegister
                ? "Start your journey to interview success"
                : "Sign in to continue your practice"}
            </p>
          </div>

          {/* OAuth buttons */}
          <div className="space-y-3 mb-6">
            <button
              onClick={() => oauthSignIn("Google")}
              disabled={loading}
              className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white border-2 border-gray-200 rounded-xl hover:border-primary-500 hover:bg-primary-50 transition-all duration-200 group disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FaGoogle className="text-xl text-red-500" />
              <span className="font-medium text-gray-700 group-hover:text-primary-700">
                Continue with Google
              </span>
            </button>
            <button
              onClick={() => oauthSignIn("GitHub")}
              disabled={loading}
              className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white border-2 border-gray-200 rounded-xl hover:border-primary-500 hover:bg-primary-50 transition-all duration-200 group disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FaGithub className="text-xl text-gray-900" />
              <span className="font-medium text-gray-700 group-hover:text-primary-700">
                Continue with GitHub
              </span>
            </button>
          </div>

          {/* Divider */}
          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-white text-gray-500">
                Or continue with email
              </span>
            </div>
          </div>

          {/* Form */}
          <form className="space-y-4" onSubmit={submit}>
            {isRegister && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={loading}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-primary-500 focus:ring-0 outline-none transition-all text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
                  placeholder="John Doe"
                  required
                />
              </div>
            )}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-primary-500 focus:ring-0 outline-none transition-all text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="you@example.com"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-primary-500 focus:ring-0 outline-none transition-all text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="••••••••"
                required
              />
              {isRegister && (
                <ul className="mt-2 text-xs text-gray-600 space-y-1">
                  <li className={checks.length ? "text-green-600" : ""}>
                    At least 8 characters
                  </li>
                  <li className={checks.lower ? "text-green-600" : ""}>
                    Contains a lowercase letter (a-z)
                  </li>
                  <li className={checks.upper ? "text-green-600" : ""}>
                    Contains an uppercase letter (A-Z)
                  </li>
                  <li className={checks.digit ? "text-green-600" : ""}>
                    Contains a number (0-9)
                  </li>
                  <li className={checks.symbol ? "text-green-600" : ""}>
                    Contains a symbol (!@#$...)
                  </li>
                </ul>
              )}
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || (isRegister && !isStrong)}
              className={`w-full py-3 text-white font-semibold rounded-xl shadow-lg transition-all duration-200 bg-gradient-to-r from-primary-600 to-primary-700 hover:shadow-xl transform hover:scale-[1.02] flex items-center justify-center gap-2 ${
                loading || (isRegister && !isStrong)
                  ? "opacity-50 cursor-not-allowed hover:scale-100"
                  : ""
              }`}
            >
              {loading && <Loader2 className="w-5 h-5 animate-spin" />}
              {loading
                ? isRegister
                  ? "Creating Account..."
                  : "Signing In..."
                : isRegister
                ? "Create Account"
                : "Sign In"}
            </button>
          </form>

          {/* Footer */}
          <div className="mt-6 text-center text-sm text-gray-600">
            {isRegister ? (
              <span>
                Already have an account?{" "}
                <button
                  onClick={() => !loading && openAuthModal("login")}
                  disabled={loading}
                  className="text-primary-600 font-semibold hover:text-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Login
                </button>
              </span>
            ) : (
              <span>
                Don't have an account?{" "}
                <button
                  onClick={() => !loading && openAuthModal("register")}
                  disabled={loading}
                  className="text-primary-600 font-semibold hover:text-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Create one
                </button>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
