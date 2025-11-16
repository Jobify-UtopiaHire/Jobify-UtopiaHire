import React, { createContext, useState, useEffect, ReactNode } from "react";
import clsx from "clsx";
import API from "./api";
import { User, AuthModalState, AuthResponse } from "./types";
import { toast } from "sonner";
interface AuthContextType {
  user: User | null;
  loading: boolean;
  token: string | null;
  login: (email: string, password: string) => Promise<AuthResponse>;
  register: (
    name: string,
    email: string,
    password: string
  ) => Promise<AuthResponse>;
  logout: () => void;
  authModal: AuthModalState;
  openAuthModal: (view?: "login" | "register") => void;
  closeAuthModal: () => void;
  oauthSignIn: (provider: string) => Promise<void>;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
}

export const AuthContext = createContext<AuthContextType | undefined>(
  undefined
);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [authModal, setAuthModal] = useState<AuthModalState>({
    open: false,
    view: "login",
  });
  const [token, setToken] = useState<string | null>(localStorage.getItem("token"));

  // Expose setUser for OAuth callback handler
  (AuthProvider as any).setUser = setUser;

  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    setToken(storedToken);
    if (!storedToken) {
      setLoading(false);
      return;
    }
    API.get("/auth/me")
      .then((res) => {
        setUser(res.data.user);
      })
      .catch(() => {
        localStorage.removeItem("token");
        localStorage.removeItem("refreshToken");
        setUser(null);
        setToken(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = async (
    email: string,
    password: string
  ): Promise<AuthResponse> => {
    try {
      const res = await API.post("/auth/login", { email, password });
      localStorage.setItem("token", res.data.token);
      setToken(res.data.token);
      if (res.data.refreshToken)
        localStorage.setItem("refreshToken", res.data.refreshToken);
      setUser(res.data.user);
      toast.success(
        `Welcome back${
          res.data.user?.name ? `, ${res.data.user.name.split(" ")[0]}` : ""
        }!`
      );
      return res.data;
    } catch (err: any) {
      // Check if it's an email verification error
      if (err?.response?.status === 403) {
        const errorMsg =
          err?.response?.data?.detail ||
          "Please verify your email before logging in.";
        toast.error(errorMsg, { duration: 5000 });
      }
      throw err;
    }
  };

  const register = async (
    name: string,
    email: string,
    password: string
  ): Promise<AuthResponse> => {
    const res = await API.post("/auth/register", { name, email, password });
    // Registration now requires email verification - don't set user or tokens yet
    toast.success(
      res.data.message ||
        "Registration successful! Please check your email to verify your account."
    );
    return res.data;
  };

  const logout = () => {
  localStorage.removeItem("token");
  localStorage.removeItem("refreshToken");
  setUser(null);
  setToken(null);
  toast.success("Logged out successfully");
  };

  const openAuthModal = (view: "login" | "register" = "login") =>
    setAuthModal({ open: true, view });
  const closeAuthModal = () => setAuthModal({ open: false });

  // OAuth: redirect to provider URLs if configured; otherwise inform the user.
  const oauthSignIn = async (provider: string): Promise<void> => {
    const urls: Record<string, string | undefined> = {
      Google: (import.meta as any).env?.VITE_OAUTH_GOOGLE_URL,
      GitHub: (import.meta as any).env?.VITE_OAUTH_GITHUB_URL,
    };
    const url = urls[provider];
    if (url) {
      // Close modal and redirect for real OAuth handled by backend
      closeAuthModal();
      window.location.href = url;
      toast.info(`Redirecting to ${provider}...`);
      return;
    }
    toast.error(`${provider} sign-in is not configured yet`);
    throw new Error("OAuth not configured");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        token,
        login,
        register,
        logout,
        authModal,
        openAuthModal,
        closeAuthModal,
        oauthSignIn,
        setUser,
        setToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
