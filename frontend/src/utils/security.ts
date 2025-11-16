/**
 * Frontend Security Utilities
 * Provides XSS protection, input sanitization, and validation
 */

/**
 * Sanitize HTML input to prevent XSS attacks
 * Removes all HTML tags and dangerous characters
 */
export function sanitizeHTML(input: string): string {
  if (!input) return '';
  
  // Create a temporary div element
  const div = document.createElement('div');
  div.textContent = input;
  return div.innerHTML;
}

/**
 * Escape HTML special characters
 */
export function escapeHTML(input: string): string {
  if (!input) return '';
  
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;',
  };
  
  return input.replace(/[&<>"'/]/g, (char) => map[char]);
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return emailRegex.test(email);
}

/**
 * Validate password strength
 * Must be at least 8 characters with uppercase, lowercase, number, and symbol
 */
export function isStrongPassword(password: string): boolean {
  if (password.length < 8) return false;
  
  const hasLowercase = /[a-z]/.test(password);
  const hasUppercase = /[A-Z]/.test(password);
  const hasNumber = /\d/.test(password);
  const hasSymbol = /[^A-Za-z0-9]/.test(password);
  
  return hasLowercase && hasUppercase && hasNumber && hasSymbol;
}

/**
 * Get password strength level
 */
export function getPasswordStrength(password: string): {
  level: 'weak' | 'medium' | 'strong' | 'very-strong';
  score: number;
  feedback: string[];
} {
  const feedback: string[] = [];
  let score = 0;
  
  if (password.length >= 8) score += 1;
  else feedback.push('At least 8 characters required');
  
  if (password.length >= 12) score += 1;
  
  if (/[a-z]/.test(password)) score += 1;
  else feedback.push('Add lowercase letters');
  
  if (/[A-Z]/.test(password)) score += 1;
  else feedback.push('Add uppercase letters');
  
  if (/\d/.test(password)) score += 1;
  else feedback.push('Add numbers');
  
  if (/[^A-Za-z0-9]/.test(password)) score += 1;
  else feedback.push('Add special characters');
  
  let level: 'weak' | 'medium' | 'strong' | 'very-strong' = 'weak';
  if (score >= 6) level = 'very-strong';
  else if (score >= 5) level = 'strong';
  else if (score >= 3) level = 'medium';
  
  return { level, score, feedback };
}

/**
 * Validate file upload
 */
export function validateFileUpload(file: File | null): {
  valid: boolean;
  error?: string;
} {
  if (!file) {
    return { valid: false, error: 'No file provided' };
  }
  
  // Check file size (10MB max)
  const maxSize = 10 * 1024 * 1024;
  if (file.size > maxSize) {
    return { valid: false, error: 'File size exceeds 10MB limit' };
  }
  
  // Check file extension
  const allowedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.odt', '.tex', '.html', '.rtf'];
  const fileName = file.name.toLowerCase();
  const hasValidExtension = allowedExtensions.some(ext => fileName.endsWith(ext));
  
  if (!hasValidExtension) {
    return { 
      valid: false, 
      error: `Invalid file type. Allowed: ${allowedExtensions.join(', ')}` 
    };
  }
  
  // Check for dangerous filenames
  if (fileName.includes('..') || fileName.includes('/') || fileName.includes('\\')) {
    return { valid: false, error: 'Invalid filename' };
  }
  
  return { valid: true };
}

/**
 * Sanitize filename for safe upload
 */
export function sanitizeFilename(filename: string): string {
  // Remove directory traversal attempts
  let safe = filename.replace(/\.\./g, '').replace(/[\/\\]/g, '');
  
  // Keep only alphanumeric, dots, hyphens, underscores
  safe = safe.replace(/[^a-zA-Z0-9._-]/g, '_');
  
  return safe || 'file';
}

/**
 * Check if input contains potential SQL injection patterns
 * Note: This is client-side validation only. Server MUST use parameterized queries.
 */
export function detectSQLInjection(input: string): boolean {
  const dangerousPatterns = [
    /(\bSELECT\b.*\bFROM\b)/i,
    /(\bINSERT\b.*\bINTO\b)/i,
    /(\bUPDATE\b.*\bSET\b)/i,
    /(\bDELETE\b.*\bFROM\b)/i,
    /(\bDROP\b.*\bTABLE\b)/i,
    /(--|\#|\/\*|\*\/)/,
    /(\bOR\b.*=.*)/i,
    /(\bAND\b.*=.*)/i,
    /(\bUNION\b.*\bSELECT\b)/i,
    /(\bEXEC\b|\bEXECUTE\b)/i,
  ];
  
  return dangerousPatterns.some(pattern => pattern.test(input));
}

/**
 * Sanitize user input for display
 */
export function sanitizeInput(input: string): string {
  if (!input) return '';
  
  // Check for SQL injection
  if (detectSQLInjection(input)) {
    console.warn('Potential SQL injection detected in input');
  }
  
  // Escape HTML
  return escapeHTML(input.trim());
}

/**
 * Secure localStorage wrapper with encryption awareness
 */
export const secureStorage = {
  setItem(key: string, value: string): void {
    try {
      // Note: Consider encrypting sensitive data before storing
      localStorage.setItem(key, value);
    } catch (error) {
      console.error('Error storing data:', error);
    }
  },
  
  getItem(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch (error) {
      console.error('Error retrieving data:', error);
      return null;
    }
  },
  
  removeItem(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing data:', error);
    }
  },
  
  clear(): void {
    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing storage:', error);
    }
  }
};

/**
 * Generate CSRF token (simple implementation)
 * In production, this should come from the server
 */
export function generateCSRFToken(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(32)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Validate URL to prevent open redirects
 */
export function isValidRedirectURL(url: string, allowedDomains: string[]): boolean {
  try {
    const parsed = new URL(url);
    return allowedDomains.some(domain => parsed.hostname === domain || parsed.hostname.endsWith(`.${domain}`));
  } catch {
    // If it's a relative URL, it's safe
    return url.startsWith('/') && !url.startsWith('//');
  }
}

/**
 * Rate limiting helper (client-side)
 */
export class RateLimiter {
  private attempts: number[] = [];
  
  constructor(
    private maxAttempts: number,
    private windowMs: number
  ) {}
  
  canAttempt(): boolean {
    const now = Date.now();
    const cutoff = now - this.windowMs;
    
    // Remove old attempts
    this.attempts = this.attempts.filter(time => time > cutoff);
    
    return this.attempts.length < this.maxAttempts;
  }
  
  recordAttempt(): void {
    this.attempts.push(Date.now());
  }
  
  reset(): void {
    this.attempts = [];
  }
}

/**
 * Debounce function for input validation
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  
  return function(...args: Parameters<T>) {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}
