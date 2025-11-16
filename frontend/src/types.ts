export interface User {
  id: number
  name: string
  email: string
}

export interface AuthResponse {
  user: User
  token: string
  refreshToken?: string
}

export interface AuthModalState {
  open: boolean
  view?: 'login' | 'register'
}

export interface InterviewConfig {
  jobTitle: string
  language: string
  company: string
  experience: string
  seniority: string
  description: string
}

export interface HistoryEntry {
  id: number
  ts: string
  [key: string]: any
}

// Interview history types
export interface InterviewItem {
  id: number
  job_title: string
  interview_score: number | null
  conclusion: string
  created_at: string | null
}

export interface UserInterviewsResponse {
  success: boolean
  user_id: number
  user_name: string
  user_email: string
  total_interviews: number
  interviews: InterviewItem[]
}

// Skills Gap Analysis types
export interface GapItem {
  skill: string
  proficiency_you: number
  proficiency_req: number
  gap: number
  is_must_have: boolean
}

export interface AIScores {
  coverage: number
  depth: number
  alignment: number
}

export interface QuantitativeSummary {
  overall_score: number
  skills_matched: number
  skills_missing: number
  total_required: number
}

export interface PriorityAction {
  action: string
  time_estimate: string
  why: string
}

export interface LearningPath {
  skill: string
  path_title: string
  platform: string
}

export interface ResumeEdit {
  before: string
  after: string
  reason: string
}

export interface FullAnalysisResponse {
  job_skill_profile: GapItem[]
  ai_summary: string
  ai_scores: AIScores
  quantitative_summary: QuantitativeSummary
  priority_actions: PriorityAction[]
  learning_paths: LearningPath[]
  resume_edits: ResumeEdit[]
}
export interface CVFile {
  name: string;
  size: number;
  type: string;
}

export interface ContactInfo {
  name?: string;
  email?: string;
  phone?: string;
  location?: string;
  linkedin?: string;
  github?: string;
  website?: string;
}

export interface Experience {
  title?: string;
  company?: string;
  location?: string;
  startDate?: string;
  endDate?: string;
  description?: string;
  bullets?: string[];
}

export interface Education {
  degree?: string;
  institution?: string;
  location?: string;
  startDate?: string;
  endDate?: string;
  description?: string;
}

export interface Project {
  name?: string;
  description?: string;
  technologies?: string[];
  url?: string;
}

export interface Certification {
  name?: string;
  issuer?: string;
  date?: string;
  credential?: string;
}

export interface Activity {
  organization?: string;
  title?: string;
  startDate?: string;
  endDate?: string;
  description?: string;
}

export interface Skills {
  [category: string]: string[];
}

export interface StructuredCV {
  contact?: ContactInfo;
  summary?: string;
  experience?: Experience[];
  education?: Education[];
  projects?: Project[];
  skills?: Skills;
  certifications?: Certification[];
  activities?: Activity[];
  other_sections?: {
    [key: string]: any[];
  };
  original_text?: string;
}

export interface FieldSuggestion {
  suggestionId: number;
  targetField: string;
  fieldPath: (string | number)[];
  fieldId?: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  issue_type: string;
  problem: string;
  originalValue: string;
  improvedValue: string;
  explanation: string;
}

export interface SectionAnalysis {
  section: string;
  status: string;
  issues: string[];
  suggestions: string[];
}

export interface JobMatch {
  relevance_score: number;
  keyword_matches?: string[];
  missing_keywords?: string[];
  recommendations?: string[];
}

export interface TopPriority {
  priority: number;
  action: string;
  impact: 'High' | 'Medium' | 'Low';
  time_estimate: string;
  category: string;
}

export interface Analysis {
  overall_score: number;
  ats_score: number;
  field_suggestions?: FieldSuggestion[];
  quick_wins?: string[];
  section_analysis?: SectionAnalysis[];
  job_match_analysis?: JobMatch;
  top_priorities?: TopPriority[];
  metadata?: {
    filename?: string;
    file_size_bytes?: number;
  };
  structured?: StructuredCV;
}

export interface OriginalFile {
  filename: string;
  content_type: string;
  data: string;
  size: number;
}

export interface AnalyzeResponse {
  summary: string;
  status: string;
  structured_cv: StructuredCV;
  original_file?: OriginalFile;
  file_info: {
    filename?: string;
    file_size_bytes?: number;
    links_found?: number;
    processing_method?: string;
    source?: string;
  };
  gemini_analysis?: {
    status: string;
    analysis: Partial<Analysis>;
  };
}

export interface ApplySuggestionResponse {
  status: string;
  updated_cv: StructuredCV;
  applied_suggestion: FieldSuggestion;
}

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

