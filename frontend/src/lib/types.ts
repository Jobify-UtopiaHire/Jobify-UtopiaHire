export interface GapItem {
  skill: string;
  proficiency_you: number;
  proficiency_req: number;
  gap: number;
  is_must_have: boolean;
}

export interface AIScores {
  coverage: number;
  depth: number;
  alignment: number;
}

export interface QuantitativeSummary {
  overall_score: number;
  skills_matched: number;
  skills_missing: number;
  total_required: number;
}

export interface PriorityAction {
  action: string;
  time_estimate: string;
  why: string;
}

export interface LearningPath {
  skill: string;
  path_title: string;
  platform: string;
}

export interface ResumeEdit {
  before: string;
  after: string;
  reason: string;
}

export interface FullAnalysisResponse {
  job_skill_profile: GapItem[];
  ai_summary: string;
  ai_scores: AIScores;
  quantitative_summary: QuantitativeSummary;
  priority_actions: PriorityAction[];
  learning_paths: LearningPath[];
  resume_edits: ResumeEdit[];
}
