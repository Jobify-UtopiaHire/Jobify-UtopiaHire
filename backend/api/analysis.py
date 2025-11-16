from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pdfplumber
import io
import re

from core.skill_extractor import extract_skills_from_text, normalize_skills
from core.data_loader import data_loader
from core.ai_analyzer import call_gemini_analyzer, call_gemini_coach

router = APIRouter()

class AnalysisRequest(BaseModel):
    cv_text: str
    job_description: str
    job_title: Optional[str] = "Target Role"

class QuantitativeSkill(BaseModel):
    original: str
    normalized: str
    source: str
    evidence: str
    match_type: str

class MarketDemandSkill(BaseModel):
    skill: str
    total_demand: int
    top_roles: List[Dict[str, Any]]
    priority: str

class QuantitativeAnalysisResponse(BaseModel):
    overall_score: float
    skills_breakdown: Dict[str, int]
    matched_skills: List[QuantitativeSkill]
    missing_skills_prioritized: List[MarketDemandSkill]
    cv_skills: List[QuantitativeSkill]
    job_skills: List[QuantitativeSkill]

class SkillProfile(BaseModel):
    skill: str
    proficiency_you: int = Field(..., description="Proficiency 1-5")
    evidence: str

class JobRequirement(BaseModel):
    skill: str
    proficiency_req: int = Field(..., description="Proficiency 1-5")
    is_must_have: bool

class OverallScores(BaseModel):
    coverage: int
    depth: int
    recency: int

class GapItem(BaseModel):
    skill: str
    proficiency_req: int
    proficiency_you: int
    gap: int = Field(..., description="proficiency_req - proficiency_you")
    is_must_have: bool
    market_demand: MarketDemandSkill

class PriorityAction(BaseModel):
    action: str
    difficulty: str
    time_estimate: str
    why: str

class LearningPath(BaseModel):
    skill: str
    path_title: str
    platform: str

class ResumeEdit(BaseModel):
    before: str
    after: str

class FullAnalysisResponse(BaseModel):
    quantitative_summary: QuantitativeAnalysisResponse
    ai_scores: OverallScores
    ai_summary: str
    cv_skill_profile: List[SkillProfile]
    job_skill_profile: List[GapItem]
    priority_actions: List[PriorityAction]
    learning_paths: List[LearningPath]
    resume_edits: List[ResumeEdit]
    low_value_skills: List[str]

# --- 2. INTERNAL LOGIC (Unchanged) ---

async def get_quantitative_analysis(cv_text: str, job_text: str) -> QuantitativeAnalysisResponse:
    raw_cv_skills = extract_skills_from_text(cv_text)
    raw_job_skills = extract_skills_from_text(job_text)
    cv_skills = normalize_skills(raw_cv_skills, data_loader)
    job_skills = normalize_skills(raw_job_skills, data_loader)
    cv_skill_names = set(s['normalized'].lower() for s in cv_skills)
    job_skill_names = set(s['normalized'].lower() for s in job_skills)
    matched_names = cv_skill_names & job_skill_names
    missing_names = job_skill_names - cv_skill_names
    matched_skills_info = [s for s in cv_skills if s['normalized'].lower() in matched_names]
    missing_skills_prioritized = []
    for skill_name in missing_names:
        demand_info = data_loader.get_market_demand(skill_name)
        missing_skills_prioritized.append(demand_info)
    missing_skills_prioritized.sort(key=lambda x: x['total_demand'], reverse=True)
    if not job_skill_names:
        reliable_score = 50.0
    else:
        reliable_score = (len(matched_names) / len(job_skill_names)) * 100
    return QuantitativeAnalysisResponse(
        overall_score=round(reliable_score, 2),
        skills_breakdown={
            'cv_skills_count': len(cv_skill_names),
            'job_skills_count': len(job_skill_names),
            'matched_count': len(matched_names),
            'missing_count': len(missing_names)
        },
        matched_skills=matched_skills_info,
        missing_skills_prioritized=missing_skills_prioritized,
        cv_skills=cv_skills,
        job_skills=job_skills
    )

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Helper function to parse PDF file stream."""
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def _sanitize_for_ai(text: str) -> str:
    """Redact potentially sensitive content."""
    if not text:
        return text
    original_len = len(text)
    # Remove emails more aggressively
    text = re.sub(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+', '[EMAIL_REDACTED]', text)
    # Remove phone numbers
    text = re.sub(r'(?:\+?\d[\s.-]?){7,15}', '[PHONE_REDACTED]', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL_REDACTED]', text)
    # Remove addresses
    cleaned_lines = []
    for line in text.splitlines():
        lower = line.lower()
        # Skip lines with address indicators
        if any(tok in lower for tok in ["street", "st.", "avenue", "ave", "road", "rd.", "boulevard", "blvd", "apt", "suite", "unit"]):
            continue
        # Skip lines with zip codes
        if re.search(r'\b\d{5}(?:-\d{4})?\b', line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove long number sequences
    text = re.sub(r'\b\d{9,}\b', '[NUMBER_REDACTED]', text)
    # Limit length
    if len(text) > 8000:
        text = text[:8000]
    return text

@router.post("/analyze", response_model=FullAnalysisResponse, tags=["Analysis"])
async def full_ai_analysis(
    cv_file: UploadFile = File(..., description="The user's CV in PDF format."),
    job_description: str = Form(..., description="The full text of the job description."),
    job_title: str = Form("Target Role", description="The job title (e.g., 'Senior Financial Analyst').")
):
    """Performs a full, AI-powered analysis from a PDF CV and job description text."""
    
    try:
        cv_bytes = await cv_file.read()
        cv_text = extract_text_from_pdf(io.BytesIO(cv_bytes))
        if not cv_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. The file might be an image or corrupt.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    finally:
        await cv_file.close()

    sanitized_attempted = False
    max_attempts = 3
    try:
        quant_report = await get_quantitative_analysis(cv_text, job_description)
        
        attempt = 1
        while attempt <= max_attempts:
            try:
                ai_analyzer_response = await call_gemini_analyzer(
                    cv_text=cv_text,
                    job_text=job_description,
                    cv_skills=quant_report.cv_skills,
                    job_skills=quant_report.job_skills
                )
                break
            except HTTPException as he:
                if he.status_code != 400 or 'safety' not in he.detail.lower():
                    raise he
                if attempt >= max_attempts:
                    # If all attempts failed, return a more helpful error
                    raise HTTPException(
                        status_code=400,
                        detail="The AI safety filter has blocked this content. Please ensure your CV and job description don't contain sensitive personal information (addresses, phone numbers, etc.) or potentially harmful content. Try uploading a sanitized version of your CV."
                    )
                # Try sanitizing more aggressively
                cv_text = _sanitize_for_ai(cv_text)
                job_description = _sanitize_for_ai(job_description)
                sanitized_attempted = True
                attempt += 1
                print(f"⚠️ Safety filter triggered, retrying with sanitization (attempt {attempt}/{max_attempts})")
                continue

        cv_profile_map = {p['skill'].lower(): p for p in ai_analyzer_response['cv_profile']}
        job_gap_profile = []
        
        for job_req in ai_analyzer_response['job_profile']:
            req_skill_lower = job_req['skill'].lower()
            cv_match = cv_profile_map.get(req_skill_lower)
            
            proficiency_you = cv_match['proficiency_you'] if cv_match else 0
            proficiency_req = job_req['proficiency_req']
            
            market_data = data_loader.get_market_demand(job_req['skill'])
            
            job_gap_profile.append(GapItem(
                skill=job_req['skill'],
                proficiency_req=proficiency_req,
                proficiency_you=proficiency_you,
                gap=(proficiency_req - proficiency_you),
                is_must_have=job_req.get('is_must_have', False),
                market_demand=market_data
            ))
        
        job_gap_profile.sort(key=lambda x: (not x.is_must_have, -x.gap, -x.market_demand.total_demand))

        critical_gaps_for_coach = [g.dict() for g in job_gap_profile if g.gap > 0][:10]

        ai_coach_response = await call_gemini_coach(
            job_title=job_title,
            gap_list=critical_gaps_for_coach,
            low_value_skills=ai_analyzer_response.get('low_value_skills', []),
            cv_text=cv_text
        )

        return FullAnalysisResponse(
            quantitative_summary=quant_report,
            ai_scores=ai_analyzer_response['overall_scores'],
            ai_summary=ai_coach_response['summary'],
            cv_skill_profile=ai_analyzer_response['cv_profile'],
            job_skill_profile=job_gap_profile,
            priority_actions=ai_coach_response['priority_actions'],
            learning_paths=ai_coach_response['learning_paths'],
            resume_edits=ai_coach_response['resume_edits'],
            low_value_skills=ai_analyzer_response.get('low_value_skills', [])
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal analysis failure: " + str(e))