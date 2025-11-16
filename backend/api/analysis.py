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

def _sanitize_for_ai(text: str, aggressive: bool = False) -> str:
    """Redact potentially sensitive content to pass AI safety filters."""
    if not text:
        return text
    
    # STEP 1: Remove all email addresses (multiple patterns)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z0-9._%+-]+\s*@\s*[A-Z0-9.-]+\.[A-Z]{2,}\b', '', text, flags=re.IGNORECASE)
    
    # STEP 2: Remove all phone numbers (comprehensive patterns)
    text = re.sub(r'\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}', '', text)
    
    # STEP 3: Remove all URLs and domains
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|io|co|uk|ca)\b', '', text, flags=re.IGNORECASE)
    
    # STEP 4: Remove social media handles and links
    text = re.sub(r'(?:linkedin|github|twitter|facebook|instagram)\.com[^\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # STEP 5: Remove identification numbers
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '', text)  # SSN
    text = re.sub(r'\b\d{9,}\b', '', text)  # Long numbers
    text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '', text)  # Zip codes
    
    # STEP 6: Remove address-related content (line by line)
    cleaned_lines = []
    address_keywords = [
        "street", "st.", "st ", "avenue", "ave.", "ave ", "road", "rd.", "rd ",
        "drive", "dr.", "dr ", "boulevard", "blvd.", "blvd ", "lane", "ln.", "ln ",
        "court", "ct.", "ct ", "circle", "cir.", "way", "place", "pl.",
        "apartment", "apt", "apt.", "suite", "ste", "ste.", "unit", "floor",
        "building", "p.o. box", "po box", "pobox"
    ]
    
    for line in text.splitlines():
        lower = line.lower().strip()
        
        # Skip empty lines
        if not lower:
            cleaned_lines.append(line)
            continue
            
        # Skip lines with address indicators
        has_address = any(f" {keyword}" in f" {lower}" or f"{keyword} " in f"{lower} " 
                         for keyword in address_keywords)
        if has_address:
            continue
            
        # Skip lines that look like addresses (number + street pattern)
        if re.search(r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)', line, re.IGNORECASE):
            continue
            
        # Skip lines with city, state patterns
        if re.search(r',\s*[A-Z]{2}\s*\d{5}', line):
            continue
        if re.search(r',\s*[A-Z][a-z]+\s+[A-Z]{2}', line):
            continue
            
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # STEP 7: Aggressive sanitization for second attempt
    if aggressive:
        # Remove dates (potential DOB)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
        
        # Remove any remaining long sequences of digits
        text = re.sub(r'\b\d{4,}\b', '', text)
        
        # Remove PII-related lines
        text = re.sub(r'(?:DOB|Date of Birth|SSN|Social Security|Driver\'?s License|Passport)[:\s]+[^\n]+', '', text, flags=re.IGNORECASE)
        
        # Remove lines with "born" or "age"
        lines = []
        for line in text.splitlines():
            lower = line.lower()
            if 'born' not in lower and 'age:' not in lower and 'age ' not in lower:
                lines.append(line)
        text = "\n".join(lines)
    
    # STEP 8: Clean up formatting
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r' {2,}', ' ', text)  # Remove excessive spaces
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
    
    # STEP 9: Limit length
    if len(text) > 6000:
        text = text[:6000]
    
    return text.strip()

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

    # Pre-sanitize inputs aggressively from the start
    cv_text = _sanitize_for_ai(cv_text, aggressive=True)
    job_description = _sanitize_for_ai(job_description, aggressive=True)
    print(f"✅ CV sanitized: {len(cv_text)} chars")
    print(f"✅ Job description sanitized: {len(job_description)} chars")
    
    try:
        quant_report = await get_quantitative_analysis(cv_text, job_description)
        
        try:
            ai_analyzer_response = await call_gemini_analyzer(
                cv_text=cv_text,
                job_text=job_description,
                cv_skills=quant_report.cv_skills,
                job_skills=quant_report.job_skills
            )
            print(f"✅ AI analysis completed successfully")
        except HTTPException as he:
            print(f"❌ AI analysis failed: {he.detail}")
            if he.status_code == 400 and 'safety' in he.detail.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Unable to process this CV/job description. Please ensure you've removed ALL personal information including: names, addresses, phone numbers, emails, and any identifying details. Focus on skills, experience, and qualifications only."
                )
            raise he

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