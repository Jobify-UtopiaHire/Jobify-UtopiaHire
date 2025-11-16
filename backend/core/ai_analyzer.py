import google.generativeai as genai
import os
import json
import re
from typing import Dict, Any, List
from fastapi import HTTPException
from dotenv import load_dotenv
from enum import IntEnum

load_dotenv()

try:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    genai.configure(api_key=API_KEY)
    
    generation_config = {
        "temperature": 0.2, 
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

except Exception as e:
    model = None

def clean_json_response(text: str) -> Dict[str, Any]:
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    json_str = ""
    
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {text[:200]}")
        
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError(f"Incomplete JSON object (unbalanced braces). Response may have been truncated.")
        
        json_str = text[start_idx:end_idx]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        json_str_fixed = re.sub(r',\s*([\]\}])', r'\1', json_str)
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON. Original error: {e}. The response might be incomplete or invalid.")

class FinishReason(IntEnum):
    STOP = 0
    MAX_TOKENS = 1
    SAFETY = 2
    RECITATION = 3
    OTHER = 4
    BLOCKLIST = 5
    PROHIBITED_CONTENT = 6
    SPII = 7
    MALFORMED_FUNCTION_CALL = 8


def _extract_candidate_text(candidate) -> str:
    """Safely pull text from a single candidate's content parts."""
    if not getattr(candidate, "content", None):
        return ""
    parts = getattr(candidate.content, "parts", []) or []
    texts = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            texts.append(t)
    return "\n".join(texts).strip()


def check_response_safety(response):
    """Validate Gemini response and handle safety blocks."""
    try:
        candidates = getattr(response, "candidates", None) or []
    except Exception:
        raise HTTPException(status_code=500, detail="AI Error: Missing candidates in response.")

    if not candidates:
        block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if block_reason:
            raise HTTPException(status_code=400, detail=f"AI request blocked for safety: {block_reason}")
        raise HTTPException(status_code=502, detail="AI Error: Empty response (no candidates).")

    collected_texts = []
    blocked_reasons = []
    finish_reasons = []

    for cand in candidates:
        fr = getattr(cand, "finish_reason", None)
        finish_reasons.append(fr)
        if fr in {FinishReason.SAFETY, FinishReason.BLOCKLIST, FinishReason.PROHIBITED_CONTENT, FinishReason.SPII}:
            blocked_reasons.append(fr)
            continue
        text_segment = _extract_candidate_text(cand)
        if text_segment:
            collected_texts.append(text_segment)

    if blocked_reasons and not collected_texts:
        raise HTTPException(status_code=400, detail="AI request blocked for safety. Please review the CV / job description content.")

    if not collected_texts:
        if FinishReason.MAX_TOKENS in finish_reasons:
            raise HTTPException(status_code=502, detail="AI response truncated (max tokens reached) and contained no usable text. Try reducing input size.")
        raise HTTPException(status_code=502, detail="AI returned no textual content. Try again or simplify input.")

    full_text = "\n".join(collected_texts).strip()
    if not full_text:
        raise HTTPException(status_code=502, detail="AI produced empty text after extraction.")
    return full_text

ANALYZER_PROMPT_TEMPLATE = """
You are a 'Skills Gap Analyzer' AI. I have extracted text from a CV and a Job Description. 
Note: Personal information (emails, phone numbers, addresses) has been removed for privacy.
Analyze them and return ONLY a valid JSON object.
Based on the provided text, you must infer proficiency levels from 1 (basic) to 5 (expert). 
- Look for words like 'expert', 'advanced', 'proficient' (3-4), or 'basic', 'familiar' (1-2) in the CV.
- Look for years of experience (e.g., 6 years = 5, 3-5 years = 4, 1-2 years = 3).
- For the job, look for 'must-have', 'expert' (4-5), 'required' (3-4), 'plus', 'nice-to-have' (2).
- If no info, estimate: 3 for CV skills, 3 for job skills.
Return ONLY a JSON object with this exact structure:
{{
  "cv_profile": [ {{"skill": "Skill Name", "proficiency_you": 1-5, "evidence": "snippet from CV"}} ],
  "job_profile": [ {{"skill": "Skill Name", "proficiency_req": 1-5, "is_must_have": true/false}} ],
  "overall_scores": {{ "coverage": 0-100, "depth": 0-100, "recency": 0-100 }},
  "low_value_skills": ["Skill 1", "Skill 2"]
}}
---
HERE IS THE DATA:
---
CV_TEXT: {cv_text}
---
JOB_DESCRIPTION_TEXT: {job_description}
---
CV_SKILLS_EXTRACTED: {cv_skills_list}
---
JOB_SKILLS_EXTRACTED: {job_skills_list}
"""
async def call_gemini_analyzer(cv_text: str, job_text: str, cv_skills: List[Any], job_skills: List[Any]) -> Dict[str, Any]:
    if not model: raise HTTPException(status_code=500, detail="Gemini AI model is not initialized.")
    cv_skills_simple = [{"skill": s.normalized, "evidence": s.evidence} for s in cv_skills]
    job_skills_simple = [s.normalized for s in job_skills]
    prompt = ANALYZER_PROMPT_TEMPLATE.format(
        cv_text=cv_text[:3000], job_description=job_text[:3000],
        cv_skills_list=json.dumps(cv_skills_simple, indent=2),
        job_skills_list=json.dumps(job_skills_simple, indent=2)
    )
    try:
        response = await model.generate_content_async(prompt)
        response_text = check_response_safety(response)
        return clean_json_response(response_text)
    except Exception as e:
        raise e

COACH_PROMPT_TEMPLATE = """
You are 'UtopiaHire', an AI Career Coach. A user has a skills gap. 
I will provide the target role, their critical gaps, irrelevant skills, AND THEIR CV TEXT.
Note: Personal information (emails, phone numbers, addresses) has been removed for privacy.
Your job is to provide encouraging, actionable advice in a JSON format.

Guidelines:
- Be encouraging and professional.
- For `priority_actions`, focus on the top 3-4 gaps. Give a brief 'why'.
- For `learning_paths`, find 1-2 REAL courses from platforms like Coursera, Udacity, freeCodeCamp, or official documentation.
- For `resume_edits`:
  - You MUST find an ACTUAL sentence or bullet point from the `USER_CV_TEXT` to use as the "before" snippet.
  - The "before" snippet MUST be a real quote. DO NOT invent it.
  - Then, provide an "after" snippet that re-frames that exact sentence to be more aligned with the target role.

Return ONLY a JSON object with this exact structure:
{{
  "summary": "Brief, 2-sentence summary of the user's position.",
  "priority_actions": [
    {{"action": "Learn Skill X", "difficulty": "low/medium/high", "time_estimate": "e.g., 2 weeks", "why": "Briefly explain impact."}}
  ],
  "learning_paths": [
    {{"skill": "Skill Name", "path_title": "Course/Resource Name", "platform": "Coursera / Udacity / etc."}}
  ],
  "resume_edits": [
    {{"before": "An actual sentence from the user's CV...", "after": "Suggested improved text snippet..."}}
  ]
}}

---
HERE IS THE DATA:
---
TARGET_ROLE: {job_title}
---
CRITICAL_GAPS: {gap_list}
---
LOW_VALUE_SKILLS_FOR_THIS_JOB: {low_value_skills_json}
---
USER_CV_TEXT:
{cv_text}
"""

async def call_gemini_coach(job_title: str, gap_list: List[Dict], low_value_skills: List[str], cv_text: str) -> Dict[str, Any]:
    if not model:
        raise HTTPException(status_code=500, detail="Gemini AI model is not initialized.")

    if not job_title:
        job_title = "Target Role" 

    prompt = COACH_PROMPT_TEMPLATE.format(
        job_title=job_title,
        gap_list=json.dumps(gap_list, indent=2),
        low_value_skills_json=json.dumps(low_value_skills, indent=2),
        cv_text=cv_text[:3000]
    )
    
    try:
        response = await model.generate_content_async(prompt)
        response_text = check_response_safety(response)
        return clean_json_response(response_text)
    except Exception as e:
        raise e