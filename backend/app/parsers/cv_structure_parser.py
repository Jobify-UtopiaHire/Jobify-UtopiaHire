"""
CV Structure Parser Module
Parses CV text into structured fields using AI (Gemini)
"""

from google import generativeai as genai
import json
import re
import os
import mimetypes
from datetime import datetime


def parse_and_analyze_cv(cv_input, job_description: str, api_key: str, links: list = None) -> dict:
    """
    Parse CV (from file or text) into structured data AND analyze it in a single API call.
    Uses HYBRID approach: PDF visual analysis + extracted links metadata.
    
    Args:
        cv_input: Either a file path (str) to PDF file, or raw CV text (str)
        job_description: The job description to match against (optional)
        api_key: Gemini API key
        links: List of URLs/links found in the CV (optional, recommended for PDFs)
    
    Returns:
        dict: Combined result with structured_data and analysis
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Determine if input is a file path or text
    is_file = False
    cv_text = None
    uploaded_file = None
    
    if isinstance(cv_input, str):
        if os.path.isfile(cv_input):
            is_file = True
            print(f"   📄 Processing PDF file: {cv_input}")
            
            # Get file type
            mime_type, _ = mimetypes.guess_type(cv_input)
            if mime_type is None:
                ext = os.path.splitext(cv_input)[1].lower()
                mime_map = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.txt': 'text/plain'
                }
                mime_type = mime_map.get(ext, 'application/octet-stream')
            
            print(f"   📎 MIME type: {mime_type}")
            
            # Upload file to Gemini for visual analysis
            try:
                uploaded_file = genai.upload_file(cv_input, mime_type=mime_type)
                print(f"   ✅ File uploaded to Gemini: {uploaded_file.name}")
                print(f"   ⏳ Waiting for file to be processed...")
                
                # Wait for file to be processed
                import time
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2)
                    uploaded_file = genai.get_file(uploaded_file.name)
                
                if uploaded_file.state.name == "FAILED":
                    print(f"   ❌ File processing failed: {uploaded_file.state}")
                    return {
                        'status': 'error',
                        'message': 'File processing failed on Gemini API'
                    }
                
                print(f"   ✅ File processing complete: {uploaded_file.state.name}")
                
            except Exception as e:
                print(f"   ❌ Error uploading file to Gemini: {str(e)}")
                return {
                    'status': 'error',
                    'message': f'Failed to upload file to Gemini: {str(e)}'
                }
        else:
            # It's text content
            cv_text = cv_input
            print(f"   📝 Processing text content ({len(cv_text)} characters)")
    else:
        cv_text = str(cv_input)
    
    if links:
        print(f"   🔗 Including {len(links)} extracted links as metadata")
    
    # Build job context
    is_job_desciption = job_description and job_description.strip()
    if is_job_desciption:
        job_context = f"""
        **JOB DESCRIPTION PROVIDED - FULL ATS CHECK MODE**

Job Description:
{job_description}

You will perform BOTH:
1. Technical ATS Check (formatting, parsability, structure)
2. Job Match Analysis (keyword matching, relevance scoring against the job description)

For the ats_analysis section:
- relevance_score: Score 0-100 based on how well the CV matches the job description
- keyword_matches: List specific keywords from the JD that appear in the CV with their count
- missing_keywords: List critical keywords from the JD that are missing from the CV
- recommendations: Provide specific suggestions to improve job match by incorporating missing keywords
"""
    else:
        job_context = """
**NO JOB DESCRIPTION PROVIDED - TECHNICAL ATS CHECK ONLY**

You will perform ONLY:
- Technical ATS Check (formatting, parsability, section structure, readability by ATS systems)

CRITICAL INSTRUCTIONS for ats_analysis section when NO job description is provided:
- relevance_score: Score 0-100 based on OVERALL CV READABILITY AND ATS COMPATIBILITY (not job matching). Consider:
  * Formatting consistency and clarity (dates, section headers, bullet points)
  * Information completeness (contact info, education, skills present)
  * ATS-friendly structure (standard sections, no tables/columns, clear hierarchy)
  * Content quality (descriptive, quantified achievements, professional language)
  * This score represents how well-formatted and parseable the CV is for automated systems
- keyword_matches: Use EMPTY ARRAY [] - do not invent job-specific keywords
- missing_keywords: Use EMPTY ARRAY [] - cannot determine missing keywords without a job description
- recommendations: Focus ONLY on technical ATS improvements (e.g., "Use standard section headers like 'Experience' instead of 'Work History'", "Remove tables/columns that ATS may not parse correctly", "Improve consistency in date formatting", "Add more descriptive action verbs to bullet points")

DO NOT:
- Invent or assume a job description
- Mention specific job requirements or skills gaps
- Provide job-tailoring advice
- Calculate relevance to an imaginary position

Focus on whether the CV is technically ATS-compatible AND well-structured for readability.
"""
    
    # Build links context
    links_context = ""
    if links and len(links) > 0:
        links_context = f"""

**IMPORTANT - Links/URLs found in this CV:**
{chr(10).join(f"- {link}" for link in links)}

These links may appear as icons, underlined text, or embedded in the visual document.
Please include these links in the appropriate contact fields (LinkedIn, GitHub, Portfolio, etc.).
When you see text that appears to be a profile name or icon, check if it corresponds to one of these URLs.
"""
    
    # Build prompt based on input type
    if is_file and uploaded_file:
        # File-based prompt (Gemini can see the visual layout)
        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = f"""
**CONTEXT**: Today's date is {current_date}. Use this to calculate experience durations, graduation timelines, and determine if dates are current or past.

your language output should follow the same language as the CV document.

You need to perform TWO tasks in one response:

TASK 1: Parse this CV document into structured JSON format
TASK 2: Analyze the CV and provide detailed, field-targeted feedback

**IMPORTANT**: The CV is provided as a visual document (PDF). You can see the actual formatting, icons, and layout.
{links_context}

{job_context}

When parsing the document:
- Look at the visual layout to understand section headers and structure
- Icons and symbols in the document may represent social media profiles (LinkedIn, GitHub, Zindi, etc.)
- The links listed above correspond to clickable elements in the visual document
- Match profile names/icons with the provided URLs to populate contact information correctly
- DO NOT criticize icons or symbols - they are intentional design elements for social links
- Use today's date ({current_date}) to calculate experience durations and validate if "Present" positions are truly current

Return ONLY valid JSON (no markdown) with this EXACT structure:

{{
    "structured_cv": {{
        "summary": "Professional summary or objective statement (if present)",
        "contact": {{
            "name": "Full name",
            "email": "email@example.com",
            "phone": "phone number",
            "location": "city, country",
            "linkedin": "LinkedIn URL",
            "github": "GitHub URL",
            "portfolio": "Portfolio URL"
        }},
        "experience": [
            {{
                "id": "exp_1",
                "title": "Job title",
                "company": "Company name",
                "location": "Location",
                "startDate": "Start date",
                "endDate": "End date or Present",
                "description": "Full description of responsibilities and achievements",
                "bullets": ["Bullet point 1", "Bullet point 2"]
            }}
        ],
        "education": [
            {{
                "id": "edu_1",
                "degree": "Degree name",
                "institution": "School/University name",
                "location": "Location",
                "startDate": "Start date",
                "endDate": "End date or Expected",
                "gpa": "GPA if mentioned",
                "description": "Additional details",
                "achievements": ["Achievement 1", "Achievement 2"]
            }}
        ],
        "skills": {{
            "technical": ["skill1", "skill2"],
            "languages": ["language1", "language2"],
            "tools": ["tool1", "tool2"],
            "soft_skills": ["skill1", "skill2"],
            "other": ["other skill 1", "other skill 2"]
        }},
        "projects": [
            {{
                "id": "proj_1",
                "name": "Project name",
                "description": "Project description",
                "technologies": ["tech1", "tech2"],
                "link": "Project URL if available"
            }}
        ],
        "certifications": [
            {{
                "id": "cert_1",
                "name": "Certification name",
                "issuer": "Issuing organization",
                "date": "Date obtained",
                "credential": "Credential ID or URL"
            }}
        ],
        "awards": [
            {{
                "id": "award_1",
                "name": "Award name",
                "issuer": "Issuing organization",
                "date": "Date",
                "description": "Description"
            }}
        ],
        "publications": [
            {{
                "id": "pub_1",
                "title": "Publication title",
                "authors": "Authors",
                "venue": "Conference/Journal name",
                "date": "Date",
                "link": "URL if available"
            }}
        ],
        "activities": [
            {{
                "id": "act_1",
                "organization": "Organization name (e.g., IEEE, club, association)",
                "title": "Role/Position",
                "startDate": "Start date",
                "endDate": "End date or Present",
                "description": "Description of involvement and achievements"
            }}
        ],
        "volunteer": [
            {{
                "id": "vol_1",
                "organization": "Organization name",
                "role": "Role/Position",
                "startDate": "Start date",
                "endDate": "End date or Present",
                "description": "Description"
            }}
        ],
        "other_sections": {{
            "section_name_1": [
                {{
                    "id": "other_1",
                    "content": "Any content that doesn't fit standard categories"
                }}
            ]
        }}
    }},
    "analysis": {{
        "formatting": {{
            "score": <number 0-10>,
            "issues": ["Specific issue with exact location (IN CV LANGUAGE), e.g., 'Inconsistent date formatting in Experience section - mix of MM/YYYY and Month Year'"],
            "suggestions": ["Actionable fix with example (IN CV LANGUAGE), e.g., 'Standardize all dates to MM/YYYY format throughout CV'"]
        }},
        "content": {{
            "score": <number 0-10>,
            "strengths": ["Specific strength with concrete example from CV (IN CV LANGUAGE)"],
            "weaknesses": ["Specific weakness with location (IN CV LANGUAGE)"],
            "suggestions": ["Concrete improvement with before/after example (IN CV LANGUAGE)"]
        }},
        "general": {{
            "overall_score": <number 0-10>,
            "summary": "Honest 2-3 sentence assessment explaining the score (IN CV LANGUAGE). Mention specific strengths and the main areas holding the CV back.",
            "top_priorities": [
                {{
                    "priority": 1,
                    "action": "Specific, actionable task with exact location (IN CV LANGUAGE)",
                    "impact": "High|Medium|Low",
                    "time_estimate": "5 mins|15 mins|30 mins|1 hour",
                    "category": "Formatting|Content|Keywords|ATS"
                }}
            ]
        }},
        "sections": [
            {{
                "name": "Experience|Education|Skills|Summary|etc",
                "quality_score": <number 0-10>,
                "feedback": "Specific, honest feedback with examples from section (IN CV LANGUAGE)",
                "suggestions": ["Concrete suggestion with example (IN CV LANGUAGE)"]
            }}
        ],
        "field_suggestions": [
            {{
                "suggestionId": <unique number>,
                "targetField": "summary|experience|education|skills|etc",
                "fieldPath": ["experience", 0, "description"],
                "fieldId": "exp_1|edu_1|skill_1",
                "originalValue": "EXACT current value of the field from CV (IN ORIGINAL CV LANGUAGE)",
                "improvedValue": "Complete suggested replacement text (IN CV LANGUAGE)",
                "issue_type": "grammar|clarity|impact|keyword|formatting",
                "severity": "critical|high|medium|low",
                "problem": "Clear, specific explanation of what's wrong (IN CV LANGUAGE)",
                "explanation": "Why this change matters for recruiters/ATS (IN CV LANGUAGE)",
                "impact": "High|Medium|Low"
            }}
        ],
        "quick_wins": [
            {{
                "change": "Specific change with exact location and current issue (IN CV LANGUAGE)",
                "where": "Exact section and position (IN CV LANGUAGE)",
                "targetField": "field name if applicable",
                "fieldPath": ["path", "to", "field"],
                "effort": "5 mins|15 mins",
                "impact": "High|Medium"
            }}
        ],
        "ats_analysis": {{
            "relevance_score": <number 0-100>,
            "keyword_matches": ["Specific keywords found with count (IN CV LANGUAGE)"],
            "missing_keywords": ["Critical keywords from job description that are absent (IN CV LANGUAGE)"],
            "recommendations": ["Specific placement suggestions (IN CV LANGUAGE)"]
        }}
    }}
}}

LANGUAGE REQUIREMENT:
- **CRITICAL**: Detect the language of the CV from the content
- **All recommendations, suggestions, and improved text MUST be in the SAME language as the CV**
- If CV is in French, respond in French; Arabic in Arabic; English in English; Spanish in Spanish
- Mixed language CVs: match the dominant language (>60% of content)
- Field names in JSON structure remain in English, but ALL text content must match CV language

ADDITIONAL INSTRUCTIONS FOR SUGGESTIONS (READ CAREFULLY):

- If this CV appears to be technical (programming, engineering, ML, web, DevOps, data science, etc.) and the contact section does NOT include a GitHub profile URL, ALWAYS include a field-targeted suggestion that requests the candidate to add their GitHub URL. The suggestion's `fieldPath` should target ["contact", "github"]. The `improvedValue` must be normal human-readable sentences (not code or dict syntax). Example improvedValue: "Add your GitHub URL (e.g., https://github.com/yourusername) to allow recruiters to review your code, projects, and contributions."

- If this CV appears to be technical and there is no portfolio or personal website (both missing or "Not provided"), ALWAYS include a field-targeted suggestion requesting a portfolio/website URL. Use `fieldPath` ["contact", "portfolio"] or ["contact", "website"]. Use natural-language improvedValue like: "Add a portfolio URL to showcase your projects, demos, and live work in action."

- For every project entry that appears to be a software project (technologies list contains programming/tool keywords or description mentions coding/development), if the project lacks a repo/link, include a field-targeted suggestion targeting that project's `link` field (e.g., ["projects", 0, "link"]) asking the candidate to provide the GitHub repository URL and explain why (e.g., "Add the GitHub repository URL for this project so reviewers can inspect the implementation, code quality, and contributions").

- IMPORTANT: Do NOT return dict-like strings or code snippets in your field suggestions. All `improvedValue`, `problem`, and `explanation` fields must be written as natural language sentences (in the same language as the CV). The `fieldPath` should still be a valid JSON array.

- PRESERVE POSITION/ROLE: If the CV header or summary mentions the candidate's current position or role (e.g., "ICT Engineering student", "Senior Software Engineer", "Data Analyst"), ensure that rewritten summaries or improved versions preserve and emphasize this current role/position. Do not drop it.

PARSING GUIDELINES (for structured_cv):
- Extract ALL information from the CV accurately, including unusual or non-standard sections
- Use "Not provided" for missing fields
- Generate unique IDs for each entry (exp_1, exp_2, edu_1, etc.)
- Preserve the original wording and details
- If a section is completely absent, use an empty array or object
- For experience/education bullets, try to extract individual points
- Skills should be categorized appropriately
- Activities should include: clubs, associations, memberships, extracurricular activities, leadership roles
- If you find sections that don't match standard fields, add them to "other_sections"

SCORING RUBRIC (for analysis) - Be strict and honest:

**Overall Score (0-10):**
- 9-10: Exceptional - Ready to compete for top positions, zero improvements needed
- 7-8: Strong - Minor tweaks needed, well-structured with good metrics
- 5-6: Good foundation - Several improvements needed, lacks quantification
- 3-4: Needs work - Major gaps, weak descriptions, poor structure
- 0-2: Poor - Requires complete overhaul

**Automatic Penalties:**
- Each generic statement without metrics: -0.5 points
- Each bullet missing quantification: -0.5 points
- Grammatical errors or typos: -0.5 points each
- Passive voice or weak verbs: -0.3 points each
- Vague descriptions: -0.5 points each
- Irrelevant content: -1 point per section
- Missing critical information: -1 point per instance

**Severity Classifications:**
- **Critical**: Blocks job application or severely damages credibility
- **High**: Significantly weakens CV impact
- **Medium**: Noticeable but not deal-breaking
- **Low**: Polish and optimization

FIELD-TARGETED SUGGESTIONS REQUIREMENTS:
- Must identify EXACT field location using fieldPath
- Every suggestion needs concrete "improvedValue" with complete replacement text **IN THE SAME LANGUAGE AS THE CV**
- Focus on highest-impact changes first (prioritize High severity)
- Identify at least 3-5 critical/high severity issues if they exist
- Provide 8-15 high-impact suggestions total
- Use fieldPath array notation: ["experience", 0, "description"] for nested fields
- originalValue must EXACTLY match current value in structured_cv
- improvedValue must be COMPLETE replacement text, ready to use
- **MAINTAIN THE SAME LANGUAGE AS THE CV IN ALL TEXT FIELDS**

REMEMBER:
- Average CVs score 5-6/10, not 8/10
- Only exceptional CVs deserve 9-10/10
- Be specific with exact field locations
- All content analysis must reference the structured data you just created
"""
    else:
        # Text-based prompt (fallback for non-PDF or text input)
        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = f"""
**CONTEXT**: Today's date is {current_date}. Use this to calculate experience durations, graduation timelines, and determine if dates are current or past.

You need to perform TWO tasks in one response:

TASK 1: Parse this CV text into structured JSON format
TASK 2: Analyze the CV and provide detailed, field-targeted feedback

{job_context}
{links_context}

CV Text:
{cv_text}

When parsing:
- Use today's date ({current_date}) to calculate experience durations and validate if "Present" positions are truly current
- Calculate years of experience based on start/end dates
- Determine if the candidate is currently employed or studying

Return ONLY valid JSON (no markdown) with this EXACT structure:

{{
    "structured_cv": {{
        "summary": "Professional summary or objective statement (if present)",
        "contact": {{
            "name": "Full name",
            "email": "email@example.com",
            "phone": "phone number",
            "location": "city, country",
            "linkedin": "LinkedIn URL",
            "github": "GitHub URL",
            "portfolio": "Portfolio URL"
        }},
        "experience": [
            {{
                "id": "exp_1",
                "title": "Job title",
                "company": "Company name",
                "location": "Location",
                "startDate": "Start date",
                "endDate": "End date or Present",
                "description": "Full description of responsibilities and achievements",
                "bullets": ["Bullet point 1", "Bullet point 2"]
            }}
        ],
        "education": [
            {{
                "id": "edu_1",
                "degree": "Degree name",
                "institution": "School/University name",
                "location": "Location",
                "startDate": "Start date",
                "endDate": "End date or Expected",
                "gpa": "GPA if mentioned",
                "description": "Additional details",
                "achievements": ["Achievement 1", "Achievement 2"]
            }}
        ],
        "skills": {{
            "technical": ["skill1", "skill2"],
            "languages": ["language1", "language2"],
            "tools": ["tool1", "tool2"],
            "soft_skills": ["skill1", "skill2"],
            "other": ["other skill 1", "other skill 2"]
        }},
        "projects": [
            {{
                "id": "proj_1",
                "name": "Project name",
                "description": "Project description",
                "technologies": ["tech1", "tech2"],
                "link": "Project URL if available"
            }}
        ],
        "certifications": [
            {{
                "id": "cert_1",
                "name": "Certification name",
                "issuer": "Issuing organization",
                "date": "Date obtained",
                "credential": "Credential ID or URL"
            }}
        ],
        "awards": [
            {{
                "id": "award_1",
                "name": "Award name",
                "issuer": "Issuing organization",
                "date": "Date",
                "description": "Description"
            }}
        ],
        "publications": [
            {{
                "id": "pub_1",
                "title": "Publication title",
                "authors": "Authors",
                "venue": "Conference/Journal name",
                "date": "Date",
                "link": "URL if available"
            }}
        ],
        "activities": [
            {{
                "id": "act_1",
                "organization": "Organization name (e.g., IEEE, club, association)",
                "title": "Role/Position",
                "startDate": "Start date",
                "endDate": "End date or Present",
                "description": "Description of involvement and achievements"
            }}
        ],
        "volunteer": [
            {{
                "id": "vol_1",
                "organization": "Organization name",
                "role": "Role/Position",
                "startDate": "Start date",
                "endDate": "End date or Present",
                "description": "Description"
            }}
        ],
        "other_sections": {{
            "section_name_1": [
                {{
                    "id": "other_1",
                    "content": "Any content that doesn't fit standard categories"
                }}
            ]
        }}
    }},
    "analysis": {{
        "formatting": {{
            "score": <number 0-10>,
            "issues": ["Specific issue with exact location (IN CV LANGUAGE)"],
            "suggestions": ["Actionable fix with example (IN CV LANGUAGE)"]
        }},
        "content": {{
            "score": <number 0-10>,
            "strengths": ["Specific strength with concrete example from CV (IN CV LANGUAGE)"],
            "weaknesses": ["Specific weakness with location (IN CV LANGUAGE)"],
            "suggestions": ["Concrete improvement with before/after example (IN CV LANGUAGE)"]
        }},
        "general": {{
            "overall_score": <number 0-10>,
            "summary": "Honest 2-3 sentence assessment explaining the score (IN CV LANGUAGE)",
            "top_priorities": [
                {{
                    "priority": 1,
                    "action": "Specific, actionable task with exact location (IN CV LANGUAGE)",
                    "impact": "High|Medium|Low",
                    "time_estimate": "5 mins|15 mins|30 mins|1 hour",
                    "category": "Formatting|Content|Keywords|ATS"
                }}
            ]
        }},
        "sections": [
            {{
                "name": "Experience|Education|Skills|Summary|etc",
                "quality_score": <number 0-10>,
                "feedback": "Specific, honest feedback with examples from section (IN CV LANGUAGE)",
                "suggestions": ["Concrete suggestion with example (IN CV LANGUAGE)"]
            }}
        ],
        "field_suggestions": [
            {{
                "suggestionId": <unique number>,
                "targetField": "summary|experience|education|skills|etc",
                "fieldPath": ["experience", 0, "description"],
                "fieldId": "exp_1|edu_1|skill_1",
                "originalValue": "EXACT current value of the field from CV (IN ORIGINAL CV LANGUAGE)",
                "improvedValue": "Complete suggested replacement text (IN CV LANGUAGE)",
                "issue_type": "grammar|clarity|impact|keyword|formatting",
                "severity": "critical|high|medium|low",
                "problem": "Clear, specific explanation of what's wrong (IN CV LANGUAGE)",
                "explanation": "Why this change matters for recruiters/ATS (IN CV LANGUAGE)",
                "impact": "High|Medium|Low"
            }}
        ],
        "quick_wins": [
            {{
                "change": "Specific change with exact location (IN CV LANGUAGE)",
                "where": "Exact section and position (IN CV LANGUAGE)",
                "targetField": "field name if applicable",
                "fieldPath": ["path", "to", "field"],
                "effort": "5 mins|15 mins",
                "impact": "High|Medium"
            }}
        ],
        "ats_analysis": {{
            "relevance_score": <number 0-100>,
            "keyword_matches": ["Specific keywords found (IN CV LANGUAGE)"],
            "missing_keywords": ["Critical keywords missing (IN CV LANGUAGE)"],
            "recommendations": ["Specific placement suggestions (IN CV LANGUAGE)"]
        }}
    }}
}}

LANGUAGE REQUIREMENT:
- **CRITICAL**: Detect the language of the CV from the content
- **All recommendations, suggestions, and improved text MUST be in the SAME language as the CV**
- If CV is in French, respond in French; Arabic in Arabic; English in English; Spanish in Spanish
- Mixed language CVs: match the dominant language (>60% of content)
- Field names in JSON structure remain in English, but ALL text content must match CV language


PARSING GUIDELINES:
- Extract ALL information accurately
- Generate unique IDs for each entry (exp_1, exp_2, etc.)
- Use "Not provided" for missing fields

FIELD-TARGETED SUGGESTIONS:
- Provide 8-15 high-impact suggestions
- Use fieldPath array notation
- originalValue must EXACTLY match current value
- improvedValue must be COMPLETE replacement text

REMEMBER:
- Average CVs score 5-6/10, not 8/10
- Only exceptional CVs deserve 9-10/10
- Be specific with exact field locations
"""
    
    try:
        print("   📤 Sending request to Gemini API (parsing + analysis in one call)...")
        
        # Send request based on input type
        if is_file and uploaded_file:
            # Send file + prompt for visual analysis
            response = model.generate_content([uploaded_file, prompt])
        else:
            # Send text-only prompt
            response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove any markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()
        
        parsed = json.loads(response_text)
        
        # Extract structured CV and analysis
        structured_cv = parsed.get('structured_cv', {})
        analysis_raw = parsed.get('analysis', {})
        
        # Validate keywords for ATS analysis
        cv_text_lower = json.dumps(structured_cv).lower()
        ats_analysis = analysis_raw.get("ats_analysis", {})
        
        def is_keyword_in_text(keyword, text):
            escaped_keyword = re.escape(keyword.lower())
            pattern = rf'\b{escaped_keyword}\b'
            return bool(re.search(pattern, text.lower()))
        
        validated_keyword_matches = []
        validated_missing_keywords = []
        
        for keyword in ats_analysis.get("keyword_matches", []):
            if is_keyword_in_text(keyword, cv_text_lower):
                validated_keyword_matches.append(keyword)
            else:
                validated_missing_keywords.append(keyword)
        
        for keyword in ats_analysis.get("missing_keywords", []):
            if not is_keyword_in_text(keyword, cv_text_lower):
                validated_missing_keywords.append(keyword)
            else:
                validated_keyword_matches.append(keyword)
        
        validated_keyword_matches = list(set(validated_keyword_matches))
        validated_missing_keywords = list(set(validated_missing_keywords))
        
        # Build final result
        # Determine if job description was provided
        has_job_description = is_job_desciption
        
        result = {
            "status": "success",
            "structured_data": structured_cv,
            "original_text": cv_text if cv_text else "Processed from file",
            "analysis": {
                'overall_score': int(analysis_raw["general"]["overall_score"] * 10),
                'ats_score': ats_analysis.get("relevance_score", int((analysis_raw["content"]["score"] + analysis_raw["formatting"]["score"]) * 5)),
                'readability_score': 80 + (analysis_raw["formatting"]["score"] - 5) * 4,
                'summary': analysis_raw["general"]["summary"],
                'critical_issues': analysis_raw["formatting"]["issues"] + analysis_raw["content"]["weaknesses"],
                'field_suggestions': analysis_raw.get("field_suggestions", []),
                'section_analysis': analysis_raw.get("sections", []),
                'quick_wins': analysis_raw.get("quick_wins", []),
                'top_priorities': analysis_raw["general"]["top_priorities"],
                'grammar_and_clarity': {"issues": analysis_raw["formatting"]["issues"]},
                'job_match_analysis': {
                    "relevance_score": 0 if not has_job_description else ats_analysis.get("relevance_score", analysis_raw["content"]["score"] * 10),
                    "keyword_matches": validated_keyword_matches,
                    "missing_keywords": validated_missing_keywords,
                    "recommendations": ats_analysis.get("recommendations", [])
                },
                'global_analysis': {
                    "formatting_score": analysis_raw["formatting"]["score"],
                    "content_score": analysis_raw["content"]["score"]
                }
            }
        }
        
        # Clean up uploaded file if exists
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                print(f"   🗑️  Cleaned up uploaded file: {uploaded_file.name}")
            except Exception as e:
                print(f"   ⚠️  Could not delete uploaded file: {str(e)}")
        
        print(f"✅ Successfully parsed and analyzed CV in one API call")
        print(f"   - Experience entries: {len(structured_cv.get('experience', []))}")
        print(f"   - Education entries: {len(structured_cv.get('education', []))}")
        print(f"   - Skills categories: {len(structured_cv.get('skills', {}).keys())}")
        print(f"   - Overall score: {result['analysis']['overall_score']}/100")
        print(f"   - Field suggestions: {len(result['analysis']['field_suggestions'])}")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {str(e)}")
        print(f"Raw response: {response_text[:500]}")
        return {
            "status": "error",
            "message": "Failed to parse CV structure and analysis",
            "original_text": cv_text
        }
    
    except Exception as e:
        print(f"❌ Error parsing and analyzing CV: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "original_text": cv_text
        }




def apply_suggestion_to_structured_cv(structured_cv: dict, suggestion: dict) -> dict:
    """
    Apply a suggestion to the structured CV data.
    
    Args:
        structured_cv: The structured CV data
        suggestion: Suggestion with targetField and improvedValue
    
    Returns:
        dict: Updated structured CV data
    """
    target_field = suggestion.get('targetField')
    improved_value = suggestion.get('improvedValue')
    field_path = suggestion.get('fieldPath', [])
    
    print(f"🔄 Applying suggestion:")
    print(f"   Target field: {target_field}")
    print(f"   Field path (original): {field_path}")
    
    # FIX: Remove 'structured_cv' from the beginning of field_path if present
    # Gemini sometimes generates paths like ['structured_cv', 'summary'] when it should be ['summary']
    if field_path and len(field_path) > 0 and field_path[0] == 'structured_cv':
        field_path = field_path[1:]
        print(f"   Field path (corrected): {field_path}")
    
    print(f"   Improved value: {improved_value}")
    
    if not target_field or improved_value is None:
        print(f"❌ Missing target_field or improved_value")
        return structured_cv
    
    # Create a copy to avoid mutating the original
    updated_cv = json.loads(json.dumps(structured_cv))
    
    # Navigate to the target field using the field_path
    current = updated_cv
    
    try:
        # If field_path is provided, navigate through nested structure
        if field_path:
            for i, key in enumerate(field_path[:-1]):
                if isinstance(current, list):
                    current = current[int(key)]
                else:
                    current = current[key]
            
            # Update the final field
            final_key = field_path[-1]
            if isinstance(current, list):
                current[int(final_key)] = improved_value
            else:
                current[final_key] = improved_value
        else:
            # Simple field update
            updated_cv[target_field] = improved_value
        
        print(f"✅ Applied suggestion to field: {target_field} with value: {improved_value}")
        return updated_cv
        
    except (KeyError, IndexError, TypeError) as e:
        print(f"❌ Error applying suggestion: {str(e)}")
        print(f"   Field path: {field_path}")
        print(f"   Current structure type: {type(current)}")
        return structured_cv
