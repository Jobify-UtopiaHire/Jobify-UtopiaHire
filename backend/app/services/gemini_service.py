"""
Enhanced Gemini API Module for Structured CV Analysis
Generates suggestions that target specific fields in the structured CV data
"""

from google import generativeai as genai
from ollama import chat, ChatResponse
import json
import re
from PIL import Image


def analyze_structured_cv_with_gemini(structured_cv: dict, job_description: str, api_key: str, cv_images: list = None):
    """
    Analyze structured CV data with Gemini and return field-targeted suggestions.
    
    Args:
        structured_cv: The structured CV data (from cv_structure_parser)
        job_description: The job description to match against (optional)
        api_key: Gemini API key
        cv_images: Optional list of PIL Image objects showing CV layout
    
    Returns:
        dict: Analysis with field-targeted suggestions
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Convert structured CV to readable format for AI
    cv_json = json.dumps(structured_cv, indent=2)
    
    # Build prompt based on whether job description is provided
    job_context = f"""
Job Description:
{job_description}

Analyze this structured CV against the job description and provide detailed feedback on job relevance and keyword matching.
""" if job_description and job_description.strip() else """
No specific job description provided. Analyze the CV for general quality, formatting, and best practices.
"""
    
    prompt = f"""
Analyze this structured CV and provide detailed, field-targeted feedback with STRICT, REALISTIC scoring.

Structured CV Data:
{cv_json}

{job_context}

LANGUAGE REQUIREMENT:
- **CRITICAL**: Detect the language of the CV from the content
- **All recommendations, suggestions, and improved text MUST be in the SAME language as the CV**
- If CV is in French, respond in French
- If CV is in Arabic, respond in Arabic  
- If CV is in English, respond in English
- If CV is in Spanish, respond in Spanish
- Mixed language CVs: match the dominant language (>60% of content)
- Field names in JSON structure remain in English, but ALL text content (improvedValue, explanations, suggestions, feedback) must match CV language

SCORING RUBRIC (Be strict and honest):

**Overall Score (0-10):**
- 9-10: Exceptional - Ready to compete for top positions, zero improvements needed, perfect quantification
- 7-8: Strong - Minor tweaks needed, well-structured with good metrics and compelling content
- 5-6: Good foundation - Several improvements needed, lacks quantification or has generic content
- 3-4: Needs work - Major gaps in content, weak descriptions, poor structure, or missing key information
- 0-2: Poor - Requires complete overhaul, unprofessional, or incomprehensible

**Formatting Score (0-10):**
- 9-10: Perfect visual hierarchy, consistent spacing, ATS-friendly, professional structure
- 7-8: Clean layout with 1-2 minor inconsistencies
- 5-6: Readable but has formatting issues, inconsistent spacing, or cluttered sections
- 3-4: Poor structure, hard to scan, inconsistent formatting throughout
- 0-2: Chaotic layout, unprofessional, multiple major formatting issues

**Content Score (0-10):**
- 9-10: All bullets quantified, strong action verbs, highly relevant, zero fluff, impactful achievements
- 7-8: Good content with some quantification, mostly strong verbs, 1-2 weak bullets
- 5-6: Adequate but generic, minimal metrics, lacks specific achievements
- 3-4: Vague descriptions, no quantification, weak verbs, significant gaps
- 0-2: Minimal detail, no achievements, passive voice throughout

**Section Quality Score (0-10):**
- 9-10: Perfectly tailored, compelling, comprehensive, every bullet is strong
- 7-8: Strong with 1-2 bullets needing improvement
- 5-6: Functional but has generic content or missing key elements
- 3-4: Weak, unclear, poorly organized, or mostly generic
- 0-2: Missing critical information, confusing, or very poor quality

**ATS Relevance Score (0-100):**
- 90-100: Matches 90%+ of job requirements, optimal keyword density, perfect alignment
- 70-89: Matches 70-89% of requirements, good keyword usage with minor gaps
- 50-69: Matches 50-69% of requirements, some important keywords missing
- 30-49: Matches 30-49% of requirements, significant gaps in keywords
- 0-29: Poor match, most critical keywords missing

CRITICAL SCORING GUIDELINES:
- **Default to 5-6/10 for average CVs** - Most CVs are not excellent, be realistic
- **Only award 9-10/10 if truly exceptional** - No room for improvement
- **Be harsh but fair** - Identify real problems that hurt job prospects

**Automatic Penalties (deduct from scores):**
- Each generic statement without metrics ("responsible for", "worked on", "helped with"): -0.5 points
- Each bullet missing quantification (no numbers, %, scale, or timeframe): -0.5 points
- Grammatical errors or typos: -0.5 points each
- Passive voice or weak verbs ("was", "did", "made"): -0.3 points each
- Vague descriptions without specifics: -0.5 points each
- Irrelevant content for target job: -1 point per section
- Clichés without substance ("team player", "hard worker", "detail-oriented"): -0.3 each
- Missing critical information (dates, company names, achievements): -1 point per instance
- Poor formatting or inconsistent structure: -1 to -2 points total
- No summary or weak summary: -1 point

**Award Points For:**
- Specific, quantified achievements with clear impact: +1 per strong bullet
- Action verbs with measurable outcomes: +0.5
- Relevant keywords naturally integrated: +0.5
- Clear progression and growth shown: +1
- Tailored content matching job description: +1-2

**Severity Classifications:**
- **Critical**: Blocks job application or severely damages credibility (typos in contact info, missing essential sections, major formatting breaks, completely generic content)
- **High**: Significantly weakens CV impact (no metrics, vague descriptions, poor structure, weak bullets, missing keywords)
- **Medium**: Noticeable but not deal-breaking (minor formatting issues, could be more specific, weak word choices)
- **Low**: Polish and optimization (alternative phrasing, minor enhancements)

FIELD-TARGETED SUGGESTIONS REQUIREMENTS:
- Must identify EXACT field location using fieldPath
- Every suggestion needs concrete "improvedValue" with complete replacement text **IN THE SAME LANGUAGE AS THE CV**
- Focus on highest-impact changes first (prioritize High severity)
- Identify at least 3-5 critical/high severity issues if they exist
- Don't suggest changes to text that's already strong (9-10/10 quality)
- Be specific about WHY each change matters (recruiter impact, ATS compatibility, clarity)
- **All explanations, problems, and suggestions must be in the CV's language**

Return your response as **valid JSON** with this EXACT structure (no markdown):

{{
    "formatting": {{
        "score": <number 0-10>,
        "issues": ["Specific issue with exact location (IN CV LANGUAGE), e.g., 'Inconsistent date formatting in Experience section - mix of MM/YYYY and Month Year'"],
        "suggestions": ["Actionable fix with example (IN CV LANGUAGE), e.g., 'Standardize all dates to MM/YYYY format throughout CV'"]
    }},
    "content": {{
        "score": <number 0-10>,
        "strengths": ["Specific strength with concrete example from CV (IN CV LANGUAGE), e.g., 'Strong quantification in Python project: 15% efficiency improvement'"],
        "weaknesses": ["Specific weakness with location (IN CV LANGUAGE), e.g., 'Experience section lacks metrics - 4 out of 6 bullets have no quantification'"],
        "suggestions": ["Concrete improvement with before/after example (IN CV LANGUAGE)"]
    }},
    "general": {{
        "overall_score": <number 0-10>,
        "summary": "Honest 2-3 sentence assessment explaining the score (IN CV LANGUAGE). Mention specific strengths and the main areas holding the CV back.",
        "top_priorities": [
            {{
                "priority": 1,
                "action": "Specific, actionable task with exact location (IN CV LANGUAGE) (e.g., 'Add quantified metrics to all 5 bullet points in Software Developer role at TechCorp')",
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
            "feedback": "Specific, honest feedback with examples from section (IN CV LANGUAGE). Identify what's weak and why.",
            "suggestions": ["Concrete suggestion with example (IN CV LANGUAGE): 'Replace vague bullet \"Worked with databases\" with \"Optimized PostgreSQL queries reducing load time from 3s to 0.8s for 10K+ daily users\"'"]
        }}
    ],
    "field_suggestions": [
        {{
            "suggestionId": <unique number>,
            "targetField": "summary|experience|education|skills|etc",
            "fieldPath": ["experience", 0, "description"] or ["summary"] or ["contact", "email"],
            "fieldId": "exp_1|edu_1|skill_1 (the ID from structured CV)",
            "originalValue": "EXACT current value of the field from CV (IN ORIGINAL CV LANGUAGE)",
            "improvedValue": "Complete suggested replacement text (IN CV LANGUAGE) (REQUIRED - must be full, ready-to-use text)",
            "issue_type": "grammar|clarity|impact|keyword|formatting",
            "severity": "critical|high|medium|low",
            "problem": "Clear, specific explanation of what's wrong (IN CV LANGUAGE) (e.g., 'Lacks quantification and uses weak passive voice')",
            "explanation": "Why this change matters for recruiters/ATS (IN CV LANGUAGE) (e.g., 'Quantified achievements increase interview callbacks by 40% and show measurable impact')",
            "impact": "High|Medium|Low"
        }}
    ],
    "quick_wins": [
        {{
            "change": "Specific change with exact location and current issue (IN CV LANGUAGE) (e.g., 'Change \"Responsible for managing\" to \"Managed 5-person team, delivering 3 projects on time\"')",
            "where": "Exact section and position (IN CV LANGUAGE) (e.g., 'Experience section, 2nd bullet under current role')",
            "targetField": "field name if applicable",
            "fieldPath": ["path", "to", "field"],
            "effort": "5 mins|15 mins",
            "impact": "High|Medium"
        }}
    ],
    "ats_analysis": {{
        "relevance_score": <number 0-100>,
        "keyword_matches": ["Specific keywords found with count (IN CV LANGUAGE), e.g., 'Python (mentioned 3x)', 'SQL (2x)'"],
        "missing_keywords": ["Critical keywords from job description that are absent (IN CV LANGUAGE), e.g., 'Docker', 'CI/CD', 'Agile'"],
        "recommendations": ["Specific placement suggestions (IN CV LANGUAGE), e.g., 'Add Docker keyword to DevOps project description in Experience section'"]
    }}
}}

IMPORTANT Guidelines for field_suggestions:
- Provide 8-15 high-impact, field-targeted suggestions (prioritize critical/high severity first)
- Each suggestion must target a SPECIFIC field with exact fieldPath
- Use fieldPath array notation:
  * Simple fields: ["summary"] or ["contact", "email"]
  * Array items: ["experience", 0, "description"] means experience[0].description
  * Nested fields: ["experience", 1, "title"] means experience[1].title
- Include the fieldId (like "exp_1", "edu_2") to identify the entry
- originalValue must EXACTLY match current value in CV (word-for-word, IN ORIGINAL LANGUAGE)
- improvedValue must be COMPLETE replacement text, ready to use as-is (IN CV LANGUAGE)
- Focus on changes that matter: quantification, action verbs, clarity, keywords, impact
- Don't waste suggestions on already-strong content (only suggest if score <8/10)
- **MAINTAIN THE SAME LANGUAGE AS THE CV IN ALL TEXT FIELDS**

REMEMBER: 
- **Average CVs score 5-6/10, not 8/10**
- **Only exceptional CVs deserve 9-10/10**
- **Be specific with exact field locations and complete replacement text**
- **Prioritize high-severity issues that actually hurt job prospects**
- **Every criticism must be constructive with concrete fix**
- **Compare against real market standards, not idealized perfection**
- **🌍 ALL RECOMMENDATIONS MUST BE IN THE SAME LANGUAGE AS THE CV 🌍**
"""
    
    try:
        # Generate content with or without images
        # if cv_images and len(cv_images) > 0:
        #     content_parts = [prompt]
        #     for i, img in enumerate(cv_images[:3]):
        #         if isinstance(img, str):
        #             img = Image.open(img)
        #         content_parts.append(img)
        #     print(f"   🖼️  Sending {len(content_parts) - 1} images to Gemini for visual analysis...")
        #     response = model.generate_content(content_parts)
        # else:
        print("sending text-only prompt to Gemini...")
        response = chat(model="deepseek-v3.1:671b-cloud", messages=[{"role": "user", "content": prompt}])
        
        
        response_text = response['message']['content'].strip()
        
        # Remove any markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()
        
        parsed = json.loads(response_text)
        
        # Convert to frontend-compatible structure
        ats_analysis = parsed.get("ats_analysis", {})
        
        # Validate keywords (similar to original implementation)
        cv_text = json.dumps(structured_cv).lower()
        validated_keyword_matches = []
        validated_missing_keywords = []
        
        def is_keyword_in_text(keyword, text):
            escaped_keyword = re.escape(keyword.lower())
            pattern = rf'\b{escaped_keyword}\b'
            return bool(re.search(pattern, text.lower()))
        
        for keyword in ats_analysis.get("keyword_matches", []):
            if is_keyword_in_text(keyword, cv_text):
                validated_keyword_matches.append(keyword)
            else:
                validated_missing_keywords.append(keyword)
        
        for keyword in ats_analysis.get("missing_keywords", []):
            if not is_keyword_in_text(keyword, cv_text):
                validated_missing_keywords.append(keyword)
            else:
                validated_keyword_matches.append(keyword)
        
        validated_keyword_matches = list(set(validated_keyword_matches))
        validated_missing_keywords = list(set(validated_missing_keywords))
        
        analysis_result = {
            'status': 'success',
            'analysis': {
                'overall_score': int(parsed["general"]["overall_score"] * 10),
                'ats_score': ats_analysis.get("relevance_score", int((parsed["content"]["score"] + parsed["formatting"]["score"]) * 5)),
                'readability_score': 80 + (parsed["formatting"]["score"] - 5) * 4,
                'summary': parsed["general"]["summary"],
                'critical_issues': parsed["formatting"]["issues"] + parsed["content"]["weaknesses"],
                'field_suggestions': parsed.get("field_suggestions", []),  # NEW: Field-targeted suggestions
                'section_analysis': parsed.get("sections", []),
                'quick_wins': parsed.get("quick_wins", []),
                'top_priorities': parsed["general"]["top_priorities"],
                'grammar_and_clarity': {"issues": parsed["formatting"]["issues"]},
                'job_match_analysis': {
                    "relevance_score": ats_analysis.get("relevance_score", parsed["content"]["score"] * 10),
                    "keyword_matches": validated_keyword_matches,
                    "missing_keywords": validated_missing_keywords,
                    "recommendations": ats_analysis.get("recommendations", [])
                },
                'global_analysis': {
                    "formatting_score": parsed["formatting"]["score"],
                    "content_score": parsed["content"]["score"]
                }
            }
        }
        
        print(f"✅ Generated {len(parsed.get('field_suggestions', []))} field-targeted suggestions")
        return analysis_result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {str(e)}")
        print(f"Raw response: {response_text[:300]}")
        return {
            "status": "error",
            "message": "Failed to parse Gemini response",
            "raw_output": response_text[:300]
        }
    
    except Exception as e:
        print(f"❌ Gemini API error: {str(e)}")
        return {
            "status": "error",
            "message": f"Gemini API error: {str(e)}"
        }
