"""
LaTeX CV Generator
Converts structured CV data to professional LaTeX document using LLM
"""
import os
import subprocess
import tempfile
import json
from typing import Dict, List, Any, Optional
from ollama import chat


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters"""
    if not text:
        return ""
    
    # First, handle backslash (must be first to avoid double-escaping)
    result = text.replace('\\', r'\textbackslash{}')
    
    # Replace special characters
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        # Handle various dash characters
        '–': '--',  # en-dash
        '—': '---',  # em-dash
        '−': '-',    # minus sign
    }
    
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    return result


def format_date_range(start_date: Optional[str], end_date: Optional[str]) -> str:
    """Format date range for LaTeX"""
    if not start_date and not end_date:
        return ""
    
    if not end_date or end_date.lower() in ['present', 'current', 'now']:
        return f"{escape_latex(start_date or '')} -- Present"
    
    return f"{escape_latex(start_date or '')} -- {escape_latex(end_date or '')}"


def generate_latex_cv(structured_cv: Dict[str, Any]) -> str:
    """
    Generate a professional LaTeX CV from structured data using LLM
    The LLM generates the complete LaTeX code dynamically
    """
    
    print(f"🤖 Using LLM to generate LaTeX code...")
    print(f"🔍 CV Data keys: {list(structured_cv.keys())}")
    
    # Convert CV data to JSON string for the prompt
    cv_json = json.dumps(structured_cv, indent=2, ensure_ascii=False)
    
    prompt = f"""You are a LaTeX expert. Generate a professional, ATS-friendly CV using the moderncv document class.

**IMPORTANT REQUIREMENTS:**
1. Your response should ONLY contain the complete LaTeX code - no explanations, no markdown, no comments outside the LaTeX
2. Use the moderncv document class with 'banking' style and 'blue' color
3. Fit EVERYTHING into ONE PAGE using aggressive margins and compact spacing
4. Skip any field that says "Not provided" or is empty
5. Preserve the original language of the CV content (detect from the data)
6. Use proper LaTeX escaping for special characters: & % $ # _ {{ }} ~ ^
7. For URLs, extract usernames (e.g., github.com/username → username)
8. Use \\small font size and compact item spacing

**CV DATA:**
```json
{cv_json}
```

**LAYOUT REQUIREMENTS:**
- Margins: top=0.9cm, bottom=0.9cm, left=1.2cm, right=1.2cm
- Font: 10pt with \\small for body
- Spacing: \\baselinestretch{{0.95}}, \\setlength{{\\parskip}}{{0pt}}
- Hints column: \\setlength{{\\hintscolumnwidth}}{{2.2cm}}

**SECTIONS TO INCLUDE (if data exists):**
1. Professional Summary (if not "Not provided")
2. Professional Experience (use \\cventry with date, title, company, location, description)
3. Education (use \\cventry)
4. Skills (use \\cvitem or \\cvlistitem)
5. Projects (use \\cventry)
6. Certifications (compact format: date, name, issuer)
7. Awards (if present)
8. Activities/Volunteer (compact format)
9. Publications (if present)

**FORMATTING RULES:**
- Dates: Use "startDate -- endDate" or "startDate -- Present"
- Bullets: Use \\begin{{itemize}} \\item ... \\end{{itemize}} with \\setlength{{\\itemsep}}{{0pt}}
- Skip location if it's "Not provided"
- For contact info: \\name, \\email, \\phone, \\address, \\social[linkedin], \\social[github], \\homepage

**RESPONSE FORMAT:**
Return ONLY the complete LaTeX document starting with \\documentclass and ending with \\end{{document}}. No markdown code blocks, no explanations."""

    try:
        # Call LLM to generate LaTeX
        print("📡 Calling DeepSeek LLM...")
        response = chat(
            model="deepseek-v3.1:671b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )
        
        latex_code = response['message']['content'].strip()
        
        # Remove markdown code blocks if LLM adds them anyway
        if latex_code.startswith('```'):
            # Extract content between ```latex and ```
            import re
            match = re.search(r'```(?:latex)?\s*(.*?)\s*```', latex_code, re.DOTALL)
            if match:
                latex_code = match.group(1).strip()
        
        print(f"✅ Generated LaTeX code ({len(latex_code)} characters)")
        print(f"📝 First 200 chars: {latex_code[:200]}...")
        
        return latex_code
        
    except Exception as e:
        print(f"❌ Error generating LaTeX with LLM: {str(e)}")
        print("⚠️ Falling back to template-based generation...")
        
        # Fallback to basic template if LLM fails
        return generate_latex_cv_fallback(structured_cv)


def generate_latex_cv_fallback(structured_cv: Dict[str, Any]) -> str:
    """
    Fallback template-based LaTeX generator in case LLM fails
    """
    print("🔄 Using fallback template...")
    
    contact = structured_cv.get('contact', {})
    name = escape_latex(contact.get('name', 'Candidate'))
    email = escape_latex(contact.get('email', ''))
    phone = escape_latex(contact.get('phone', ''))
    location = escape_latex(contact.get('location', ''))
    linkedin = contact.get('linkedin', '')
    github = contact.get('github', '')
    website = contact.get('website', '')
    portfolio = contact.get('portfolio', '')
    
    # Debug: Log ALL contact fields to see what we have
    print(f"🔍 Contact info (raw): {contact}")
    print(f"🔍 Contact fields - Name: {name}")
    print(f"🔍 Contact fields - Email: {email}")
    print(f"🔍 Contact fields - Phone: {phone}")
    print(f"🔍 Contact fields - Location: {location}")
    print(f"🔍 Contact fields - LinkedIn: {linkedin}")
    print(f"🔍 Contact fields - GitHub: {github}")
    print(f"🔍 Contact fields - Website: {website}")
    print(f"🔍 Contact fields - Portfolio: {portfolio}")
    
    # Start LaTeX document with compact settings for one page
    latex = r"""\documentclass[10pt,a4paper,sans]{moderncv}

% Modern CV style and color
\moderncvstyle{banking}
\moderncvcolor{blue}

% Character encoding
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

% Aggressive page margins to help fit into one page
\usepackage[top=0.9cm, bottom=0.9cm, left=1.2cm, right=1.2cm]{geometry}

% Reduce spacing and font stretch
\setlength{\hintscolumnwidth}{2.2cm}
\setlength{\separatorcolumnwidth}{0.02cm}
\setlength{\parskip}{0pt}
\renewcommand{\baselinestretch}{0.95}

% Personal information
"""
    
    latex += f"\\name{{{name}}}{{}}\n"
    
    if email:
        latex += f"\\email{{{email}}}\n"
    if phone:
        latex += f"\\phone[mobile]{{{phone}}}\n"
    if location:
        latex += f"\\address{{{location}}}{{}}{{}}\n"
    if linkedin and linkedin != 'Not provided':
        # Extract username from LinkedIn URL or use as-is
        linkedin_clean = linkedin.replace('https://www.linkedin.com/in/', '').replace('https://linkedin.com/in/', '').replace('http://www.linkedin.com/in/', '').replace('http://linkedin.com/in/', '').rstrip('/')
        latex += f"\\social[linkedin]{{{linkedin_clean}}}\n"
        print(f"✅ Added LinkedIn: {linkedin_clean}")
    if github and github != 'Not provided':
        # Extract username from GitHub URL or use as-is
        github_clean = github.replace('https://github.com/', '').replace('http://github.com/', '').replace('https://www.github.com/', '').replace('http://www.github.com/', '').rstrip('/')
        latex += f"\\social[github]{{{github_clean}}}\n"
        print(f"✅ Added GitHub: {github_clean}")
    if website and website != 'Not provided':
        # Clean website URL
        website_clean = website.replace('https://', '').replace('http://', '').rstrip('/')
        latex += f"\\homepage{{{website_clean}}}\n"
        print(f"✅ Added Website: {website_clean}")
    if portfolio and portfolio != 'Not provided':
        # Handle portfolio as homepage if website is not set
        portfolio_clean = portfolio.replace('https://', '').replace('http://', '').rstrip('/')
        if not website or website == 'Not provided':
            latex += f"\\homepage{{{portfolio_clean}}}\n"
            print(f"✅ Added Portfolio as homepage: {portfolio_clean}")
    
    latex += r"""
\begin{document}
\makecvtitle

"""
    # Make the body slightly smaller to fit more content on a single page
    latex += "\\small\n"
    
    # Summary
    summary = structured_cv.get('summary', '')
    if summary and summary != 'Not provided' and summary.strip():
        latex += r"\section{Professional Summary}" + "\n"
        latex += escape_latex(summary) + "\n\n"
    
    # Experience
    experiences = structured_cv.get('experience', [])
    if experiences:
        latex += r"\section{Professional Experience}" + "\n"
        for exp in experiences:
            title = escape_latex(exp.get('title', ''))
            company = escape_latex(exp.get('company', ''))
            location = escape_latex(exp.get('location', ''))
            date_range = format_date_range(exp.get('startDate'), exp.get('endDate'))
            description = exp.get('description', '')
            bullets = exp.get('bullets', [])
            
            # Skip if location is "Not provided"
            if location == 'Not provided':
                location = ''
            
            if title or company:
                latex += f"\\cventry{{{date_range}}}{{{title}}}{{{company}}}{{{location}}}{{}}{{\n"
                
                # Only add description if it's not "Not provided" and not empty
                if description and description != 'Not provided' and description.strip():
                    # Don't add description if we have bullets (to save space)
                    if not bullets:
                        latex += escape_latex(description) + "\n"
                
                # Add bullets, filtering out "Not provided"
                if bullets:
                    valid_bullets = [b for b in bullets if b and b != 'Not provided' and b.strip()]
                    if valid_bullets:
                        latex += "\\begin{itemize}\\setlength{\\itemsep}{-2pt}\n"  # Compact spacing
                        for bullet in valid_bullets:
                            latex += f"  \\item {escape_latex(bullet)}\n"
                        latex += "\\end{itemize}\n"
                
                latex += "}\n\n"
    
    # Education
    education = structured_cv.get('education', [])
    if education:
        latex += r"\section{Education}" + "\n"
        for edu in education:
            degree = escape_latex(edu.get('degree', ''))
            institution = escape_latex(edu.get('institution', ''))
            location = escape_latex(edu.get('location', ''))
            date_range = format_date_range(edu.get('startDate'), edu.get('endDate'))
            description = edu.get('description', '')
            
            # Skip if location is "Not provided"
            if location == 'Not provided':
                location = ''
            
            if degree or institution:
                latex += f"\\cventry{{{date_range}}}{{{degree}}}{{{institution}}}{{{location}}}{{}}{{"
                
                # Only add description if it's not "Not provided" and not empty
                if description and description != 'Not provided' and description.strip():
                    latex += escape_latex(description)
                
                latex += "}\n\n"
    
    # Projects
    projects = structured_cv.get('projects', [])
    if projects:
        latex += r"\section{Projects}" + "\n"
        for proj in projects:
            name = escape_latex(proj.get('name', ''))
            description = proj.get('description', '')
            technologies = proj.get('technologies', [])
            
            if name:
                latex += f"\\cvitem{{{name}}}{{"
                
                # Only add description if it's not "Not provided" and not empty
                if description and description != 'Not provided' and description.strip():
                    latex += escape_latex(description)
                
                # Add technologies if available
                if technologies:
                    valid_techs = [t for t in technologies if t and t != 'Not provided']
                    if valid_techs:
                        tech_str = ', '.join([escape_latex(t) for t in valid_techs])
                        if description and description != 'Not provided' and description.strip():
                            latex += " "
                        latex += f"\\textit{{{tech_str}}}"
                
                latex += "}\n"
    latex += "\n"
    
    # Skills - more compact
    skills = structured_cv.get('skills', {})
    if skills:
        latex += r"\section{Technical Skills}" + "\n"
        for category, skill_list in skills.items():
            if skill_list:
                valid_skills = [s for s in skill_list if s and s != 'Not provided']
                if valid_skills:
                    category_name = category.replace('_', ' ').title()
                    skills_str = ', '.join([escape_latex(s) for s in valid_skills])
                    latex += f"\\cvitem{{{escape_latex(category_name)}}}{{{skills_str}}}\n"
        latex += "\n"
    
    # Certifications - more compact
    certifications = structured_cv.get('certifications', [])
    if certifications:
        print(f"🔍 Processing {len(certifications)} certifications")
        latex += r"\section{Certifications}" + "\n"
        for cert in certifications:
            name = cert.get('name', '')
            issuer = cert.get('issuer', '')
            date = cert.get('date', '')
            credential = cert.get('credential', '')
            
            print(f"  📜 Cert: {name}, Date: {date}, Issuer: {issuer}")
            
            # Skip if name is not provided
            if not name or name == 'Not provided':
                continue
            
            # Build compact entry
            name_escaped = escape_latex(name)
            issuer_escaped = escape_latex(issuer) if issuer and issuer != 'Not provided' else ''
            date_escaped = escape_latex(date) if date and date != 'Not provided' else ''
            
            if issuer_escaped and date_escaped:
                latex += f"\\cvitem{{{date_escaped}}}{{{name_escaped} -- {issuer_escaped}}}\n"
            elif issuer_escaped:
                latex += f"\\cvitem{{}}{{{name_escaped} -- {issuer_escaped}}}\n"
            elif date_escaped:
                latex += f"\\cvitem{{{date_escaped}}}{{{name_escaped}}}\n"
            else:
                latex += f"\\cvitem{{}}{{{name_escaped}}}\n"
        latex += "\n"

    # Activities & Volunteer (kept compact because these are important for entry-level)
    activities = structured_cv.get('activities', [])
    volunteer = structured_cv.get('volunteer', [])
    if activities or volunteer:
        latex += r"\section{Activities \& Volunteer}" + "\n"
        # Activities
        for a in activities:
            title = escape_latex(a.get('title', ''))
            org = escape_latex(a.get('organization', ''))
            date_range = format_date_range(a.get('startDate'), a.get('endDate'))
            description = a.get('description', '')
            if not title and not org:
                continue
            header = ' - '.join(filter(None, [title, org]))
            latex += f"\\cvitem{{{date_range}}}{{{escape_latex(header)}}}"
            if description and description != 'Not provided' and description.strip():
                latex += f"{{\\small {escape_latex(description)}}}"
            latex += "\n"
        # Volunteer entries (often similar structure)
        for v in volunteer:
            role = escape_latex(v.get('role', v.get('title', '')))
            org = escape_latex(v.get('organization', v.get('company', '')))
            date_range = format_date_range(v.get('startDate'), v.get('endDate'))
            description = v.get('description', '')
            if not role and not org:
                continue
            header = ' - '.join(filter(None, [role, org]))
            latex += f"\\cvitem{{{date_range}}}{{{escape_latex(header)}}}"
            if description and description != 'Not provided' and description.strip():
                latex += f"{{\\small {escape_latex(description)}}}"
            latex += "\n"
        latex += "\n"
    # for entry-level positions
    
    # End document
    latex += "\\normalsize\n\\end{document}"
    
    return latex


def compile_latex_to_pdf(latex_content: str, output_dir: str = None) -> tuple[bool, str, str]:
    """
    Compile LaTeX content to PDF using xelatex
    
    Returns:
        (success: bool, pdf_path: str, error_message: str)
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    # Create temporary LaTeX file
    timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    tex_filename = f"cv_{timestamp}.tex"
    tex_path = os.path.join(output_dir, tex_filename)
    
    try:
        # Write LaTeX content to file
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"📝 LaTeX file created: {tex_path}")
        
        # Compile with xelatex (run twice for references)
        for run in range(2):
            print(f"🔄 Compiling LaTeX (pass {run + 1}/2)...")
            result = subprocess.run(
                ['xelatex', '-interaction=nonstopmode', '-output-directory', output_dir, tex_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if PDF was created (even with warnings/errors, xelatex may still produce output)
            pdf_path = tex_path.replace('.tex', '.pdf')
            if os.path.exists(pdf_path):
                # PDF exists, consider it success even if there were warnings
                print(f"✅ PDF generated on pass {run + 1}")
                continue
            elif result.returncode != 0:
                # No PDF and error occurred
                error_msg = f"LaTeX compilation failed on pass {run + 1}:\n{result.stdout}\n{result.stderr}"
                print(f"❌ {error_msg}")
                return False, "", error_msg
        
        # Check if PDF was created after both passes
        pdf_path = tex_path.replace('.tex', '.pdf')
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"✅ PDF generated successfully: {pdf_path} ({file_size} bytes)")
            
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out', '.tex']:
                aux_file = tex_path.replace('.tex', ext)
                if os.path.exists(aux_file):
                    try:
                        os.remove(aux_file)
                    except:
                        pass
            
            return True, pdf_path, ""
        else:
            error_msg = "PDF file was not generated"
            print(f"❌ {error_msg}")
            return False, "", error_msg
    
    except FileNotFoundError:
        error_msg = "xelatex command not found. Please ensure LaTeX is installed and xelatex is in your PATH."
        print(f"❌ {error_msg}")
        return False, "", error_msg
    
    except subprocess.TimeoutExpired:
        error_msg = "LaTeX compilation timed out after 30 seconds"
        print(f"❌ {error_msg}")
        return False, "", error_msg
    
    except Exception as e:
        error_msg = f"Error during LaTeX compilation: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, "", error_msg
