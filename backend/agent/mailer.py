import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from dotenv import load_dotenv
import re
import requests

# Load .env from parent directory (backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Email configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
SENDER_NAME = os.getenv("SENDER_NAME", "AI Interview Coach")


def parse_report(report_text: str) -> dict:
    """
    Parses the interview report text and extracts structured data
    Handles the specific format from the agent's prompts
    Supports both English and French reports
    """
    # First, clean up the text - remove any preamble
    clean_text = report_text
    
    # Detect language based on keywords
    is_french = "Score Global" in report_text or "Points Forts Clés" in report_text
    
    # Remove "Thank you for your time today" type phrases (English and French)
    preamble_phrases = [
        "Thank you for your time today. That concludes the interview questions.",
        "That concludes the interview questions.",
        "Thank you for your time today.",
        "Merci pour votre temps aujourd'hui. Ceci conclut les questions d'entretien.",
        "Ceci conclut les questions d'entretien.",
        "Merci pour votre temps aujourd'hui."
    ]
    for phrase in preamble_phrases:
        clean_text = clean_text.replace(phrase, "")
    
    # Remove closing phrases (English and French)
    closing_phrases = [
        "Best of luck with your interview preparation. This session is now complete.",
        "Best of luck with your interview preparation.",
        "This session is now complete.",
        "Bonne chance pour la préparation de vos entretiens. Cette session est maintenant terminée.",
        "Bonne chance pour la préparation de vos entretiens.",
        "Cette session est maintenant terminée."
    ]
    for phrase in closing_phrases:
        if phrase in clean_text:
            clean_text = clean_text[:clean_text.find(phrase)]
    
    clean_text = clean_text.strip()
    
    data = {
        "overall_score": "",
        "score_justification": "",
        "strengths": [],
        "weaknesses": [],
        "communication": {},
        "recommendations": [],
        "role_fit": "",
        "final_note": "",
        "language": "french" if is_french else "english"
    }
    
    if is_french:
        # French report parsing - matches exact format from prompts.py
        # Format: **Score Global :** (space before colon, colon inside bold)
        # Also handles unformatted reports without bold markers
        
        # Extract overall score - handle both formatted and unformatted
        score_match = re.search(r'\*{0,2}Score Global\s*:\s*\*{0,2}\s*(\d+/10)\s*[-–]\s*(.+?)(?=\n|$)', clean_text)
        if score_match:
            data["overall_score"] = score_match.group(1)
            data["score_justification"] = score_match.group(2).strip()
        
        # Extract strengths - handle both formatted and unformatted
        strengths_section = re.search(r"\*{0,2}Points Forts Clés\s*:\s*\*{0,2}(.*?)(?:\*{0,2}Axes d['\u2019]Amélioration\s*:\s*\*{0,2})", clean_text, re.DOTALL)
        if strengths_section:
            strengths_text = strengths_section.group(1)
            strengths = re.findall(r'-\s*(.+?)(?=\n-|\n\*|\n[A-ZÉÈÊË]|\Z)', strengths_text, re.DOTALL)
            data["strengths"] = [s.strip() for s in strengths if s.strip()]
        
        # Extract weaknesses/areas for development
        weaknesses_section = re.search(r"\*{0,2}Axes d['\u2019]Amélioration\s*:\s*\*{0,2}(.*?)(?:\*{0,2}Évaluation de la Communication et de la Prestation\s*:\s*\*{0,2})", clean_text, re.DOTALL)
        if weaknesses_section:
            weaknesses_text = weaknesses_section.group(1)
            # Try to match formatted weaknesses first
            weaknesses = re.findall(r'-\s*\*{0,2}(.+?)\*{0,2}\s*:\s*(.+?)(?=\n-|\n\*|\n[A-ZÉÈÊË]|\Z)', weaknesses_text, re.DOTALL)
            if weaknesses:
                for weakness in weaknesses:
                    data["weaknesses"].append({
                        "title": weakness[0].strip(),
                        "description": weakness[1].strip()
                    })
            else:
                # If no formatted weaknesses, treat as bullet points
                simple_weaknesses = re.findall(r'-\s*(.+?)(?=\n-|\n\*|\n[A-ZÉÈÊË]|\Z)', weaknesses_text, re.DOTALL)
                for weak in simple_weaknesses:
                    data["weaknesses"].append({
                        "title": "Point d'amélioration",
                        "description": weak.strip()
                    })
        
        # Extract communication assessment
        comm_fields = ["Confiance", "Clarté", "Rythme et Fluidité", "Enthousiasme", "Professionnalisme"]
        for field in comm_fields:
            match = re.search(f'-\s*{re.escape(field)}\s*:\s*(.+?)(?=\n-|\n\*|\n[A-ZÉÈÊË]|\Z)', clean_text, re.DOTALL)
            if match:
                data["communication"][field.lower().replace(" ", "_").replace("é", "e")] = match.group(1).strip()
        
        # Extract recommendations
        recommendations_section = re.search(r"\*{0,2}Recommandations d['\u2019]Amélioration\s*:\s*\*{0,2}(.*?)(?:\*{0,2}Évaluation de l['\u2019]Adéquation au Poste\s*:\s*\*{0,2})", clean_text, re.DOTALL)
        if recommendations_section:
            recs_text = recommendations_section.group(1)
            recs = re.findall(r'-\s*(.+?)(?=\n-|\n\*|\n[A-ZÉÈÊË]|\Z)', recs_text, re.DOTALL)
            data["recommendations"] = [r.strip() for r in recs if r.strip()]
        
        # Extract role fit
        role_fit_match = re.search(r"\*{0,2}Évaluation de l['\u2019]Adéquation au Poste\s*:\s*\*{0,2}\s*(.+?)(?=\*{0,2}Note Finale\s*:\s*\*{0,2}|\n\n---|\Z)", clean_text, re.DOTALL)
        if role_fit_match:
            data["role_fit"] = role_fit_match.group(1).strip()
        
        # Extract final note (optional)
        final_note_match = re.search(r"\*{0,2}Note Finale\s*:\s*\*{0,2}\s*(.+?)(?=\n\n---|\nBonne chance|\Z)", clean_text, re.DOTALL)
        if final_note_match:
            data["final_note"] = final_note_match.group(1).strip()
    else:
        # English report parsing
        # Extract overall score
        score_match = re.search(r'\*\*Overall Score\s*:\*\*\s*(\d+/10)\s*[-–]\s*(.+?)(?=\n|$)', clean_text)
        if score_match:
            data["overall_score"] = score_match.group(1)
            data["score_justification"] = score_match.group(2).strip()
        
        # Extract strengths
        strengths_section = re.search(r'\*\*Key Strengths\s*:\*\*(.*?)\*\*Areas for Development\s*:\*\*', clean_text, re.DOTALL)
        if strengths_section:
            strengths_text = strengths_section.group(1)
            strengths = re.findall(r'-\s*(.+?)(?=\n-|\n\*\*|\Z)', strengths_text, re.DOTALL)
            data["strengths"] = [s.strip() for s in strengths if s.strip()]
        
        # Extract weaknesses/areas for development
        weaknesses_section = re.search(r'\*\*Areas for Development\s*:\*\*(.*?)\*\*Communication & Delivery Assessment\s*:\*\*', clean_text, re.DOTALL)
        if weaknesses_section:
            weaknesses_text = weaknesses_section.group(1)
            weaknesses = re.findall(r'-\s*\*\*(.+?)\*\*\s*:\s*(.+?)(?=\n-|\n\*\*|\Z)', weaknesses_text, re.DOTALL)
            for weakness in weaknesses:
                data["weaknesses"].append({
                    "title": weakness[0].strip(),
                    "description": weakness[1].strip()
                })
        
        # Extract communication assessment
        comm_fields = ["Confidence", "Clarity", "Pace & Fluency", "Enthusiasm", "Professionalism"]
        for field in comm_fields:
            match = re.search(f'-\s*{field}\s*:\s*(.+?)(?=\n-|\n\*\*|\Z)', clean_text, re.DOTALL)
            if match:
                data["communication"][field.lower().replace(" & ", "_").replace(" ", "_")] = match.group(1).strip()
        
        # Extract recommendations
        recommendations_section = re.search(r'\*\*Recommendations for Improvement\s*:\*\*(.*?)\*\*Role Fit Assessment\s*:\*\*', clean_text, re.DOTALL)
        if recommendations_section:
            recs_text = recommendations_section.group(1)
            recs = re.findall(r'-\s*(.+?)(?=\n-|\n\*\*|\Z)', recs_text, re.DOTALL)
            data["recommendations"] = [r.strip() for r in recs if r.strip()]
        
        # Extract role fit
        role_fit_match = re.search(r'\*\*Role Fit Assessment\s*:\*\*\s*(.+?)(?=\*\*Final Note\s*:\*\*|\n\n---|\Z)', clean_text, re.DOTALL)
        if role_fit_match:
            data["role_fit"] = role_fit_match.group(1).strip()
        
        # Extract final note (optional)
        final_note_match = re.search(r'\*\*Final Note\s*:\*\*\s*(.+?)(?=\n\n---|\Z)', clean_text, re.DOTALL)
        if final_note_match:
            data["final_note"] = final_note_match.group(1).strip()
    
    return data


def create_html_email(candidate_name: str, job_title: str, report_data: dict) -> str:
    """
    Creates a beautiful HTML email template for the interview report
    Supports both English and French
    """
    from datetime import datetime
    import locale
    
    is_french = report_data.get("language") == "french"
    
    # Translations
    if is_french:
        translations = {
            "title": "Rapport de Performance d'Entretien",
            "subtitle": "Analyse d'Entretien Assistée par IA",
            "candidate": "Candidat",
            "position": "Poste",
            "date": "Date",
            "overall_performance": "Performance Globale",
            "key_strengths": "✨ Points Forts Clés",
            "areas_development": "📈 Axes d'Amélioration",
            "communication_delivery": "🎤 Communication et Prestation",
            "confidence": "Confiance",
            "clarity": "Clarté",
            "pace_fluency": "Rythme et Fluidité",
            "enthusiasm": "Enthousiasme",
            "professionalism": "Professionnalisme",
            "recommendations": "💡 Recommandations d'Amélioration",
            "role_fit": "🎯 Évaluation de l'Adéquation au Poste",
            "footer_text": "Ce rapport a été généré par AI Interview Coach",
            "footer_subtext": "Conservez ce rapport pour vos dossiers et votre future préparation d'entretien"
        }
    else:
        translations = {
            "title": "Interview Performance Report",
            "subtitle": "AI-Powered Interview Analysis",
            "candidate": "Candidate",
            "position": "Position",
            "date": "Date",
            "overall_performance": "Overall Performance",
            "key_strengths": "✨ Key Strengths",
            "areas_development": "📈 Areas for Development",
            "communication_delivery": "🎤 Communication & Delivery",
            "confidence": "Confidence",
            "clarity": "Clarity",
            "pace_fluency": "Pace & Fluency",
            "enthusiasm": "Enthusiasm",
            "professionalism": "Professionalism",
            "recommendations": "💡 Recommendations for Improvement",
            "role_fit": "🎯 Role Fit Assessment",
            "footer_text": "This report was generated by AI Interview Coach",
            "footer_subtext": "Keep this report for your records and future interview preparation"
        }
    
    # Generate strengths HTML
    strengths_html = ""
    for strength in report_data["strengths"]:
        strengths_html += f"""
        <li style="margin-bottom: 12px; line-height: 1.6;">{strength}</li>
        """
    
    # Generate weaknesses HTML
    weaknesses_html = ""
    for weakness in report_data["weaknesses"]:
        weaknesses_html += f"""
        <div style="margin-bottom: 20px;">
            <strong style="color: #2563eb;">{weakness['title']}:</strong>
            <p style="margin: 8px 0 0 0; line-height: 1.6;">{weakness['description']}</p>
        </div>
        """
    
    # Generate communication assessment HTML
    communication_html = ""
    # Map the keys to translated labels
    comm_key_map = {
        "confiance": translations["confidence"],
        "clarte": translations["clarity"],
        "rythme_et_fluidite": translations["pace_fluency"],
        "enthousiasme": translations["enthusiasm"],
        "professionnalisme": translations["professionalism"],
        # English keys
        "confidence": translations["confidence"],
        "clarity": translations["clarity"],
        "pace_fluency": translations["pace_fluency"],
        "enthusiasm": translations["enthusiasm"],
        "professionalism": translations["professionalism"]
    }
    
    for key, value in report_data["communication"].items():
        label = comm_key_map.get(key, key.replace("_", " ").title())
        communication_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; font-weight: 600; color: #374151;">{label}:</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">{value}</td>
        </tr>
        """
    
    # Generate recommendations HTML
    recommendations_html = ""
    for rec in report_data["recommendations"]:
        recommendations_html += f"""
        <li style="margin-bottom: 12px; line-height: 1.6;">{rec}</li>
        """
    
    # Score color based on value
    score_value = int(report_data["overall_score"].split("/")[0]) if report_data["overall_score"] else 5
    if score_value >= 8:
        score_color = "#10b981"  # Green
    elif score_value >= 6:
        score_color = "#f59e0b"  # Orange
    else:
        score_color = "#ef4444"  # Red
    
    # Format date based on language
    if is_french:
        try:
            locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
        except:
            pass
        date_str = datetime.now().strftime("%d %B %Y")
    else:
        date_str = datetime.now().strftime("%B %d, %Y")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{translations["title"]}</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
        <table role="presentation" style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 40px 20px;">
                    <table role="presentation" style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <!-- Header -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; border-radius: 12px 12px 0 0; text-align: center;">
                                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">{translations["title"]}</h1>
                                <p style="margin: 10px 0 0 0; color: #e0e7ff; font-size: 16px;">{translations["subtitle"]}</p>
                            </td>
                        </tr>
                        
                        <!-- Candidate Info -->
                        <tr>
                            <td style="padding: 30px;">
                                <div style="background-color: #f9fafb; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;">
                                    <p style="margin: 0 0 8px 0; color: #6b7280; font-size: 14px;">{translations["candidate"]}</p>
                                    <p style="margin: 0 0 12px 0; color: #111827; font-size: 18px; font-weight: 600;">{candidate_name}</p>
                                    <p style="margin: 0 0 8px 0; color: #6b7280; font-size: 14px;">{translations["position"]}</p>
                                    <p style="margin: 0 0 12px 0; color: #111827; font-size: 16px; font-weight: 500;">{job_title}</p>
                                    <p style="margin: 0 0 8px 0; color: #6b7280; font-size: 14px;">{translations["date"]}</p>
                                    <p style="margin: 0; color: #111827; font-size: 16px;">{date_str}</p>
                                </div>
                            </td>
                        </tr>
                        
                        <!-- Overall Score -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #111827; font-size: 22px; font-weight: 700;">{translations["overall_performance"]}</h2>
                                <div style="text-align: center; background-color: #f9fafb; padding: 30px; border-radius: 8px;">
                                    <div style="display: inline-block; background-color: {score_color}; color: #ffffff; font-size: 48px; font-weight: 700; padding: 20px 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                                        {report_data["overall_score"]}
                                    </div>
                                    <p style="margin: 20px 0 0 0; color: #6b7280; font-size: 16px; line-height: 1.6;">{report_data["score_justification"]}</p>
                                </div>
                            </td>
                        </tr>
                        
                        <!-- Key Strengths -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #111827; font-size: 22px; font-weight: 700;">{translations["key_strengths"]}</h2>
                                <ul style="margin: 0; padding-left: 20px; color: #374151;">
                                    {strengths_html}
                                </ul>
                            </td>
                        </tr>
                        
                        <!-- Areas for Development -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #111827; font-size: 22px; font-weight: 700;">{translations["areas_development"]}</h2>
                                <div style="color: #374151;">
                                    {weaknesses_html}
                                </div>
                            </td>
                        </tr>
                        
                        <!-- Communication Assessment -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #111827; font-size: 22px; font-weight: 700;">{translations["communication_delivery"]}</h2>
                                <table role="presentation" style="width: 100%; border-collapse: collapse; background-color: #f9fafb; border-radius: 8px; overflow: hidden;">
                                    {communication_html}
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Recommendations -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #111827; font-size: 22px; font-weight: 700;">{translations["recommendations"]}</h2>
                                <ul style="margin: 0; padding-left: 20px; color: #374151;">
                                    {recommendations_html}
                                </ul>
                            </td>
                        </tr>
                        
                        <!-- Role Fit Assessment -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #111827; font-size: 22px; font-weight: 700;">{translations["role_fit"]}</h2>
                                <div style="background-color: #eff6ff; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                                    <p style="margin: 0; color: #1e40af; line-height: 1.6;">{report_data["role_fit"]}</p>
                                </div>
                            </td>
                        </tr>
                        
                        <!-- Final Note (if present) -->
                        {'<tr><td style="padding: 0 30px 30px 30px;"><div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b;"><p style="margin: 0; color: #92400e; line-height: 1.6; font-weight: 500;">' + report_data["final_note"] + '</p></div></td></tr>' if report_data.get("final_note") else ''}
                        
                        <!-- Footer -->
                        <tr>
                            <td style="padding: 30px; background-color: #f9fafb; border-radius: 0 0 12px 12px; text-align: center;">
                                <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px;">{translations["footer_text"]}</p>
                                <p style="margin: 0; color: #9ca3af; font-size: 12px;">{translations["footer_subtext"]}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    return html


async def send_interview_report_email(
    recipient_email: str,
    candidate_name: str,
    job_title: str,
    report_text: str
) -> bool:
    """
    Sends the interview report via email
    
    Args:
        recipient_email: Email address of the candidate
        candidate_name: Name of the candidate
        job_title: Job position title
        report_text: The full report text from the agent
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("❌ Email configuration missing. Please set SENDER_EMAIL and SENDER_PASSWORD in .env")
        return False
    
    try:
        # Parse the report
        report_data = parse_report(report_text)
        
        # Create HTML email
        html_content = create_html_email(candidate_name, job_title, report_data)
        
        # Determine language for subject
        is_french = report_data.get("language") == "french"
        subject = f"Rapport de Performance d'Entretien - {job_title}" if is_french else f"Interview Performance Report - {job_title}"
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        message["To"] = recipient_email
        
        # Create plain text version (fallback) - multilingual
        from datetime import datetime
        if is_french:
            text_content = f"""
Rapport de Performance d'Entretien

Candidat : {candidate_name}
Poste : {job_title}
Date : {datetime.now().strftime("%d %B %Y")}

{report_text}

---
Ce rapport a été généré par AI Interview Coach
        """
        else:
            text_content = f"""
Interview Performance Report

Candidate: {candidate_name}
Position: {job_title}
Date: {datetime.now().strftime("%B %d, %Y")}

{report_text}

---
This report was generated by AI Interview Coach
        """
        
        # Attach both versions
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        message.attach(part1)
        message.attach(part2)
        
        # Send email
        print(f"📧 Sending email to {recipient_email}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
        
        print(f"✅ Email sent successfully to {recipient_email}")
        interview_score = None
        if report_data.get("overall_score"):
            try:
                # Extract the number before the "/" (e.g., "8/10" -> 8.0)
                score_str = report_data["overall_score"].split("/")[0].strip()
                interview_score = float(score_str)
            except (ValueError, IndexError) as e:
                print(f"⚠️ Could not parse score '{report_data['overall_score']}': {e}")
        try:
            response = requests.post(
                "http://localhost:8000/api/interviews/save-report",
                json={
                    "candidate_email": recipient_email,
                    "job_title": job_title,
                    "interview_score": interview_score,
                    "conclusion": report_data["final_note"]
                }
            )
            if response.status_code == 200:
                print("✅ Interview report saved to database")
            else:
                print(f"❌ Failed to save interview report: {response.text}")
        except Exception as e:
            print(f"❌ Error saving interview report to database: {str(e)}")    
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")
        return False


def extract_report_from_message(message_content: str) -> str:
    """
    Extracts the report section from the agent's message
    Removes the concluding phrases and preamble
    """
    # Look for the report section
    report_start = message_content.find("INTERVIEW PERFORMANCE REPORT")
    if report_start == -1:
        report_start = message_content.find("**INTERVIEW PERFORMANCE REPORT**")
    
    if report_start != -1:
        # Extract from report start onwards
        report_content = message_content[report_start:]
        
        # Remove common closing phrases that come after the report
        closing_phrases = [
            "Best of luck with your interview preparation",
            "This session is now complete",
            "Good luck with your interview preparation",
            "Thank you for participating"
        ]
        
        for phrase in closing_phrases:
            if phrase in report_content:
                report_content = report_content[:report_content.find(phrase)]
        
        return report_content.strip()
    
    # If report marker not found, look for the preamble and remove it
    preamble_phrases = [
        "Thank you for your time today. That concludes the interview questions.",
        "That concludes the interview questions.",
        "Thank you for your time today."
    ]
    
    content = message_content
    for phrase in preamble_phrases:
        if phrase in content:
            # Remove the preamble phrase
            content = content.replace(phrase, "").strip()
    
    return content


async def send_verification_email(
    recipient_email: str,
    recipient_name: str,
    verification_token: str,
    frontend_url: str = "http://localhost:5173"
) -> bool:
    """
    Sends an email verification link to the user
    
    Args:
        recipient_email: Email address of the user
        recipient_name: Name of the user
        verification_token: Unique verification token
        frontend_url: Base URL of the frontend application
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("❌ Email configuration missing. Please set SENDER_EMAIL and SENDER_PASSWORD in .env")
        return False
    
    try:
        verification_link = f"{frontend_url}/verify-email?token={verification_token}"
        
        # Use "UtopiaHire" as sender name for verification emails
        verification_sender_name = "UtopiaHire"
        
        # Create HTML email
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verify Your Email</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 40px 20px;">
                        <table role="presentation" style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <!-- Header -->
                            <tr>
                                <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; border-radius: 12px 12px 0 0; text-align: center;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">Verify Your Email</h1>
                                    <p style="margin: 10px 0 0 0; color: #e0e7ff; font-size: 16px;">Welcome to UtopiaHire!</p>
                                </td>
                            </tr>
                            
                            <!-- Body -->
                            <tr>
                                <td style="padding: 40px 30px;">
                                    <p style="margin: 0 0 20px 0; color: #111827; font-size: 18px; font-weight: 600;">Hi {recipient_name},</p>
                                    <p style="margin: 0 0 20px 0; color: #374151; font-size: 16px; line-height: 1.6;">
                                        Thank you for registering with UtopiaHire! We're excited to have you on board.
                                    </p>
                                    <p style="margin: 0 0 30px 0; color: #374151; font-size: 16px; line-height: 1.6;">
                                        To complete your registration and start preparing for interviews, please verify your email address by clicking the button below:
                                    </p>
                                    
                                    <!-- Verification Button -->
                                    <div style="text-align: center; margin: 30px 0;">
                                        <a href="{verification_link}" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff !important; text-decoration: none; padding: 16px 40px; border-radius: 8px; font-size: 16px; font-weight: 600; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: transform 0.2s;">
                                            <span style="color: #ffffff;">Verify Email Address</span>
                                        </a>
                                    </div>
                                    
                                    <p style="margin: 30px 0 0 0; color: #6b7280; font-size: 14px; line-height: 1.6;">
                                        Or copy and paste this link into your browser:
                                    </p>
                                    <p style="margin: 10px 0 0 0; color: #3b82f6; font-size: 14px; word-break: break-all;">
                                        {verification_link}
                                    </p>
                                    
                                    <div style="margin-top: 30px; padding: 20px; background-color: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b;">
                                        <p style="margin: 0; color: #92400e; font-size: 14px; line-height: 1.6;">
                                            <strong>Note:</strong> This verification link will expire in 24 hours. If you didn't create an account, you can safely ignore this email.
                                        </p>
                                    </div>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px; background-color: #f9fafb; border-radius: 0 0 12px 12px; text-align: center;">
                                    <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px;">UtopiaHire</p>
                                    <p style="margin: 0; color: #9ca3af; font-size: 12px;">Powered by advanced AI to help you ace your interviews</p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        # Create plain text version (fallback)
        text_content = f"""
Verify Your Email

Hi {recipient_name},

Thank you for registering with UtopiaHire! We're excited to have you on board.

To complete your registration and start preparing for interviews, please verify your email address by clicking the link below:

{verification_link}

Note: This verification link will expire in 24 hours. If you didn't create an account, you can safely ignore this email.

---
UtopiaHire
Powered by advanced AI to help you ace your interviews
        """
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Verify Your Email - UtopiaHire"
        message["From"] = f"{verification_sender_name} <{SENDER_EMAIL}>"
        message["To"] = recipient_email
        message["Reply-To"] = SENDER_EMAIL
        message["X-Mailer"] = "UtopiaHire"
        message["X-Priority"] = "3"
        message["List-Unsubscribe"] = f"<mailto:{SENDER_EMAIL}?subject=unsubscribe>"
        
        # Attach both versions
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        message.attach(part1)
        message.attach(part2)
        
        # Send email
        print(f"📧 Sending verification email to {recipient_email}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
        
        print(f"✅ Verification email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send verification email: {str(e)}")
        return False