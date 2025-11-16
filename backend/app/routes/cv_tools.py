from fastapi import FastAPI, UploadFile, File, Form, Body ,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ..parsers.cv_parser import parse_document_with_metadata
from ..parsers.cv_structure_parser import parse_and_analyze_cv, apply_suggestion_to_structured_cv
from ..parsers.latex_generator import generate_latex_cv, compile_latex_to_pdf
import tempfile
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(tags=["CvTools"])

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# File storage directories (store JSON files under backend/data/...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CV_DATA_DIR = os.path.join(DATA_DIR, 'cv_data')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')

# Ensure directories exist
os.makedirs(CV_DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ⚠️ IMPORTANT: Set your Gemini API key here or use environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyADSqukXGi3gtDTfLB-sUMmxnI-bdY4Oa0")

"""
Removed legacy /analyze endpoint. Use POST /analyze-structured instead.
"""


@router.post("/api/analyze-structured")
async def analyze_structured(
    cv_file: UploadFile = File(None),
    cv_text: str = Form(None),
    job_description: str = Form(""),
    use_gemini: bool = Form(True)
):
    """
    Analyze CV and return structured data with field-targeted suggestions.
    This is the enhanced version that uses structured CV data.
    """
    
    print("\n" + "="*50)
    print("NEW STRUCTURED ANALYSIS REQUEST")
    print("="*50)
    
    # Determine input type
    cv_input = None
    cv_links = []
    file_info = {}
    original_file_data = None
    temp_file_path = None
    
    if cv_file:
        print(f"📄 File uploaded: {cv_file.filename}")
        try:
            _, ext = os.path.splitext(cv_file.filename)
            content = await cv_file.read()
            
            # Store original file for response
            file_base64 = base64.b64encode(content).decode('utf-8')
            original_file_data = {
                "filename": cv_file.filename,
                "content_type": cv_file.content_type,
                "data": file_base64,
                "size": len(content)
            }
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                temp_file_path = tmp.name
            
            # Extract links from the PDF (but don't parse text - let Gemini see the visual layout)
            print("� Extracting links from PDF...")
            if ext.lower() == '.pdf':
                parse_result = parse_document_with_metadata(temp_file_path)
                if parse_result:
                    cv_links = parse_result.get('links', [])
                    print(f"✅ Found {len(cv_links)} links: {cv_links}")
            
            # Use file path as input (Gemini will process the visual document)
            cv_input = temp_file_path
            
            file_info = {
                "filename": cv_file.filename,
                "file_size_bytes": len(content),
                "links_found": len(cv_links),
                "processing_method": "PDF visual analysis + extracted links"
            }
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up temp file if it was created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            return {"error": "Failed to process file. Please ensure the file is valid and try again."}
    elif cv_text:
        # Use text input
        cv_input = cv_text
        print(f"📝 Processing text content ({len(cv_text)} characters)")
    else:
        return {"error": "No CV text or file provided"}
    
    # Parse CV and analyze it in ONE API call (hybrid: PDF visual + links metadata)
    print("🔄 Analyzing CV with hybrid approach (PDF visual + extracted links)...")
    combined_result = parse_and_analyze_cv(cv_input, job_description, GEMINI_API_KEY, cv_links)
    
    # Clean up temp file
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
            print(f"🗑️  Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            print(f"⚠️  Could not delete temp file: {str(e)}")
    
    if combined_result['status'] != 'success':
        print(f"❌ Failed to parse and analyze CV: {combined_result}")
        return {"error": "Failed to parse and analyze CV. Please try again or use a different format."}
    
    structured_cv = combined_result['structured_data']
    gemini_analysis = {
        'status': 'success',
        'analysis': combined_result['analysis']
    } if use_gemini else None
    
    # Save structured CV data
    cv_data = {
        "timestamp": datetime.now().isoformat(),
        "structured_cv": structured_cv,
        "original_text": combined_result.get('original_text', ''),
        "job_description": job_description,
        "file_info": file_info if cv_file else {"source": "raw_text"}
    }
    
    output_basename = f"structured_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_filename = os.path.join(CV_DATA_DIR, output_basename)
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(cv_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved structured CV to: {output_filename}")
    except Exception as e:
        print(f"❌ Error saving structured CV: {str(e)}")
    
    # Save analysis
    if gemini_analysis and 'error' not in gemini_analysis:
        analysis_basename = f"structured_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analysis_filename = os.path.join(ANALYSIS_DIR, analysis_basename)
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(gemini_analysis, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved structured analysis to: {analysis_filename}")
    
    # Return response
    response = {
        "summary": "CV successfully analyzed with structured data!",
        "status": "success",
        "structured_cv": structured_cv,
        "original_file": original_file_data,
        "file_info": file_info if cv_file else {"source": "raw_text"}
    }
    
    if gemini_analysis:
        response["gemini_analysis"] = gemini_analysis
    
    return response


@router.get("/api/latest-structured-cv")
async def get_latest_structured_cv():
    """Get the most recently saved structured CV data"""
    try:
        json_files = [f for f in os.listdir(CV_DATA_DIR) if f.startswith('structured_cv_') and f.endswith('.json')]
        
        if not json_files:
            return {"error": "No structured CV data files found"}
        
        latest_basename = sorted(json_files)[-1]
        latest_path = os.path.join(CV_DATA_DIR, latest_basename)
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "filename": latest_basename,
            "data": data
        }
    
    except Exception as e:
        print(f"❌ Error reading structured CV: {str(e)}")
        return {"error": "Failed to read structured CV. Please try again."}


@router.post("/api/apply-suggestion")
async def apply_suggestion(
    structured_cv: dict = Body(...),
    suggestion: dict = Body(...)
):
    """
    Apply a suggestion to the structured CV data.
    
    Expected request body:
    {
        "structured_cv": { ... },
        "suggestion": {
            "suggestionId": 1,
            "targetField": "summary",
            "fieldPath": ["summary"],
            "improvedValue": "New improved text"
        }
    }
    """
    try:
        updated_cv = apply_suggestion_to_structured_cv(structured_cv, suggestion)
        
        return {
            "status": "success",
            "updated_cv": updated_cv,
            "applied_suggestion": suggestion
        }
    
    except Exception as e:
        print(f"❌ Error applying suggestion: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": "Failed to apply suggestion. Please try again."
        }




@router.post("/api/export-latex-pdf")
async def export_latex_pdf(request_data: dict = Body(...)):
    """
    Generate and compile a LaTeX CV to PDF
    
    Expected request body:
    {
        "structured_cv": { ... }
    }
    
    Returns the compiled PDF file
    """
    print("\n" + "="*50)
    print("LATEX PDF EXPORT REQUEST")
    print("="*50)
    
    try:
        # Extract structured_cv from request
        structured_cv = request_data.get('structured_cv', {})
        
        # Debug: print what we received
        print(f"📋 Received CV data: {len(str(structured_cv))} characters")
        print(f"📋 CV keys: {list(structured_cv.keys())}")
        
        # Debug certifications specifically (example of applied changes)
        if 'certifications' in structured_cv:
            print(f"📋 Certifications data: {structured_cv['certifications']}")
        
        # Generate LaTeX content
        print("📝 Generating LaTeX content...")
        latex_content = generate_latex_cv(structured_cv)
        
        # Create output directory for PDFs
        pdf_output_dir = os.path.join(DATA_DIR, 'pdfs')
        os.makedirs(pdf_output_dir, exist_ok=True)
        
        # Compile to PDF
        print("🔄 Compiling LaTeX to PDF with xelatex...")
        success, pdf_path, error_msg = compile_latex_to_pdf(latex_content, pdf_output_dir)
        
        if not success:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to compile LaTeX to PDF",
                    "error": error_msg
                }
            )
        
        # Get candidate name for filename
        contact = structured_cv.get('contact', {})
        name = contact.get('name', 'Candidate').replace(' ', '_')
        
        # Return the PDF file
        print(f"✅ Sending PDF file: {pdf_path}")
        return FileResponse(
            path=pdf_path,
            media_type='application/pdf',
            filename=f"{name}_CV.pdf",
            headers={
                "Content-Disposition": f"attachment; filename={name}_CV.pdf"
            }
        )
    
    except Exception as e:
        print(f"❌ Error exporting LaTeX PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to export PDF",
                "error": str(e)
            }
        )


@router.post("/api/get-latex-code")
async def get_latex_code(request_data: dict = Body(...)):
    """
    Generate LaTeX code for the CV without compilation to PDF
    
    Expected request body:
    {
        "structured_cv": { ... }
    }
    
    Returns the LaTeX code as JSON
    """
    print("\n" + "="*50)
    print("GET LATEX CODE REQUEST")
    print("="*50)
    
    try:
        # Extract structured_cv from request
        structured_cv = request_data.get('structured_cv', {})
        
        # Debug: print what we received
        print(f"📋 Received CV data: {len(str(structured_cv))} characters")
        print(f"📋 CV keys: {list(structured_cv.keys())}")
        
        # Generate LaTeX content
        print("📝 Generating LaTeX content...")
        latex_content = generate_latex_cv(structured_cv)
        
        # Return the LaTeX code
        print(f"✅ Returning LaTeX code ({len(latex_content)} characters)")
        return {
            "status": "success",
            "latex_code": latex_content
        }
    
    except Exception as e:
        print(f"❌ Error generating LaTeX code: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to generate LaTeX code",
                "error": str(e)
            }
        )