import torch
import re
from typing import List, Dict, Set
from gliner import GLiNER
from core.data_loader import DataLoader # Import our data loader class

print("Initializing Skill Extractor...")

# --- 1. GLiNER MODEL LOADER (SINGLETON) ---

class GLiNERSkillExtractor:
    def __init__(self, model_name="urchade/gliner_base"):
        print(f"Loading GLiNER model: {model_name}...")
        self.model_name = model_name
        print("This may take a moment on first run as it downloads...")
        
        # Check for GPU
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print("CUDA (GPU) available. Loading model on GPU.")
        else:
            print("CUDA not available. Loading model on CPU.")
            
        self.model = GLiNER.from_pretrained(model_name)
        print("✅ GLiNER model loaded successfully.")

        # Define non-skill words to filter out
        self.NON_SKILLS = {
            'experience', 'years', 'strong', 'skills', 'knowledge', 
            'proficient', 'expert', 'good', 'excellent', 'required',
            'must', 'have', 'should', 'nice', 'team', 'work',
            'ability', 'understanding', 'background', 'looking',
            'developer', 'engineer', 'and', 'with', 'using',
            'pursue', 'excellence', 'creation', 'products'
        }
        
        # Define labels for extraction
        self.labels = [
            "skill", "technology", "tool", "software",
            "programming language", "framework", "platform",
            "certification", "competency", "methodology"
        ]

    def __call__(self, text: str) -> List[Dict[str, any]]:
        """Extracts skills using GLiNER with filtering."""
        
        # Note: GLiNER's predict_entities doesn't use the 'device' param
        # It auto-detects if the model was loaded onto a device.
        entities = self.model.predict_entities(
            text, 
            self.labels,
            threshold=0.3, # Use a slightly higher threshold for better precision
        )
        
        results = []
        seen = set()
        
        for entity in entities:
            skill_text = entity['text'].strip()
            skill_lower = skill_text.lower()
            
            # Filter aggressively
            if (skill_lower not in self.NON_SKILLS and 
                len(skill_text) > 1 and # Filter single letters
                skill_lower not in seen):
                
                seen.add(skill_lower)
                results.append({
                    'word': skill_text,
                    'entity_group': entity['label'],
                    'score': entity['score'],
                    # Also return the context for Stage 3
                    'evidence': text[max(0, entity['start']-30) : min(len(text), entity['end']+30)]
                })
        
        return results

# Create a single, shared instance of the extractor
# This runs ONE time when the server starts.
gliner_extractor = GLiNERSkillExtractor()


# --- 2. HYBRID EXTRACTION & NORMALIZATION LOGIC ---

# Common skills for regex backup (from your Cell 7)
COMMON_SKILLS_PATTERN = re.compile(r'\b(' + r'|'.join([
    # Tech
    'python', 'sql', 'java', 'javascript', 'r', 'vba',
    'excel', 'powerpoint', 'word', 'tableau', 'power bi',
    'sap', 'oracle', 'quickbooks', 'netsuite', 'salesforce',
    # Finance
    'gaap', 'ifrs', 'sox', 'financial modeling', 'forecasting',
    'budgeting', 'fp&a', 'fpa', 'variance analysis',
    'financial statements', 'general ledger', 'accounts payable',
    'accounts receivable', 'month-end close', 'year-end close',
    'audit', 'tax', 'payroll', 'financial analysis',
    # Certs
    'cpa', 'cfa', 'cma', 'cia', 'mba',
    # Soft skills
    'leadership', 'communication', 'teamwork', 'problem-solving',
    'analytical', 'strategic thinking'
]) + r')\b', re.IGNORECASE)

# Bad ESCO terms to filter out (from your Cell 7)
BAD_ESCO_TERMS = [
    'pursue excellence', 'food products', 'creation',
    'manage', 'perform', 'ensure', 'coordinate'
]

def extract_skills_from_text(text: str) -> List[Dict[str, any]]:
    """
    Extracts skills using GLiNER + common patterns (Hybrid Approach).
    Returns a list of skill dictionaries, each including 'word' and 'evidence'.
    """
    found_skills_map = {} # Use a map to avoid duplicates, key = skill_lower

    # Method 1: GLiNER (Primary)
    try:
        gliner_skills = gliner_extractor(text)
        for skill_info in gliner_skills:
            if skill_info['score'] > 0.3:
                found_skills_map[skill_info['word'].lower()] = {
                    "skill": skill_info['word'],
                    "source": "gliner",
                    "evidence": skill_info['evidence']
                }
    except Exception as e:
        print(f"Error during GLiNER extraction: {e}")
        pass
    
    # Method 2: Common skill patterns (Backup)
    text_lower = text.lower()
    for match in COMMON_SKILLS_PATTERN.finditer(text):
        skill = match.group(0)
        skill_lower = skill.lower()
        if skill_lower not in found_skills_map:
            found_skills_map[skill_lower] = {
                "skill": skill.title(), # Capitalize for consistency
                "source": "regex",
                "evidence": text[max(0, match.start()-30) : min(len(text), match.end()+30)]
            }
    
    return list(found_skills_map.values())


def normalize_skills(skills: List[Dict[str, any]], data_loader: DataLoader) -> List[Dict[str, any]]:
    """
    Normalizes a list of skill dictionaries using the ESCO database.
    We pass in the data_loader to access its pre-loaded data.
    """
    normalized_list = []
    
    for skill_info in skills:
        skill_name = skill_info['skill']
        esco_match = data_loader.get_esco_match(skill_name) # Use fast lookup
        
        if esco_match:
            match_label = esco_match['normalized']
            
            # Filter bad ESCO matches
            if not any(bad in match_label.lower() for bad in BAD_ESCO_TERMS):
                normalized_list.append({
                    'original': skill_name,
                    'normalized': match_label,
                    'uri': esco_match['uri'],
                    'skill_type': esco_match['skill_type'],
                    'evidence': skill_info['evidence'],
                    'source': skill_info['source'],
                    'match_type': 'exact'
                })
                continue
        
        # If no good ESCO match, keep original
        normalized_list.append({
            'original': skill_name,
            'normalized': skill_name, # Keep as-is
            'uri': None,
            'skill_type': 'unknown',
            'evidence': skill_info['evidence'],
            'source': skill_info['source'],
            'match_type': 'none'
        })
    
    return normalized_list
