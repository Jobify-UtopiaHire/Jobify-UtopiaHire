import pandas as pd
import os
import time
from typing import Dict, Any, List, Optional

class DataLoader:
    def __init__(self, usa_jobs_path: str, esco_skills_path: str):
        print("Data loader initializing...")
        
        # Load ESCO skills
        print(f"Loading ESCO skills from {esco_skills_path}...")
        self.esco_df = pd.read_csv(esco_skills_path)
        # Pre-process for fast lookups
        self.esco_df['preferredLabel_lower'] = self.esco_df['preferredLabel'].str.lower()
        print(f"✅ Loaded {len(self.esco_df):,} ESCO skills.")

        # Load USA jobs data
        print(f"Loading USA jobs data from {usa_jobs_path}...")
        usa_jobs_df = pd.read_csv(usa_jobs_path)
        print(f"✅ Loaded {len(usa_jobs_df):,} job records.")

        # --- OPTIMIZATION ---
        # Pre-compute market demand and top roles
        # This moves the 3-second 'groupby' from runtime to startup
        print("Pre-computing market demand statistics...")
        start_time = time.time()
        
        # 1. Calculate total demand per skill
        usa_jobs_df['Skill_Keyword_lower'] = usa_jobs_df['Skill Keyword'].str.lower()
        self.skill_demand = usa_jobs_df.groupby('Skill_Keyword_lower')['Count'].sum().to_dict()

        # 2. Calculate top roles per skill (fixed to avoid FutureWarning)
        self.top_roles_for_skill = usa_jobs_df.groupby('Skill_Keyword_lower', group_keys=False).apply(
            lambda g: g.nlargest(5, 'Count')[['Job Posting Title', 'Count']].to_dict('records'),
            include_groups=False
        ).to_dict()
        
        end_time = time.time()
        print(f"✅ Pre-computation finished in {end_time - start_time:.2f} seconds.")
        # We no longer need the full 300k+ row dataframe
        del usa_jobs_df

    def get_market_demand(self, skill: str) -> Dict[str, Any]:
        """
        Gets pre-computed market demand for a skill (case-insensitive).
        This is now a lightning-fast dictionary lookup.
        """
        skill_lower = skill.lower()
        total_demand = self.skill_demand.get(skill_lower, 0)
        
        if total_demand > 5000:
            priority = 'Critical'
        elif total_demand > 2000:
            priority = 'High'
        elif total_demand > 500:
            priority = 'Medium'
        else:
            priority = 'Low'

        return {
            'skill': skill,
            'total_demand': int(total_demand),
            'top_roles': self.top_roles_for_skill.get(skill_lower, []),
            'priority': priority
        }

    def get_esco_match(self, skill: str) -> Optional[Dict[str, Any]]:
        """
        Gets the standardized ESCO skill name (case-insensitive).
        """
        matches = self.esco_df[self.esco_df['preferredLabel_lower'] == skill.lower()]
        if not matches.empty:
            match_row = matches.iloc[0]
            return {
                'normalized': match_row['preferredLabel'],
                'uri': match_row['conceptUri'],
                'skill_type': match_row.get('skillType', 'unknown')
            }
        return None

# --- SINGLETON PATTERN ---
# Define the paths
USA_JOBS_PATH = os.path.join("data", "usa_job_posting_dataset.csv")
ESCO_SKILLS_PATH = os.path.join("data", "skills_en.csv")

# Create a single, shared instance of the DataLoader
# This code runs ONE time when the server starts.
print("Creating shared data_loader instance...")
data_loader = DataLoader(usa_jobs_path=USA_JOBS_PATH, esco_skills_path=ESCO_SKILLS_PATH)
print("✅ Shared data_loader instance is ready.")
