"""
Job Search and Scraping Routes
Integrates LinkedIn job scraping functionality with the main FastAPI app
Restricted to Region 8 countries (Europe, Middle East, and Africa)
"""
from dataclasses import dataclass
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import time
import random
import json
import os
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from datetime import datetime, timedelta


# Region 8 countries (Europe, Middle East, and Africa)
REGION_8_COUNTRIES = {
    # Europe
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland",
    "Italy", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
    "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway",
    "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine",
    "United Kingdom", "Vatican City",
    
    # Middle East
    "Bahrain", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait", "Lebanon",
    "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "United Arab Emirates",
    "Yemen",
    
    # Africa
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
    "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo (Democratic Republic)", "Congo (Republic)", "Djibouti", "Equatorial Guinea",
    "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
    "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
}

# Allowed max_jobs values
ALLOWED_MAX_JOBS = {10, 20, 30}

# Experience levels
EXPERIENCE_LEVELS = {"internship", "entry_level", "associate", "mid_senior", "director", "executive"}


# Pydantic models for request/response validation
class JobSearchRequest(BaseModel):
    keywords: str
    location: str
    max_jobs: int = 10
    timeRange: str = ""
    experience_levels: List[str] = []

    @field_validator('location')
    @classmethod
    def validate_location(cls, v: str) -> str:
        if v not in REGION_8_COUNTRIES:
            raise ValueError(
                f"Location must be a Region 8 country. Allowed countries: {', '.join(sorted(REGION_8_COUNTRIES))}"
            )
        return v

    @field_validator('max_jobs')
    @classmethod
    def validate_max_jobs(cls, v: int) -> int:
        if v not in ALLOWED_MAX_JOBS:
            raise ValueError(f"max_jobs must be one of: {', '.join(map(str, sorted(ALLOWED_MAX_JOBS)))}")
        return v

    @field_validator('experience_levels')
    @classmethod
    def validate_experience_levels(cls, v: List[str]) -> List[str]:
        if v and not all(level in EXPERIENCE_LEVELS for level in v):
            raise ValueError(f"experience_levels must contain only: {', '.join(sorted(EXPERIENCE_LEVELS))}")
        return v


class JobSearchResponse(BaseModel):
    message: str
    jobs: List[dict]


@dataclass
class JobData:
    title: str
    company: str
    location: str
    job_link: str
    posted_date: str
    logo: str
    description: str
    logo_tag: str = ""
    in_time_range: bool = True
    time_note: str = ""
    experience_level: str = "Not specified"


class ScraperConfig:
    BASE_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    JOBS_PER_PAGE = 25
    MIN_DELAY = 2
    MAX_DELAY = 5
    RATE_LIMIT_DELAY = 30
    RATE_LIMIT_THRESHOLD = 10

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
        "Cache-Control": "no-cache",
    }


class LinkedInJobsScraper:
    def __init__(self):
        self.session = self._setup_session()

    def _setup_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _build_search_url(self, keywords: str, location: str, time_range: str = "", start: int = 0) -> str:
        # Base parameters
        params = {
            "keywords": keywords,
            "location": location,
            "start": start,
        }
        
        # Only add time range parameter if it's provided
        if time_range!=-1:
            params["f_TPR"] = f"r{time_range}"
        return f"{ScraperConfig.BASE_URL}?{'&'.join(f'{k}={quote(str(v))}' for k, v in params.items())}"

    def _clean_job_url(self, url: str) -> str:
        return url.split("?")[0] if "?" in url else url
    
    def get_job_description(self, job_link: str) -> str:
        """Fetch full job description from the job page."""
        try:
            response = requests.get(job_link, headers=ScraperConfig.HEADERS, timeout=10)
            if response.status_code != 200:
                return "N/A"
            soup = BeautifulSoup(response.text, "html.parser")
            desc_container = soup.find("div", class_="show-more-less-html__markup")
            return desc_container.get_text(strip=True) if desc_container else "N/A"
        except Exception as e:
            print(f"Error fetching job description: {str(e)}")
            return "N/A"

    def _check_time_range(self, posted_date_str: str, time_range: str) -> tuple[bool, str]:
        if time_range=="-1" or posted_date_str == "N/A":
            return True, ""
        
        try:
            # Parse LinkedIn's relative date format
            if "hour" in posted_date_str:
                hours = int(posted_date_str.split()[0])
                post_time = datetime.now() - timedelta(hours=hours)
            elif "day" in posted_date_str:
                days = int(posted_date_str.split()[0])
                post_time = datetime.now() - timedelta(days=days)
            elif "week" in posted_date_str:
                weeks = int(posted_date_str.split()[0])
                post_time = datetime.now() - timedelta(weeks=weeks)
            elif "month" in posted_date_str:
                months = int(posted_date_str.split()[0])
                post_time = datetime.now() - timedelta(days=months*30)
            else:
                return True, ""

            # Convert time_range from seconds to timedelta
            range_seconds = int(time_range)
            time_limit = datetime.now() - timedelta(seconds=range_seconds)

            if post_time > time_limit:
                return True, ""
            else:
                return False, "This job is not in your time range"
        except (ValueError, IndexError):
            return True, ""

    def _extract_experience_level(self, job_card: BeautifulSoup, description: str = "") -> str:
        """
        Extract experience level from job card or description.
        LinkedIn doesn't always show experience level directly, so we try to infer it from text.
        """
        try:
            # First, try to find explicit experience level metadata in the card
            # Check for span or div with experience level information
            level_indicators = {
                "internship": ["intern", "internship"],
                "entry_level": ["entry level", "entry-level", "graduate", "fresh", "junior", "junior level"],
                "associate": ["associate", "level 2"],
                "mid_senior": ["mid-level", "mid level", "senior", "lead", "5+ years", "6+ years", "7+ years", "8+ years"],
                "director": ["director", "vp", "vice president", "head of"],
                "executive": ["ceo", "cto", "cfo", "executive", "president", "chief"]
            }
            
            # Combine card text and description for analysis
            text_to_analyze = (job_card.get_text().lower() + " " + description.lower()).strip()
            
            # Check for each level
            for level, keywords in level_indicators.items():
                for keyword in keywords:
                    if keyword in text_to_analyze:
                        return level
            
            # Default to not specified if no match
            return "Not specified"
        except Exception as e:
            print(f"Error extracting experience level: {str(e)}")
            return "Not specified"

    def _extract_job_data(self, job_card: BeautifulSoup, time_range: str = "") -> Optional[JobData]:
        try:
            # title
            title = job_card.find("h3", class_="base-search-card__title").text.strip()
            company = job_card.find(
                "h4", class_="base-search-card__subtitle"
            ).text.strip()
            
            # logo (small card logo) and logo_tag (fallback/company image)
            def _find_logo_url(card: BeautifulSoup) -> str:
                attrs = ["src", "data-delayed-url", "data-src", "data-original", "data-img-src"]
                selectors = ["img.base-search-card__logo", "img.ivm-image-view_model", "img"]
                for sel in selectors:
                    el = card.select_one(sel)
                    if el:
                        for a in attrs:
                            v = el.get(a)
                            if v and v.strip():
                                return v
                # fallback: check any img inside the card and return first absolute URL found
                for img in card.find_all("img"):
                    for a in attrs:
                        v = img.get(a)
                        if v and v.strip() and (v.startswith("http") or v.startswith("//")):
                            # normalize protocol-relative URLs
                            return v if v.startswith("http") else f"https:{v}"
                return ""

            logo = _find_logo_url(job_card)
            
            # location
            location = job_card.find(
                "span", class_="job-search-card__location"
            ).text.strip()
            job_link = self._clean_job_url(
                job_card.find("a", class_="base-card__full-link")["href"]
            )
            
            # date - LinkedIn uses different class names for date
            posted_date_elem = (
                job_card.find("time", class_="job-search-card__listdate") or 
                job_card.find("time", class_="job-search-card__listdate--new") or
                job_card.find("time")  # Fallback to any time element
            )
            posted_date = posted_date_elem.text.strip() if posted_date_elem else "N/A"
            
            # Check if the job is within the requested time range
            in_range, time_note = self._check_time_range(posted_date, time_range)
            
            # Description
            description = self.get_job_description(job_link) if job_link != "N/A" else "N/A"

            # prefer logo (already found) as logo_tag
            logo_tag = logo or ""

            # If we still don't have a logo URL, try fetching the job page and look for og:image/twitter:image
            if not logo_tag and job_link and job_link != "N/A":
                try:
                    resp = self.session.get(job_link, headers=ScraperConfig.HEADERS, timeout=8)
                    if resp.status_code == 200:
                        soup_job = BeautifulSoup(resp.text, "html.parser")
                        meta = soup_job.find("meta", property="og:image") or soup_job.find("meta", attrs={"name": "twitter:image"})
                        if meta and meta.get("content"):
                            candidate = meta.get("content").strip()
                            if candidate.startswith("//"):
                                candidate = f"https:{candidate}"
                            logo_tag = candidate
                except Exception:
                    pass

            # Extract experience level
            experience_level = self._extract_experience_level(job_card, description)

            return JobData(
                title=title,
                company=company,
                location=location,
                job_link=job_link,
                posted_date=posted_date,
                description=description,
                logo=logo,
                logo_tag=logo_tag,
                in_time_range=in_range,
                time_note=time_note,
                experience_level=experience_level
            )
        except Exception as e:
            print(f"Failed to extract job data: {str(e)}")
            return None

    def _fetch_job_page(self, url: str) -> BeautifulSoup:
        try:
            response = self.session.get(url, headers=ScraperConfig.HEADERS, timeout=15)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch data: Status code {response.status_code}"
                )
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {str(e)}")

    def scrape_jobs(
        self, keywords: str, location: str, max_jobs: int = 50, time_range: str = "", experience_levels: List[str] = None
    ) -> List[JobData]:
        if experience_levels is None:
            experience_levels = []
        max_jobs = int(max_jobs)
        all_jobs = []
        start = 0

        while len(all_jobs) < max_jobs:
            try:
                url = self._build_search_url(keywords, location, time_range, start)
                soup = self._fetch_job_page(url)
                job_cards = soup.find_all("div", class_="base-card")

                if not job_cards:
                    break
                    
                for card in job_cards:
                    job_data = self._extract_job_data(card, time_range)
                    if job_data:
                        # Filter by experience level if specified
                        if experience_levels and job_data.experience_level not in experience_levels:
                            continue
                        all_jobs.append(job_data)
                        if len(all_jobs) >= max_jobs:
                            break
                            
                print(f"📊 Scraped {len(all_jobs)} jobs...")
                start += ScraperConfig.JOBS_PER_PAGE
                time.sleep(
                    random.uniform(ScraperConfig.MIN_DELAY, ScraperConfig.MAX_DELAY)
                )
            except Exception as e:
                print(f"❌ Scraping error: {str(e)}")
                break
                
        return all_jobs[:max_jobs]

    def save_results(
        self, jobs: List[JobData], filename: str = "linkedin_jobs.json"
    ) -> None:
        if not jobs:
            return
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([vars(job) for job in jobs], f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(jobs)} jobs to {filename}")


# Initialize router
router = APIRouter(tags=["jobs"])

# File path for storing scraped jobs
JOBS_FILE = "linkedin_jobs.json"


@router.post("/api/jobs/search", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest):
    """
    Search for jobs on LinkedIn and save results.
    Restricted to Region 8 countries (Europe, Middle East, and Africa).
    
    - **keywords**: Job search keywords
    - **location**: Country (must be from Region 8: Europe, Middle East, or Africa)
    - **max_jobs**: Maximum number of jobs to scrape (10, 20, or 30)
    - **timeRange**: Time range filter in seconds (optional)
    - **experience_levels**: Filter by experience levels (optional)
    """
    try:
        print(f"🔍 Starting job search: {request.keywords} in {request.location}")
        print(f"📊 Max jobs: {request.max_jobs}, Time range: {request.timeRange or 'Any'}")
        print(f"🌍 Region 8 country validated: {request.location}")
        if request.experience_levels:
            print(f"🎓 Experience levels filter: {', '.join(request.experience_levels)}")
        
        scraper = LinkedInJobsScraper()
        jobs = scraper.scrape_jobs(
            keywords=request.keywords,
            location=request.location,
            max_jobs=request.max_jobs,
            time_range=request.timeRange,
            experience_levels=request.experience_levels
        )
        scraper.save_results(jobs, JOBS_FILE)

        print(f"✅ Job search completed: {len(jobs)} jobs found")
        return JobSearchResponse(
            message=f"{len(jobs)} jobs found and saved.",
            jobs=[vars(job) for job in jobs]
        )
    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error scraping jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scraping jobs: {str(e)}")


@router.get("/api/jobs")
async def get_jobs():
    """
    Retrieve saved jobs from the JSON file.
    """
    try:
        if not os.path.exists(JOBS_FILE):
            return {"jobs": []}
            
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            jobs = json.load(f)
            
        print(f"📋 Retrieved {len(jobs)} jobs from storage")
        return {"jobs": jobs}
    except FileNotFoundError:
        return {"jobs": []}
    except Exception as e:
        print(f"❌ Error retrieving jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
