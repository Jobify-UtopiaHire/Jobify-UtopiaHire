import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

// Define the Job interface
interface Job {
  title: string;
  company: string;
  location: string;
  posted_date: string;
  description?: string;
  logo_tag?: string;
  time_note?: string;
  job_link: string;
}

const GetJob = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/jobs");
        if (!response.ok) {
          throw new Error("Failed to fetch jobs");
        }
        const data = await response.json();
        setJobs(data.jobs);
      } catch (err) {
        setError((err as Error).message);
      }
    };

    fetchJobs();
  }, []);

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p className="bg-[#fdecea] text-[#b71c1c] py-4 px-6 rounded-lg font-medium shadow-[0_2px_8px_rgba(183,28,28,0.2)]">
          Error: {error}
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-[60px] px-5 font-['Inter',sans-serif] text-[#333]">
      <h1 className="text-center text-[2.5rem] font-bold mb-10 text-white">
        Job Listings
      </h1>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-[25px] max-w-[1100px] mx-auto">
        {jobs.map((job, index) => (
          <JobCard key={index} job={job} />
        ))}
      </div>
    </div>
  );
};

// Small subcomponent to render a job card with expandable description
const JobCard = ({ job }: { job: Job }) => {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();

  const description = job.description || "";
  const PREVIEW_LEN = 120; // shorter preview
  const shortDesc =
    description.length > PREVIEW_LEN
      ? description.slice(0, PREVIEW_LEN) + "..."
      : description;

  const handleTakeInterview = () => {
    // Save job details to localStorage for the InterviewSetup page
    localStorage.setItem("interviewer_job_description", description);
    localStorage.setItem("interviewer_job_title", job.title);

    // Navigate to the interview setup page
    navigate("/interviewer/setup");
  };
  const handleMakePersonalizedCv = ()=> {
    localStorage.setItem("cvtool_job_description", description);
    navigate("/cv");
  }
const handleSkillGap = ()=> {
  localStorage.setItem("gap_job_description",description);
  localStorage.setItem("gap_job_title",job.title);
  navigate("/skills-gap");
}
  return (
    <div className="backdrop-blur-xl bg-black/40 rounded-2xl p-[25px] shadow-[0_2px_10px_rgba(0,0,0,0.08)] transition-[transform,box-shadow] duration-[0.25s] ease-[ease] hover:-translate-y-1.5 hover:shadow-[0_6px_20px_rgba(0,0,0,0.12)]">
      <div className="flex items-center gap-3 mb-2">
        {job.logo_tag && (
          <img
            src={job.logo_tag}
            alt={job.company + " logo"}
            className="block w-16 h-16 min-w-[64px] min-h-[64px] rounded-lg object-contain bg-[#f3f3f3] shadow-[0_1px_4px_rgba(0,0,0,0.07)]"
          />
        )}
        <h2 className="text-xl font-semibold text-white mb-2.5">{job.title}</h2>
      </div>
      <p className="text-[#D1D5DB]">
        <strong className="text-white">Company:</strong> {job.company}
      </p>
      <p className="text-[#D1D5DB]">
        <strong className="text-white">Location:</strong> {job.location}
      </p>
      <p className="text-[0.9rem] text-[#D1D5DB] my-2 mb-4">
        <strong className="text-white">Posted:</strong> {job.posted_date}
      </p>
      {job.time_note && (
        <p className="text-red-600 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950/30 dark:to-blue-900/30 py-1.5 px-2.5 rounded-md inline-block mb-3 font-semibold">
          {job.time_note}!
        </p>
      )}

      {description ? (
        <>
          <div className="font-semibold text-white mb-1.5">Description:</div>
          {expanded ? (
            <>
              <p className="text-[#D1D5DB] leading-[1.4] my-2 mb-3 text-sm font-[-apple-system,BlinkMacSystemFont,'Segoe_UI',Roboto,'Helvetica_Neue',Arial,sans-serif]">
                {description}
              </p>
              {description.length > PREVIEW_LEN && (
                <button
                  className="bg-transparent border-none text-[#D1D5DB] cursor-pointer font-semibold py-0.5 px-0 text-xs normal-case w-full"
                  onClick={() => setExpanded(false)}
                >
                  Show less
                </button>
              )}
            </>
          ) : (
            <p className="text-[#D1D5DB] leading-[1.4] my-2 mb-3 text-sm font-[-apple-system,BlinkMacSystemFont,'Segoe_UI',Roboto,'Helvetica_Neue',Arial,sans-serif]">
              {shortDesc.replace("...", "")}
              {description.length > PREVIEW_LEN && (
                <button
                  className="inline-block ml-1.5 bg-transparent border-none text-[#6b7280] cursor-pointer font-medium text-xs normal-case hover:text-[#374151] hover:no-underline hover:opacity-95"
                  onClick={() => setExpanded(true)}
                >
                  ...Show more
                </button>
              )}
            </p>
          )}
        </>
      ) : (
        <>
          <div className="font-semibold text-[#374151] mb-1.5">
            Description:
          </div>
          <p className="text-[#D1D5DB] leading-[1.4] my-2 mb-3 text-sm font-[-apple-system,BlinkMacSystemFont,'Segoe_UI',Roboto,'Helvetica_Neue',Arial,sans-serif]">
            No description available.
          </p>
        </>
      )}
      <div className="flex gap-3 mt-4">
        <Button variant="outline">
          <a
            href={job.job_link}
            target="_blank"
            rel="noopener noreferrer"
            className="text-white"
          >
            View Job
          </a>
        </Button>
        <Button variant="outline" onClick={handleTakeInterview}>
          <p className="text-white">Take an Interview</p>
        </Button>
      </div>
      <div className="flex gap-3 mt-4">
        <Button variant="outline" onClick={handleMakePersonalizedCv}>
          <p className="text-white">Correct Cv</p>
        </Button>
        <Button variant="outline" onClick={handleSkillGap}>
          <p className="text-white">Check Skill gap</p>
        </Button>

      </div>
    </div>
  );
};

export default GetJob;
