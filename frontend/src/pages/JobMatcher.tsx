import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Experience levels like LinkedIn
const EXPERIENCE_LEVELS = [
  { value: "internship", label: "Internship" },
  { value: "entry_level", label: "Entry level" },
  { value: "associate", label: "Associate" },
  { value: "mid_senior", label: "Mid-Senior level" },
  { value: "director", label: "Director" },
  { value: "executive", label: "Executive" },
];

// Region 8 countries (Europe, Middle East, and Africa)
const REGION_8_COUNTRIES = [
  // Europe
  { value: "Albania", label: "Albania", region: "Europe" },
  { value: "Andorra", label: "Andorra", region: "Europe" },
  { value: "Austria", label: "Austria", region: "Europe" },
  { value: "Belarus", label: "Belarus", region: "Europe" },
  { value: "Belgium", label: "Belgium", region: "Europe" },
  {
    value: "Bosnia and Herzegovina",
    label: "Bosnia and Herzegovina",
    region: "Europe",
  },
  { value: "Bulgaria", label: "Bulgaria", region: "Europe" },
  { value: "Croatia", label: "Croatia", region: "Europe" },
  { value: "Cyprus", label: "Cyprus", region: "Europe" },
  { value: "Czech Republic", label: "Czech Republic", region: "Europe" },
  { value: "Denmark", label: "Denmark", region: "Europe" },
  { value: "Estonia", label: "Estonia", region: "Europe" },
  { value: "Finland", label: "Finland", region: "Europe" },
  { value: "France", label: "France", region: "Europe" },
  { value: "Germany", label: "Germany", region: "Europe" },
  { value: "Greece", label: "Greece", region: "Europe" },
  { value: "Hungary", label: "Hungary", region: "Europe" },
  { value: "Iceland", label: "Iceland", region: "Europe" },
  { value: "Ireland", label: "Ireland", region: "Europe" },
  { value: "Italy", label: "Italy", region: "Europe" },
  { value: "Latvia", label: "Latvia", region: "Europe" },
  { value: "Liechtenstein", label: "Liechtenstein", region: "Europe" },
  { value: "Lithuania", label: "Lithuania", region: "Europe" },
  { value: "Luxembourg", label: "Luxembourg", region: "Europe" },
  { value: "Malta", label: "Malta", region: "Europe" },
  { value: "Moldova", label: "Moldova", region: "Europe" },
  { value: "Monaco", label: "Monaco", region: "Europe" },
  { value: "Montenegro", label: "Montenegro", region: "Europe" },
  { value: "Netherlands", label: "Netherlands", region: "Europe" },
  { value: "North Macedonia", label: "North Macedonia", region: "Europe" },
  { value: "Norway", label: "Norway", region: "Europe" },
  { value: "Poland", label: "Poland", region: "Europe" },
  { value: "Portugal", label: "Portugal", region: "Europe" },
  { value: "Romania", label: "Romania", region: "Europe" },
  { value: "Russia", label: "Russia", region: "Europe" },
  { value: "San Marino", label: "San Marino", region: "Europe" },
  { value: "Serbia", label: "Serbia", region: "Europe" },
  { value: "Slovakia", label: "Slovakia", region: "Europe" },
  { value: "Slovenia", label: "Slovenia", region: "Europe" },
  { value: "Spain", label: "Spain", region: "Europe" },
  { value: "Sweden", label: "Sweden", region: "Europe" },
  { value: "Switzerland", label: "Switzerland", region: "Europe" },
  { value: "Turkey", label: "Turkey", region: "Europe" },
  { value: "Ukraine", label: "Ukraine", region: "Europe" },
  { value: "United Kingdom", label: "United Kingdom", region: "Europe" },
  { value: "Vatican City", label: "Vatican City", region: "Europe" },

  // Middle East
  { value: "Bahrain", label: "Bahrain", region: "Middle East" },
  { value: "Egypt", label: "Egypt", region: "Middle East" },
  { value: "Iran", label: "Iran", region: "Middle East" },
  { value: "Iraq", label: "Iraq", region: "Middle East" },
  { value: "Jordan", label: "Jordan", region: "Middle East" },
  { value: "Kuwait", label: "Kuwait", region: "Middle East" },
  { value: "Lebanon", label: "Lebanon", region: "Middle East" },
  { value: "Oman", label: "Oman", region: "Middle East" },
  { value: "Palestine", label: "Palestine", region: "Middle East" },
  { value: "Qatar", label: "Qatar", region: "Middle East" },
  { value: "Saudi Arabia", label: "Saudi Arabia", region: "Middle East" },
  { value: "Syria", label: "Syria", region: "Middle East" },
  {
    value: "United Arab Emirates",
    label: "United Arab Emirates",
    region: "Middle East",
  },
  { value: "Yemen", label: "Yemen", region: "Middle East" },

  // Africa
  { value: "Algeria", label: "Algeria", region: "Africa" },
  { value: "Angola", label: "Angola", region: "Africa" },
  { value: "Benin", label: "Benin", region: "Africa" },
  { value: "Botswana", label: "Botswana", region: "Africa" },
  { value: "Burkina Faso", label: "Burkina Faso", region: "Africa" },
  { value: "Burundi", label: "Burundi", region: "Africa" },
  { value: "Cameroon", label: "Cameroon", region: "Africa" },
  { value: "Cape Verde", label: "Cape Verde", region: "Africa" },
  {
    value: "Central African Republic",
    label: "Central African Republic",
    region: "Africa",
  },
  { value: "Chad", label: "Chad", region: "Africa" },
  { value: "Comoros", label: "Comoros", region: "Africa" },
  {
    value: "Congo (Democratic Republic)",
    label: "Congo (Democratic Republic)",
    region: "Africa",
  },
  { value: "Congo (Republic)", label: "Congo (Republic)", region: "Africa" },
  { value: "Djibouti", label: "Djibouti", region: "Africa" },
  { value: "Equatorial Guinea", label: "Equatorial Guinea", region: "Africa" },
  { value: "Eritrea", label: "Eritrea", region: "Africa" },
  { value: "Eswatini", label: "Eswatini", region: "Africa" },
  { value: "Ethiopia", label: "Ethiopia", region: "Africa" },
  { value: "Gabon", label: "Gabon", region: "Africa" },
  { value: "Gambia", label: "Gambia", region: "Africa" },
  { value: "Ghana", label: "Ghana", region: "Africa" },
  { value: "Guinea", label: "Guinea", region: "Africa" },
  { value: "Guinea-Bissau", label: "Guinea-Bissau", region: "Africa" },
  { value: "Ivory Coast", label: "Ivory Coast", region: "Africa" },
  { value: "Kenya", label: "Kenya", region: "Africa" },
  { value: "Lesotho", label: "Lesotho", region: "Africa" },
  { value: "Liberia", label: "Liberia", region: "Africa" },
  { value: "Libya", label: "Libya", region: "Africa" },
  { value: "Madagascar", label: "Madagascar", region: "Africa" },
  { value: "Malawi", label: "Malawi", region: "Africa" },
  { value: "Mali", label: "Mali", region: "Africa" },
  { value: "Mauritania", label: "Mauritania", region: "Africa" },
  { value: "Mauritius", label: "Mauritius", region: "Africa" },
  { value: "Morocco", label: "Morocco", region: "Africa" },
  { value: "Mozambique", label: "Mozambique", region: "Africa" },
  { value: "Namibia", label: "Namibia", region: "Africa" },
  { value: "Niger", label: "Niger", region: "Africa" },
  { value: "Nigeria", label: "Nigeria", region: "Africa" },
  { value: "Rwanda", label: "Rwanda", region: "Africa" },
  {
    value: "São Tomé and Príncipe",
    label: "São Tomé and Príncipe",
    region: "Africa",
  },
  { value: "Senegal", label: "Senegal", region: "Africa" },
  { value: "Seychelles", label: "Seychelles", region: "Africa" },
  { value: "Sierra Leone", label: "Sierra Leone", region: "Africa" },
  { value: "Somalia", label: "Somalia", region: "Africa" },
  { value: "South Africa", label: "South Africa", region: "Africa" },
  { value: "South Sudan", label: "South Sudan", region: "Africa" },
  { value: "Sudan", label: "Sudan", region: "Africa" },
  { value: "Tanzania", label: "Tanzania", region: "Africa" },
  { value: "Togo", label: "Togo", region: "Africa" },
  { value: "Tunisia", label: "Tunisia", region: "Africa" },
  { value: "Uganda", label: "Uganda", region: "Africa" },
  { value: "Zambia", label: "Zambia", region: "Africa" },
  { value: "Zimbabwe", label: "Zimbabwe", region: "Africa" },
];

// Group countries by region
const groupedCountries = REGION_8_COUNTRIES.reduce((acc, country) => {
  if (!acc[country.region]) {
    acc[country.region] = [];
  }
  acc[country.region].push(country);
  return acc;
}, {} as Record<string, typeof REGION_8_COUNTRIES>);

const PostJob = () => {
  const [formData, setFormData] = useState({
    keywords: "",
    location: "",
    max_jobs: "",
    timeRange: "",
    experience_levels: [] as string[],
  });
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const navigate = useNavigate();

  // Restore last search parameters on component mount
  useEffect(() => {
    const lastSearch = localStorage.getItem("lastJobSearch");
    if (lastSearch) {
      try {
        const parsedSearch = JSON.parse(lastSearch);
        // Ensure experience_levels exists (for compatibility with old saved data)
        if (!parsedSearch.experience_levels) {
          parsedSearch.experience_levels = [];
        }
        setFormData(parsedSearch);
        setSearchQuery(parsedSearch.location || "");
      } catch (error) {
        console.error("Failed to parse last search:", error);
      }
    }
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  // Handler for location select
  const handleLocationChange = (value: string) => {
    setFormData({ ...formData, location: value });
    setSearchQuery(value); // Show selected country in input
    setIsDropdownOpen(false); // Close dropdown
  };

  // Handler for search input
  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchQuery(value);
    setIsDropdownOpen(true); // Open dropdown when typing

    // If user clears the input, clear the location
    if (!value) {
      setFormData({ ...formData, location: "" });
    }
  };

  // Filter countries based on search query
  const filteredCountries = searchQuery
    ? REGION_8_COUNTRIES.filter((country) =>
        country.label.toLowerCase().startsWith(searchQuery.toLowerCase())
      )
    : REGION_8_COUNTRIES;

  // Group filtered countries by region
  const filteredGroupedCountries = filteredCountries.reduce((acc, country) => {
    if (!acc[country.region]) {
      acc[country.region] = [];
    }
    acc[country.region].push(country);
    return acc;
  }, {} as Record<string, typeof REGION_8_COUNTRIES>);

  // Handler for max jobs select
  const handleMaxJobsChange = (value: string) => {
    setFormData({ ...formData, max_jobs: value });
  };

  // Handler for time range select
  const handleTimeRangeChange = (value: string) => {
    setFormData({ ...formData, timeRange: value });
  };

  // Handler for experience level checkboxes
  const handleExperienceLevelChange = (level: string) => {
    setFormData((prevData) => {
      const levels = prevData.experience_levels.includes(level)
        ? prevData.experience_levels.filter((l) => l !== level)
        : [...prevData.experience_levels, level];
      return { ...prevData, experience_levels: levels };
    });
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/jobs/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        // Save the last search parameters to localStorage
        localStorage.setItem("lastJobSearch", JSON.stringify(formData));

        // Check if any jobs were found
        if (data.jobs && data.jobs.length > 0) {
          toast.success(data.message || "Jobs found successfully!");
          navigate("/GetJobs");
        } else {
          // No jobs found - redirect to NoJobsFound page
          toast.info("No jobs found for your search");
          navigate("/no-jobs-found");
        }
      } else {
        const errorData = await response.json();
        toast.error(errorData.detail || "Failed to search jobs");
        console.error("Failed to submit form:", errorData);
        setIsLoading(false);
      }
    } catch (error) {
      console.error("Error:", error);
      toast.error("Network error. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="max-w-[500px] my-[50px] mx-auto p-[30px] rounded-[10px] backdrop-blur-xl bg-black/40 border border-white/10 shadow-[0_4px_8px_rgba(0,0,0,0.1)] font-['Arial',sans-serif] max-[600px]:p-5 font-extrabold"
    >
      <h2 className="text-center mb-5 text-white text-3xl">Search a Job</h2>
      <div>
        <label
          htmlFor="keywords"
          className="font-bold block mb-2 text-[#D1D5DB]"
        >
          Job Title:
        </label>
        <input
          type="text"
          id="keywords"
          name="keywords"
          value={formData.keywords}
          onChange={handleChange}
          required
          className="w-full p-3 border border-[#007bff] my-[15px] rounded-[6px] text-white bg-transparent transition-[border-color] duration-300 ease-[ease] focus:border-white focus:outline-none focus:shadow-[0_0_5px_rgba(0,123,255,0.5)] max-[600px]:text-sm"
        />
      </div>
      <div>
        <label
          htmlFor="location"
          className="font-bold block mb-2 text-[#D1D5DB]"
        >
          Country (Region 8):
        </label>
        <div className="relative" ref={dropdownRef}>
          <input
            type="text"
            placeholder="Type to search country..."
            value={searchQuery}
            onChange={handleSearchInputChange}
            onFocus={() => setIsDropdownOpen(true)}
            onClick={() => setIsDropdownOpen(true)}
            className="w-full p-3 border border-[#007bff] my-[15px] rounded-[6px] text-white bg-transparent transition-[border-color] duration-300 ease-[ease] focus:border-white focus:outline-none focus:shadow-[0_0_5px_rgba(0,123,255,0.5)] max-[600px]:text-sm"
          />
          {isDropdownOpen && (
            <div className="absolute z-50 w-full mt-1 max-h-[300px] overflow-y-auto bg-gray-900 border border-[#007bff] rounded-md shadow-lg">
              {Object.keys(filteredGroupedCountries).length > 0 ? (
                Object.entries(filteredGroupedCountries).map(
                  ([region, countries]) => (
                    <div key={region}>
                      <div className="px-3 py-2 text-xs font-semibold text-gray-400 bg-gray-800 sticky top-0">
                        {region}
                      </div>
                      {countries.map((country) => (
                        <button
                          key={country.value}
                          type="button"
                          onClick={() => handleLocationChange(country.value)}
                          className="w-full text-left px-4 py-2 text-white hover:bg-gray-800 transition-colors duration-150 border-b border-gray-800 last:border-b-0"
                        >
                          {country.label}
                        </button>
                      ))}
                    </div>
                  )
                )
              ) : (
                <div className="px-4 py-6 text-center text-sm text-gray-400">
                  No countries found
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <div>
        <label
          htmlFor="max_jobs"
          className="font-bold block mb-2 text-[#D1D5DB]"
        >
          Max Jobs:
        </label>
        <Select value={formData.max_jobs} onValueChange={handleMaxJobsChange}>
          <SelectTrigger className="w-full border border-[#007bff] my-[15px]">
            <SelectValue placeholder="Select max jobs" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="10">10 jobs</SelectItem>
            <SelectItem value="20">20 jobs</SelectItem>
            <SelectItem value="30">30 jobs</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div>
        <label
          htmlFor="timeRange"
          className="font-bold block mb-2 text-[#D1D5DB]"
        >
          Time Range:
        </label>
        <Select
          value={formData.timeRange}
          onValueChange={handleTimeRangeChange}
        >
          <SelectTrigger className="w-full border border-[#007bff] my-[15px]">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="-1">None</SelectItem>
            <SelectItem value="86400">24 hours</SelectItem>
            <SelectItem value="604800">7 days</SelectItem>
            <SelectItem value="2592000">30 days</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div>
        <label className="font-bold block mb-3 text-[#D1D5DB]">
          Experience Level:
        </label>
        <div className="space-y-2 bg-gray-900/50 p-4 rounded-[6px]">
          {EXPERIENCE_LEVELS.map((level) => (
            <div key={level.value} className="flex items-center">
              <input
                type="checkbox"
                id={level.value}
                checked={(formData.experience_levels || []).includes(
                  level.value
                )}
                onChange={() => handleExperienceLevelChange(level.value)}
                className="w-4 h-4 rounded border-gray-300 text-[#007bff] focus:ring-[#007bff] cursor-pointer"
              />
              <label
                htmlFor={level.value}
                className="ml-3 text-[#D1D5DB] cursor-pointer select-none"
              >
                {level.label}
              </label>
            </div>
          ))}
        </div>
      </div>
      <div className="has-[button:disabled]:cursor-not-allowed mt-10">
        <Button
          type="submit"
          className="w-full"
          variant="outline"
          disabled={
            !formData.keywords ||
            !formData.location ||
            !formData.max_jobs ||
            !formData.timeRange ||
            isLoading
          }
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Searching...
            </>
          ) : (
            "Submit"
          )}
        </Button>
      </div>
    </form>
  );
};

export default PostJob;
