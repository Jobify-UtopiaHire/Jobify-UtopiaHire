import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

const NoJobsFound = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center py-[60px] px-5 font-['Inter',sans-serif]">
      <div className="flex flex-col items-center justify-center max-w-2xl mx-auto text-center">
        <div className="backdrop-blur-xl bg-black/40 rounded-2xl p-12 border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.3)]">
          {/* Sad Face Icon */}
          <div className="mb-8">
            <svg
              className="mx-auto h-32 w-32 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>

          {/* Main Message */}
          <h1 className="text-4xl font-bold text-white mb-4">We're Sorry!</h1>
          <h2 className="text-2xl font-semibold text-gray-300 mb-6">
            No Jobs Found
          </h2>

          {/* Description */}
          <p className="text-gray-400 text-lg mb-8 leading-relaxed">
            Unfortunately, we couldn't find any jobs matching your search
            criteria.
            <br />
            Try adjusting your search parameters or check back later for new
            opportunities.
          </p>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              variant="outline"
              onClick={() => navigate("/job-matcher")}
              className="hover:bg-white hover:text-gray-900 text-lg px-8 py-6"
            >
              Try Another Search
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/dashboard")}
              className="hover:bg-blue-600 hover:text-white text-lg px-8 py-6 border-blue-500"
            >
              Go to Dashboard
            </Button>
          </div>

          {/* Tips Section */}
          <div className="mt-10 pt-8 border-t border-gray-700">
            <h3 className="text-white font-semibold mb-4 text-left">
              💡 Search Tips:
            </h3>
            <ul className="text-gray-400 text-left space-y-2">
              <li>• Try different keywords or job titles</li>
              <li>• Expand your location search to nearby countries</li>
              <li>• Increase the time range to find more results</li>
              <li>• Check your spelling and try broader terms</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NoJobsFound;
