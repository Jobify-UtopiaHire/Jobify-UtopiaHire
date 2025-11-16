import React, { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FaDownload, FaFilePdf } from 'react-icons/fa';

interface CVPreviewProps {
  cvData: any;
  markdown: string;
}

const CVPreview: React.FC<CVPreviewProps> = ({ cvData, markdown }) => {
  const previewRef = useRef<HTMLDivElement>(null);

  const downloadPDF = async () => {
    try {
      // Use html2pdf or jsPDF library to convert the preview to PDF
      // For now, we'll create a simple implementation
      const element = previewRef.current;
      if (!element) return;

      // Dynamic import of html2pdf
      const html2pdf = (await import('html2pdf.js')).default;
      
      const options = {
        margin: 10,
        filename: 'cv.pdf',
        image: { type: 'jpeg' as const, quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const }
      };

      html2pdf().set(options).from(element).save();
    } catch (error) {
      console.error('Failed to generate PDF:', error);
      alert('Failed to generate PDF. Please try again.');
    }
  };

  const downloadMarkdown = () => {
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cv.md';
    a.click();
    URL.revokeObjectURL(url);
  };

  const name = cvData?.name || cvData?.full_name || cvData?.personal_info?.name || 'Your Name';
  const title = cvData?.title || cvData?.headline || '';
  const summary = cvData?.summary || cvData?.objective || cvData?.profile || '';
  const skills = cvData?.skills || cvData?.technical_skills || [];
  const experience = cvData?.experience || cvData?.work_experience || [];
  const education = cvData?.education || [];
  const projects = cvData?.projects || [];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <Card className="bg-slate-900/50 backdrop-blur-md border-slate-700 shadow-lg">
        <CardHeader className="border-b border-slate-800">
          <div className="flex items-center justify-between">
            <CardTitle className="text-slate-100">CV Preview</CardTitle>
            <div className="flex gap-3">
              <Button
                onClick={downloadMarkdown}
                variant="outline"
                className="border-slate-600 text-slate-200 hover:border-indigo-500"
              >
                <FaDownload className="mr-2" /> Download Markdown
              </Button>
              <Button
                onClick={downloadPDF}
                className="bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-500 hover:to-pink-500"
              >
                <FaFilePdf className="mr-2" /> Download PDF
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div
            ref={previewRef}
            className="bg-white p-12 min-h-[842px]"
            style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}
          >
            {/* Header */}
            <header className="text-center border-b-2 border-gray-300 pb-6 mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-2">{name}</h1>
              {title && <p className="text-xl text-indigo-600 font-medium">{title}</p>}
            </header>

            {/* Summary */}
            {summary && (
              <section className="mb-8">
                <h2 className="text-2xl font-bold text-gray-900 border-b-2 border-indigo-600 pb-2 mb-4">
                  Summary
                </h2>
                <p className="text-gray-700 leading-relaxed">{summary}</p>
              </section>
            )}

            {/* Skills */}
            {skills.length > 0 && (
              <section className="mb-8">
                <h2 className="text-2xl font-bold text-gray-900 border-b-2 border-indigo-600 pb-2 mb-4">
                  Skills
                </h2>
                <div className="flex flex-wrap gap-2">
                  {skills.map((skill: any, idx: number) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium"
                    >
                      {typeof skill === 'string' ? skill : skill?.name}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {/* Experience */}
            {experience.length > 0 && (
              <section className="mb-8">
                <h2 className="text-2xl font-bold text-gray-900 border-b-2 border-indigo-600 pb-2 mb-4">
                  Experience
                </h2>
                <div className="space-y-6">
                  {experience.map((exp: any, idx: number) => (
                    <div key={idx}>
                      <div className="flex justify-between items-baseline mb-2">
                        <h3 className="text-xl font-semibold text-gray-900">
                          {exp?.role || exp?.position || 'Position'}
                        </h3>
                        <span className="text-sm text-gray-600">
                          {exp?.start_date || exp?.start} - {exp?.end_date || exp?.end || 'Present'}
                        </span>
                      </div>
                      <p className="text-gray-700 font-medium mb-2">
                        {exp?.company || exp?.organization}
                      </p>
                      {Array.isArray(exp?.responsibilities) ? (
                        <ul className="list-disc ml-6 space-y-1">
                          {exp.responsibilities.map((resp: any, i: number) => (
                            <li key={i} className="text-gray-700">{resp}</li>
                          ))}
                        </ul>
                      ) : exp?.description ? (
                        <p className="text-gray-700">{exp.description}</p>
                      ) : null}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Education */}
            {education.length > 0 && (
              <section className="mb-8">
                <h2 className="text-2xl font-bold text-gray-900 border-b-2 border-indigo-600 pb-2 mb-4">
                  Education
                </h2>
                <div className="space-y-4">
                  {education.map((edu: any, idx: number) => (
                    <div key={idx}>
                      <div className="flex justify-between items-baseline mb-1">
                        <h3 className="text-xl font-semibold text-gray-900">
                          {edu?.degree || edu?.qualification}
                        </h3>
                        <span className="text-sm text-gray-600">
                          {edu?.year || edu?.start} {edu?.end ? `- ${edu.end}` : ''}
                        </span>
                      </div>
                      <p className="text-gray-700 font-medium">{edu?.institution || edu?.school}</p>
                      {edu?.details && <p className="text-gray-700 mt-1">{edu.details}</p>}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Projects */}
            {projects.length > 0 && (
              <section className="mb-8">
                <h2 className="text-2xl font-bold text-gray-900 border-b-2 border-indigo-600 pb-2 mb-4">
                  Projects
                </h2>
                <div className="space-y-4">
                  {projects.map((proj: any, idx: number) => (
                    <div key={idx}>
                      <h3 className="text-xl font-semibold text-gray-900 mb-1">
                        {proj?.name || proj?.title}
                      </h3>
                      {proj?.description && (
                        <p className="text-gray-700 mb-2">{proj.description}</p>
                      )}
                      {Array.isArray(proj?.technologies) && proj.technologies.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                          <span className="text-sm font-medium text-gray-600">Technologies:</span>
                          {proj.technologies.map((tech: any, i: number) => (
                            <span
                              key={i}
                              className="px-2 py-1 bg-gray-200 text-gray-700 rounded text-sm"
                            >
                              {tech}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CVPreview;
