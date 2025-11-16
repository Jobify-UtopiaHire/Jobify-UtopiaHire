// LaTeX-based PDF export for structured CV
// Uses backend LaTeX compiler (xelatex) for professional PDF generation

import type { StructuredCV } from '../types';
import { exportLatexPdf } from '../api';

export async function exportStructuredCVToPdf(structuredCV: StructuredCV = {}): Promise<void> {
  try {
    const contact = structuredCV.contact || {};
    const name = contact.name || 'Candidate';
    
    console.log('📄 Exporting CV to LaTeX PDF...');
    console.log('📋 Structured CV keys:', Object.keys(structuredCV));
    console.log('📋 Contact info:', contact);
    console.log('📋 Full CV data:', structuredCV);
    
    // Call backend to generate and compile LaTeX
    const pdfBlob = await exportLatexPdf(structuredCV);
    
    // Create download link
    const url = window.URL.createObjectURL(pdfBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${name.replace(/\s+/g, '_')}_CV.pdf`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    
    // Cleanup
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    console.log('✅ PDF downloaded successfully');
  } catch (e) {
    console.error('Failed to export LaTeX PDF:', e);
    const errorMsg = e instanceof Error ? e.message : 'Unknown error';
    alert(`Failed to export PDF: ${errorMsg}\n\nPlease ensure:\n1. Backend is running\n2. xelatex is installed and in PATH\n3. LaTeX packages (moderncv) are installed`);
  }
}
