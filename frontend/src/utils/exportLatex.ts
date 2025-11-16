// LaTeX code export utility
// Handles getting and downloading LaTeX code

import type { StructuredCV } from '../types';
import { getLatexCode } from '../api';

export async function downloadLatexCode(structuredCV: StructuredCV = {}, cachedCode?: string): Promise<void> {
  try {
    const contact = structuredCV.contact || {};
    const name = contact.name || 'Candidate';
    
    console.log('📝 Downloading LaTeX code...');
    
    // Use cached code if provided, otherwise fetch from backend
    let latexCode = cachedCode;
    if (!latexCode) {
      console.log('📡 Fetching LaTeX code from backend...');
      latexCode = await getLatexCode(structuredCV);
    } else {
      console.log('♻️ Using cached LaTeX code');
    }
    
    // Create text file and download
    const blob = new Blob([latexCode], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${name.replace(/\s+/g, '_')}_CV.tex`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    
    // Cleanup
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    console.log('✅ LaTeX code downloaded successfully');
  } catch (e) {
    console.error('Failed to export LaTeX code:', e);
    const errorMsg = e instanceof Error ? e.message : 'Unknown error';
    alert(`Failed to download LaTeX code: ${errorMsg}\n\nPlease ensure backend is running.`);
    throw e;
  }
}

export async function copyLatexCode(structuredCV: StructuredCV = {}, cachedCode?: string): Promise<void> {
  try {
    console.log('📝 Getting LaTeX code for clipboard...');
    
    // Use cached code if provided, otherwise fetch from backend
    let latexCode = cachedCode;
    if (!latexCode) {
      console.log('📡 Fetching LaTeX code from backend...');
      latexCode = await getLatexCode(structuredCV);
    } else {
      console.log('♻️ Using cached LaTeX code');
    }
    
    // Copy to clipboard
    await navigator.clipboard.writeText(latexCode);
    
    console.log('✅ LaTeX code copied to clipboard');
    alert('LaTeX code copied to clipboard!');
  } catch (e) {
    console.error('Failed to copy LaTeX code:', e);
    const errorMsg = e instanceof Error ? e.message : 'Unknown error';
    alert(`Failed to copy LaTeX code: ${errorMsg}\n\nPlease ensure backend is running.`);
    throw e;
  }
}
