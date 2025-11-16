import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react'
import { InterviewConfig, HistoryEntry } from './types'

interface ServiceContextType {
  cvFile: File | null
  setCvFile: (file: File | null) => void
  cvText: string
  setCvText: (text: string) => void
  interviewConfig: InterviewConfig
  setInterviewConfig: (config: InterviewConfig) => void
  history: HistoryEntry[]
  addHistory: (entry: Omit<HistoryEntry, 'id' | 'ts'>) => void
  clearHistory: () => void
}

const ServiceContext = createContext<ServiceContextType | undefined>(undefined)

export const useServices = () => {
  const context = useContext(ServiceContext)
  if (!context) {
    throw new Error('useServices must be used within a ServiceProvider')
  }
  return context
}

export function ServiceProvider({ children }: { children: ReactNode }) {
  const [cvFile, setCvFile] = useState<File | null>(null)
  const [cvText, setCvText] = useState('')
  const [interviewConfig, setInterviewConfig] = useState<InterviewConfig>({
    jobTitle: '',
    language: 'English',
    company: '',
    experience: '',
    seniority: 'Junior',
    description: ''
  })
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('history') || '[]')
    } catch {
      return []
    }
  })

  useEffect(() => {
    localStorage.setItem('history', JSON.stringify(history))
  }, [history])

  const addHistory = (entry: Omit<HistoryEntry, 'id' | 'ts'>) => {
    const withId: HistoryEntry = {
      id: Date.now(),
      ts: new Date().toISOString(),
      ...entry
    }
    setHistory(prev => [withId, ...prev].slice(0, 100))
  }

  const clearHistory = () => setHistory([])

  const value: ServiceContextType = {
    cvFile,
    setCvFile,
    cvText,
    setCvText,
    interviewConfig,
    setInterviewConfig,
    history,
    addHistory,
    clearHistory,
  }

  return <ServiceContext.Provider value={value}>{children}</ServiceContext.Provider>
}
