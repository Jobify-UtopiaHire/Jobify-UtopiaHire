import React, { useContext, useEffect, useMemo, useState } from 'react'
import { AuthContext } from '../AuthContext'
import { useServices } from '../ServiceContext'
import { useNavigate } from 'react-router-dom'
import { FaComments, FaFileAlt, FaBriefcase, FaChartLine } from 'react-icons/fa'
import API from '@/api'
import type { InterviewItem, UserInterviewsResponse } from '@/types'
import { Button } from '@/components/ui/button'

export default function Dashboard() {
  const context = useContext(AuthContext)
  const { history, clearHistory } = useServices()
  const navigate = useNavigate()
  const [loadingInterviews, setLoadingInterviews] = useState(false)
  const [interviews, setInterviews] = useState<InterviewItem[]>([])

  if (!context) {
    throw new Error('Dashboard must be used within AuthProvider')
  }

  const { user } = context

  // Fetch interviews for the logged-in user
  useEffect(() => {
    const email = user?.email
    if (!email) return
    
    let active = true
    setLoadingInterviews(true)
    
    API.get(`/interviews/email/${encodeURIComponent(email)}`)
      .then((res) => {
        if (!active) return
        const data: UserInterviewsResponse = res.data
        const list = data?.interviews || []
        setInterviews(Array.isArray(list) ? list : [])
      })
      .catch((err) => {
        if (!active) return
        setInterviews([])
      })
      .finally(() => {
        if (active) setLoadingInterviews(false)
      })
    
    return () => {
      active = false
    }
  }, [user?.email])

  const totalInterviews = interviews.length
  const averageScore = useMemo(() => {
    const scores = interviews
      .map(i => (typeof i.interview_score === 'number' ? i.interview_score : null))
      .filter((v): v is number => v !== null)
    if (scores.length === 0) return null
    return Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 10) / 10
  }, [interviews])

  // Build last 6 months counts for a mini bar chart
  const monthlyData = useMemo(() => {
    const now = new Date()
    const months: { key: string; label: string; count: number }[] = []
    for (let i = 5; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1)
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
      const label = d.toLocaleString(undefined, { month: 'short' })
      months.push({ key, label, count: 0 })
    }
    const map = new Map(months.map(m => [m.key, m]))
    for (const it of interviews) {
      if (!it.created_at) continue
      const d = new Date(it.created_at)
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
      const item = map.get(key)
      if (item) item.count += 1
    }
    return months
  }, [interviews])

  const lastFour = useMemo(() => {
    return [...interviews]
      .sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime())
      .slice(0, 4)
  }, [interviews])

  // Calculate score distribution for circular chart (scores are out of 10)
  const scoreDistribution = useMemo(() => {
    const ranges = { excellent: 0, good: 0, fair: 0, poor: 0 }
    interviews.forEach(i => {
      const score = i.interview_score
      if (score === null || score === undefined) return
      if (score >= 8) ranges.excellent++
      else if (score >= 6) ranges.good++
      else if (score >= 4) ranges.fair++
      else ranges.poor++
    })
    return ranges
  }, [interviews])

  const [showAllInterviews, setShowAllInterviews] = useState(false)

  return (
    <div className="relative min-h-[60vh]">
      <section className="container pt-12 pb-6">
        <h2 className="text-3xl font-bold text-foreground">Welcome back{user ? `, ${user.name}` : ''}</h2>
        <p className="text-muted-foreground mt-2">Pick a tool to continue your preparation</p>
      </section>

      <section className="container pb-16 grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* AI Interviewer */}
        <div className="group bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/30 dark:to-pink-950/30 rounded-2xl p-6 border border-purple-200 dark:border-purple-800 hover:border-purple-400 dark:hover:border-purple-600 shadow-sm hover:shadow-lg transition-all">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mb-4">
            <FaComments className="text-white" />
          </div>
          <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100">AI Interviewer</h3>
          <p className="text-purple-700 dark:text-purple-300 mt-1">Practice behavioral and technical interviews with instant AI feedback.</p>
          <button onClick={() => navigate('/interviewer/setup')} className="mt-4 px-4 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:from-purple-600 hover:to-pink-600 transition-all">Start session</button>
        </div>

        {/* CV Reviewer */}
        <div className="group bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 rounded-2xl p-6 border border-green-200 dark:border-green-800 hover:border-green-400 dark:hover:border-green-600 shadow-sm hover:shadow-lg transition-all">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mb-4">
            <FaFileAlt className="text-white" />
          </div>
          <h3 className="text-lg font-semibold text-green-900 dark:text-green-100">CV Reviewer & Enhancer</h3>
          <p className="text-green-700 dark:text-green-300 mt-1">Upload your CV to get clarity, impact, and keyword suggestions.</p>
          <button onClick={() => navigate('/cv')} className="mt-4 px-4 py-2 rounded-lg bg-gradient-to-r from-green-500 to-emerald-500 text-white font-medium hover:from-green-600 hover:to-emerald-600 transition-all">Upload CV</button>
        </div>

        {/* Job Matcher */}
        <div className="group bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-950/30 dark:to-amber-950/30 rounded-2xl p-6 border border-orange-200 dark:border-orange-800 hover:border-orange-400 dark:hover:border-orange-600 shadow-sm hover:shadow-lg transition-all">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center mb-4">
            <FaBriefcase className="text-white" />
          </div>
          <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-100">Job Matcher</h3>
          <p className="text-orange-700 dark:text-orange-300 mt-1">Discover tailored roles that fit your profile and preparation level.</p>
          <button onClick={() => navigate('/jobs')} className="mt-4 px-4 py-2 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 text-white font-medium hover:from-orange-600 hover:to-amber-600 transition-all">Find jobs</button>
        </div>

        {/* Skills Gap Analyzer - NEW */}
        <div className="group bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-950/30 dark:to-blue-950/30 rounded-2xl p-6 border border-cyan-200 dark:border-cyan-800 hover:border-cyan-400 dark:hover:border-cyan-600 shadow-sm hover:shadow-lg transition-all">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center mb-4">
            <FaChartLine className="text-white" />
          </div>
          <h3 className="text-lg font-semibold text-cyan-900 dark:text-cyan-100">Skills Gap Analyzer</h3>
          <p className="text-cyan-700 dark:text-cyan-300 mt-1">Analyze your skills against job requirements and get personalized learning paths.</p>
          <button onClick={() => navigate('/skills-gap')} className="mt-4 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium hover:from-cyan-600 hover:to-blue-600 transition-all">Analyze Skills</button>
        </div>
      </section>

      {/* Skills Gap Analysis Spotlight */}
      <section className="container pb-12">
        <div className="bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-purple-500/10 rounded-3xl p-8 border-2 border-cyan-200 dark:border-cyan-800 relative overflow-hidden">
          {/* Decorative background elements */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-cyan-400/20 to-blue-400/20 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-gradient-to-tr from-purple-400/20 to-blue-400/20 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2"></div>
          
          <div className="relative z-10 flex flex-col md:flex-row items-center gap-6">
            <div className="flex-shrink-0">
              <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center shadow-xl">
                <FaChartLine className="text-white text-4xl" />
              </div>
            </div>
            
            <div className="flex-1 text-center md:text-left">
              <h3 className="text-2xl md:text-3xl font-bold text-foreground mb-2">
                Discover Your Skills Gap
              </h3>
              <p className="text-muted-foreground text-lg mb-4">
                Upload your CV and a job description to get AI-powered insights on your strengths, 
                gaps, and personalized learning recommendations to close the gap.
              </p>
              <div className="flex flex-wrap gap-3 justify-center md:justify-start">
                <div className="flex items-center gap-2 text-sm bg-white/50 dark:bg-gray-800/50 px-3 py-1.5 rounded-full border border-cyan-200 dark:border-cyan-800">
                  <span className="text-green-500">✓</span>
                  <span>AI-Powered Analysis</span>
                </div>
                <div className="flex items-center gap-2 text-sm bg-white/50 dark:bg-gray-800/50 px-3 py-1.5 rounded-full border border-cyan-200 dark:border-cyan-800">
                  <span className="text-green-500">✓</span>
                  <span>Learning Paths</span>
                </div>
                <div className="flex items-center gap-2 text-sm bg-white/50 dark:bg-gray-800/50 px-3 py-1.5 rounded-full border border-cyan-200 dark:border-cyan-800">
                  <span className="text-green-500">✓</span>
                  <span>Resume Tips</span>
                </div>
              </div>
            </div>
            
            <div className="flex-shrink-0">
              <button 
                onClick={() => navigate('/skills-gap')} 
                className="px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold text-lg hover:from-cyan-600 hover:to-blue-600 transition-all shadow-xl hover:shadow-2xl hover:scale-105 flex items-center gap-3"
              >
                <span>Start Analysis</span>
                <span className="text-2xl">→</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Interview Profile */}
      <section className="container pb-16">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-foreground">Your Interview Profile</h3>
          {loadingInterviews && <span className="text-sm text-muted-foreground">Loading…</span>}
        </div>

        {/* Stats Grid with Charts */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Total Interviews - Colorful Circle Chart */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950/30 dark:to-blue-900/30 rounded-2xl p-6 border border-blue-200 dark:border-blue-800">
            <div className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-3">Total Interviews</div>
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="text-4xl font-bold text-blue-900 dark:text-blue-100">{totalInterviews}</div>
                <div className="text-sm text-blue-600/70 dark:text-blue-400/70 mt-1">out of 10</div>
                <div className="text-xs text-blue-600/70 dark:text-blue-400/70 mt-2">
                  {totalInterviews === 0 ? 'Start your first interview' : `${10 - totalInterviews} remaining`}
                </div>
              </div>
              <div className="relative w-24 h-24 ml-3">
                <svg className="transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="12"
                    className="text-blue-200 dark:text-blue-800"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="12"
                    strokeDasharray={`${(Math.min(totalInterviews, 10) / 10) * 251} 251`}
                    className="text-blue-500 dark:text-blue-400"
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">{totalInterviews}</div>
                  <div className="text-xs text-blue-600/70 dark:text-blue-400/70">/10</div>
                </div>
              </div>
            </div>
          </div>

          {/* Average Score */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950/30 dark:to-purple-900/30 rounded-2xl p-6 border border-purple-200 dark:border-purple-800">
            <div className="text-sm font-medium text-purple-600 dark:text-purple-400 mb-3">Average Score</div>
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="text-4xl font-bold text-purple-900 dark:text-purple-100">{averageScore ?? 'N/A'}</div>
                <div className="text-sm text-purple-600/70 dark:text-purple-400/70 mt-1">out of 10</div>
                {averageScore === null && (
                  <div className="text-xs text-purple-600/70 dark:text-purple-400/70 mt-2">No scores yet</div>
                )}
              </div>
              <div className="relative w-24 h-24 ml-3">
                <svg className="transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="12"
                    className="text-purple-200 dark:text-purple-800"
                  />
                  {averageScore !== null && (
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="12"
                      strokeDasharray={`${(averageScore / 10) * 251} 251`}
                      className="text-purple-500 dark:text-purple-400"
                      strokeLinecap="round"
                    />
                  )}
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-lg font-bold text-purple-600 dark:text-purple-400">{averageScore ?? '—'}</div>
                  {averageScore !== null && <div className="text-xs text-purple-600/70 dark:text-purple-400/70">/10</div>}
                </div>
              </div>
            </div>
          </div>

          {/* Score Distribution Donut Chart */}
          <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-950/30 dark:to-emerald-900/30 rounded-2xl p-6 border border-emerald-200 dark:border-emerald-800">
            <div className="text-sm font-medium text-emerald-600 dark:text-emerald-400 mb-3">Score Distribution</div>
            <div className="flex flex-col items-center">
              <div className="relative w-32 h-32 mb-3">
                {(() => {
                  const total = Object.values(scoreDistribution).reduce((a, b) => a + b, 0)
                  if (total === 0) {
                    return (
                      <div className="w-full h-full rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                        <span className="text-xs text-gray-500 text-center px-2">No data</span>
                      </div>
                    )
                  }
                  const radius = 45
                  const circumference = 2 * Math.PI * radius
                  let currentOffset = 0
                  
                  const segments = [
                    { label: 'excellent', count: scoreDistribution.excellent, color: '#10b981', darkColor: '#34d399' },
                    { label: 'good', count: scoreDistribution.good, color: '#3b82f6', darkColor: '#60a5fa' },
                    { label: 'fair', count: scoreDistribution.fair, color: '#f59e0b', darkColor: '#fbbf24' },
                    { label: 'poor', count: scoreDistribution.poor, color: '#ef4444', darkColor: '#f87171' }
                  ]
                  
                  return (
                    <>
                      <svg viewBox="0 0 100 100" className="transform -rotate-90 w-full h-full">
                        {/* Background circle */}
                        <circle
                          cx="50"
                          cy="50"
                          r={radius}
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="10"
                          className="text-gray-200 dark:text-gray-700"
                        />
                        {/* Colored segments */}
                        {segments.map((segment, idx) => {
                          if (segment.count === 0) return null
                          const percent = (segment.count / total)
                          const dashArray = percent * circumference
                          const dashOffset = -currentOffset
                          currentOffset += dashArray
                          
                          return (
                            <circle
                              key={segment.label}
                              cx="50"
                              cy="50"
                              r={radius}
                              fill="none"
                              stroke={segment.color}
                              strokeWidth="10"
                              strokeDasharray={`${dashArray} ${circumference}`}
                              strokeDashoffset={dashOffset}
                              strokeLinecap="round"
                              className="transition-all duration-300 hover:stroke-[12]"
                            />
                          )
                        })}
                      </svg>
                      {/* Center text */}
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <div className="text-2xl font-bold text-emerald-900 dark:text-emerald-100">{total}</div>
                        <div className="text-xs text-emerald-600 dark:text-emerald-400">Total</div>
                      </div>
                    </>
                  )
                })()}
              </div>
              {/* Legend */}
              <div className="grid grid-cols-2 gap-2 w-full text-xs">
                <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 rounded-lg px-2 py-1">
                  <div className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0"></div>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold truncate">Excellent</div>
                    <div className="text-muted-foreground">{scoreDistribution.excellent}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 rounded-lg px-2 py-1">
                  <div className="w-3 h-3 rounded-full bg-blue-500 flex-shrink-0"></div>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold truncate">Good</div>
                    <div className="text-muted-foreground">{scoreDistribution.good}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 rounded-lg px-2 py-1">
                  <div className="w-3 h-3 rounded-full bg-amber-500 flex-shrink-0"></div>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold truncate">Fair</div>
                    <div className="text-muted-foreground">{scoreDistribution.fair}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 bg-white/50 dark:bg-gray-800/50 rounded-lg px-2 py-1">
                  <div className="w-3 h-3 rounded-full bg-red-500 flex-shrink-0"></div>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold truncate">Poor</div>
                    <div className="text-muted-foreground">{scoreDistribution.poor}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Last Interview */}
          <div className="bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-950/30 dark:to-amber-900/30 rounded-2xl p-6 border border-amber-200 dark:border-amber-800">
            <div className="text-sm font-medium text-amber-600 dark:text-amber-400 mb-3">Last Interview</div>
            <div className="flex items-center justify-center h-24">
              <div className="text-center">
                <div className="text-4xl font-bold text-amber-900 dark:text-amber-100">
                  {interviews.length ? new Date((interviews
                    .map(i => i.created_at)
                    .filter(Boolean)
                    .sort()
                    .slice(-1)[0]) as string).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : 'None'}
                </div>
                {interviews.length === 0 ? (
                  <div className="text-xs text-amber-600/70 dark:text-amber-400/70 mt-2">No interviews yet</div>
                ) : (
                  <div className="text-sm text-amber-600/70 dark:text-amber-400/70 mt-2">Most recent</div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Bar chart: interviews per month (last 6 months) */}
        <div className="bg-card rounded-2xl p-6 border border-border mb-8">
          <div className="text-lg font-semibold text-foreground mb-4">Interview Activity (Last 6 Months)</div>
          <div className="flex items-end justify-around gap-2 h-48">
            {(() => {
              const max = Math.max(1, ...monthlyData.map(m => m.count))
              return monthlyData.map((m, idx) => {
                const colors = ['from-blue-500 to-blue-600', 'from-purple-500 to-purple-600', 'from-pink-500 to-pink-600', 'from-red-500 to-red-600', 'from-orange-500 to-orange-600', 'from-amber-500 to-amber-600']
                return (
                  <div key={m.key} className="flex flex-col items-center justify-end h-full flex-1 max-w-20">
                    <div className="text-sm font-bold text-foreground mb-1">{m.count > 0 ? m.count : ''}</div>
                    <div
                      className={`w-full rounded-t-lg bg-gradient-to-t ${colors[idx % colors.length]} transition-all hover:opacity-80 cursor-pointer shadow-lg`}
                      style={{ height: `${(m.count / max) * 100}%`, minHeight: m.count > 0 ? '8px' : '0' }}
                      title={`${m.label}: ${m.count} interviews`}
                    />
                    <div className="mt-2 text-xs font-medium text-muted-foreground">{m.label}</div>
                  </div>
                )
              })
            })()}
          </div>
        </div>

        {/* Recent Interviews */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-foreground">Recent Interviews</h4>
            {interviews.length > 4 && !showAllInterviews && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAllInterviews(true)}
                className="text-sm"
              >
                See All ({interviews.length})
              </Button>
            )}
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {(showAllInterviews ? interviews.sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()) : lastFour).length === 0 ? (
              <div className="col-span-2 text-center py-12 bg-card rounded-2xl border border-border">
                <div className="text-6xl mb-4">🎤</div>
                <p className="text-lg font-semibold text-muted-foreground mb-2">No interviews yet</p>
                <p className="text-sm text-muted-foreground mb-4">Start an AI interview to see your history here</p>
                <Button onClick={() => navigate('/interviewer/setup')} className="mt-2">
                  Start First Interview
                </Button>
              </div>
            ) : (
              (showAllInterviews ? interviews.sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()) : lastFour).map((it) => {
                const d = it.created_at ? new Date(it.created_at) : null
                const score = typeof it.interview_score === 'number' ? it.interview_score : null
                const scoreColor = score === null ? 'text-muted-foreground' : score >= 8 ? 'text-green-600 dark:text-green-400' : score >= 6 ? 'text-amber-600 dark:text-amber-400' : 'text-red-600 dark:text-red-400'
                const scoreBg = score === null ? 'bg-gray-100 dark:bg-gray-800' : score >= 8 ? 'bg-green-50 dark:bg-green-950/30' : score >= 6 ? 'bg-amber-50 dark:bg-amber-950/30' : 'bg-red-50 dark:bg-red-950/30'
                return (
                  <div 
                    key={it.id} 
                    className="bg-card rounded-2xl p-5 border border-border hover:shadow-xl hover:scale-[1.02] transition-all duration-200 cursor-pointer group"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="text-lg font-bold text-card-foreground group-hover:text-primary transition-colors">{it.job_title || 'Interview'}</div>
                        <div className="text-xs text-muted-foreground mt-1">{d ? d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : '—'}</div>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-sm font-bold ${scoreBg} ${scoreColor}`}>
                        {score !== null ? `${score}/10` : 'N/A'}
                      </div>
                    </div>
                    {it.conclusion && (
                      <div className="mt-3 text-sm text-muted-foreground line-clamp-2 leading-relaxed">
                        {it.conclusion}
                      </div>
                    )}
                    <div className="mt-4 pt-3 border-t border-border/50 flex items-center gap-2 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1">
                        <div className={`w-2 h-2 rounded-full ${score === null ? 'bg-gray-400' : score >= 8 ? 'bg-green-500' : score >= 6 ? 'bg-amber-500' : 'bg-red-500'}`}></div>
                        {score === null ? 'Not scored' : score >= 8 ? 'Excellent' : score >= 6 ? 'Good' : 'Needs improvement'}
                      </span>
                    </div>
                  </div>
                )
              })
            )}
          </div>
          {showAllInterviews && interviews.length > 0 && (
            <div className="mt-4 text-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAllInterviews(false)}
                className="text-sm"
              >
                Show Less
              </Button>
            </div>
          )}
        </div>
      </section>

      {/* History */}
      <section className="container pb-16">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-foreground">Recent activity</h3>
          {history.length > 0 && (
            <button onClick={clearHistory} className="text-sm px-3 py-1.5 rounded-lg border border-border hover:border-red-400 hover:text-red-400">Clear</button>
          )}
        </div>
        {history.length === 0 ? (
          <p className="text-muted-foreground">No activity yet. Your recent sessions and tools will appear here.</p>
        ) : (
          <ul className="space-y-3">
            {history.slice(0, 8).map(item => (
              <li key={item.id} className="bg-card rounded-xl p-4 border border-border flex items-center justify-between">
                <div>
                  <div className="font-medium text-card-foreground capitalize">{item.type?.replace('-', ' ') || 'Activity'}</div>
                  <div className="text-sm text-muted-foreground">{item.title} • {new Date(item.ts).toLocaleString()}</div>
                </div>
                {item.meta?.results !== undefined && (
                  <div className="text-sm text-muted-foreground">{item.meta.results} results</div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
