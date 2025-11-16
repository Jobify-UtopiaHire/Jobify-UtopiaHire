import React, { useContext, useEffect, ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { AuthContext } from '../AuthContext'

interface ProtectedRouteProps {
  children: ReactNode
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const context = useContext(AuthContext)
  const location = useLocation()

  if (!context) {
    throw new Error('ProtectedRoute must be used within AuthProvider')
  }

  const { user, loading, openAuthModal } = context

  useEffect(() => {
    if (!loading && !user) {
      openAuthModal('login')
    }
  }, [loading, user, openAuthModal])

  if (loading) return <div className="container">Loading...</div>
  if (!user) return <Navigate to="/" replace state={{ from: location }} />
  return <>{children}</>
}
