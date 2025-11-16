import React, { useContext } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { FaVideo } from 'react-icons/fa'
import { AuthContext } from '../AuthContext'

export default function Nav() {
  const context = useContext(AuthContext)
  const location = useLocation()
  
  if (!context) {
    throw new Error('Nav must be used within AuthProvider')
  }

  const { user, logout, openAuthModal } = context
  const isCVToolPage = location.pathname === '/cv'

  return (
    <header className={`sticky top-0 z-40 ${isCVToolPage ? 'bg-transparent' : 'bg-black/40 backdrop-blur-md'} border-b border-white/10 shadow-[0_0_40px_rgba(153,75,255,0.15)]`}>
      <div className="container">
        <div className="grid grid-cols-3 items-center py-4">
          {/* left: brand + interview cta */}
          <div className="justify-self-start flex items-center gap-4">
            <Link 
              className="text-2xl font-extrabold tracking-tight bg-clip-text  bg-gradient-to-r from-[#36cde1] via-[#a56bff] to-white hover:opacity-90 transition-opacity" 
              to="/"
            >
              UtopiaHire
            </Link>
          </div>

          {/* center: main links */}
          <nav className="justify-self-center flex items-center gap-8">
            <Link className="text-sm font-medium text-white/70 hover:text-white transition-colors" to="/">Home</Link>
            <Link className="text-sm font-medium text-white/70 hover:text-white transition-colors" to="/about">About</Link>
            <Link className="text-sm font-medium text-white/70 hover:text-white transition-colors" to="/pricing">Pricing</Link>
          </nav>

          {/* right: auth / dashboard */}
          <div className="justify-self-end flex items-center gap-4">
            
            {user && (
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r from-primary-600 to-primary-700 text-white text-sm font-semibold shadow-md hover:shadow-lg transform hover:scale-105 transition-all"
                aria-label="Go to Interview dashboard"
              >
                <FaVideo className="text-xs" />
                Interview
              </Link>
            )}
            {user ? (
              <button
                className="text-sm font-medium text-white/70 hover:text-white transition-colors"
                onClick={logout}
              >
                Logout
              </button>
            ) : (
              <>
                <button
                  className="text-sm font-medium text-white/70 hover:text-white transition-colors"
                  onClick={() => openAuthModal('login')}
                >
                  Login
                </button>
                <button
                  className="px-5 py-2 bg-gradient-to-r from-primary-600 to-primary-700 text-white text-sm font-semibold rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-200"
                  onClick={() => openAuthModal('register')}
                >
                  Get Started
                </button>
              </>
            )}
            {user && (
              <div className="hidden sm:flex items-center gap-2 text-white/80">
                <div className="w-7 h-7 rounded-full bg-white/10 border border-white/20 flex items-center justify-center text-xs font-semibold">
                  {user.name?.charAt(0)?.toUpperCase() || 'U'}
                </div>
                <span className="text-sm">Hi, {user.name?.split(' ')[0] || 'User'}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
