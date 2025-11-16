import React from 'react'
import { useLocation } from 'react-router-dom'

export default function Footer() {
  const location = useLocation()
  const isCVToolPage = location.pathname === '/cv'

  return (
    <footer className={`mt-16 py-6 ${isCVToolPage ? 'bg-transparent' : 'bg-black/40 backdrop-blur-md'} border-t border-white/10`}>
      <div className="container text-center text-sm text-white/60">
        <p>© {new Date().getFullYear()} UtopiaHire</p>
      </div>
    </footer>
  )
}
