import React, { useContext } from 'react'
import { Link } from 'react-router-dom'
import { AuthContext } from '../AuthContext'
import { FaRocket, FaBrain, FaChartLine, FaStar } from 'react-icons/fa'
import { motion } from 'framer-motion'
import { cn } from '../lib/utils'

export default function Home() {
  const context = useContext(AuthContext)
  
  if (!context) {
    throw new Error('Home must be used within AuthProvider')
  }

  const { user, openAuthModal } = context

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background handled globally by BackgroundShader */}

      {/* Hero Section */}
      <section className="relative container pt-20 pb-32">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className={cn(
              "inline-flex items-center gap-2 rounded-full border px-4 py-2 backdrop-blur-md mb-6",
              "border-black/20 bg-white/10 hover:bg-black/20 transition-colors"
            )}
          >
            <FaRocket className="text-primary animate-bounce" />
            <span className="text-sm font-medium text-foreground" style={{ color: 'white' }}>AI-Powered Interview Practice</span>
          </motion.div>
          
          {/* Main Heading */}
          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6"
          >
            <span className="text-gradient">Master Your Next Interview</span>
          </motion.h1>
          
          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="text-xl md:text-2xl mb-10 max-w-2xl mx-auto font-medium"
          >
            AI-powered interview practice, CV optimization, job matching, and skills analysis all in one platform.
          </motion.p>
          
          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            {user ? (
              <Link
                to="/dashboard"
                className="px-8 py-4 background-gradient text-white font-semibold rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200"
              >
                Interview
              </Link>
            ) : (
              <>
                <button
                  onClick={() => openAuthModal('register')}
                  className="group relative px-8 py-4 background-gradient text-white font-semibold rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200"
                >
                  <span className="relative z-10">Start Free Trial</span>
                </button>
                <button
                  onClick={() => openAuthModal('login')}
                  className="px-8 py-4 bg-card text-card-foreground font-semibold rounded-xl border-2 border-border hover:border-primary hover:text-primary shadow-sm hover:shadow-md transition-all duration-200"
                >
                  Sign In
                </button>
              </>
            )}
          </motion.div>

          
        </div>
      </section>

      {/* Features Section */}
      <section className="relative container py-20">
        <div className="text-center mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-4xl font-bold text-foreground mb-4"
          >
            Why Choose UtopiaHire?
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-xl text-muted-foreground max-w-3xl mx-auto"
          >
            Everything you need to ace your next interview and advance your career
          </motion.p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {/* Feature 1 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            whileHover={{ y: -8 }}
            className="group relative bg-card rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300"
          >
            <div className="absolute inset-0 background-gradient rounded-2xl opacity-0 group-hover:opacity-5 transition-opacity"></div>
            <div className="relative">
              <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <FaBrain className="text-2xl text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-card-foreground mb-3">AI Interview Coach</h3>
              <p className="text-muted-foreground">Adaptive AI interviewer that provides real-time feedback and detailed analysis of your responses.</p>
            </div>
          </motion.div>

          {/* Feature 2 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            whileHover={{ y: -8 }}
            className="group relative bg-card rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300"
          >
            <div className="absolute inset-0 background-gradient rounded-2xl opacity-0 group-hover:opacity-5 transition-opacity"></div>
            <div className="relative">
              <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <FaChartLine className="text-2xl text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-card-foreground mb-3">Smart CV & Career Tools</h3>
              <p className="text-muted-foreground">AI-powered CV optimization, ATS scoring, job matching, and personalized skill recommendations.</p>
            </div>
          </motion.div>

          {/* Feature 3 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.3 }}
            whileHover={{ y: -8 }}
            className="group relative bg-card rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300"
          >
            <div className="absolute inset-0 background-gradient rounded-2xl opacity-0 group-hover:opacity-5 transition-opacity"></div>
            <div className="relative">
              <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <FaStar className="text-2xl text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-card-foreground mb-3">Comprehensive Analytics</h3>
              <p className="text-muted-foreground">Track progress with detailed dashboards, performance metrics, and personalized insights.</p>
            </div>
          </motion.div>
        </div>
      </section>


      {/* Services Section */}
      <section className="relative container py-20">
        <div className="text-center mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-4xl font-bold text-foreground mb-4"
          >
            Complete Career Development Suite
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-xl text-muted-foreground max-w-3xl mx-auto"
          >
            All the tools you need to succeed in your job search journey
          </motion.p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {[
            {
              title: ' AI Interview Practice',
              desc: 'Adaptive AI interviewer with instant feedback and performance evaluation.'
            },
            {
              title: ' CV Optimization',
              desc: 'AI-powered analysis, ATS compatibility checks, and improvement suggestions.'
            },
            {
              title: ' Smart Job Matching',
              desc: 'Find relevant opportunities with intelligent matching based on your profile.'
            },
            {
              title: ' Skills Gap Analysis',
              desc: 'Identify missing skills and get personalized learning recommendations.'
            },
            {
              title: ' Performance Dashboard',
              desc: 'Track progress with detailed analytics and session history.'
            }
          ].map((service, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              whileHover={{ y: -5 }}
              className="bg-card rounded-xl p-6 border border-border hover:border-primary shadow-sm hover:shadow-lg transition-all"
            >
              <h3 className="text-lg font-semibold text-card-foreground mb-2">{service.title}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">{service.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative container py-20 text-center">
        <motion.h2
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-4xl md:text-5xl font-bold text-foreground mb-6"
        >
          Ready to ace your interview?
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto"
        >
          Join thousands of successful candidates today
        </motion.p>
        {user ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Link
              to="/dashboard"
              className="inline-block px-10 py-5 background-gradient text-white text-lg font-semibold rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200"
            >
              Interview
            </Link>
          </motion.div>
        ) : (
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            onClick={() => openAuthModal('register')}
            className="px-10 py-5 background-gradient text-white text-lg font-semibold rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-200"
          >
            Get Started for Free
          </motion.button>
        )}
      </section>
    </div>
  )
}
