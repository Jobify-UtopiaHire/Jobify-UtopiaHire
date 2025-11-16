import React from 'react'
import { FaCheck } from 'react-icons/fa'
import { motion } from 'framer-motion'
import { cn } from '../lib/utils'

interface Tier {
  name: string
  price: string
  period: string
  highlight: boolean
  features: string[]
  cta: string
}

const tiers: Tier[] = [
  {
    name: 'Free',
    price: '0 DT',
    period: '/mo',
    highlight: false,
    features: [
      '5 mock interviews / month',
      'Basic feedback',
      'Community support'
    ],
    cta: 'Get started'
  },
  {
    name: 'Pro',
    price: '11.9 DT',
    period: '/mo',
    highlight: true,
    features: [
      'Up to 10 seats',
      'Advanced AI feedback & tips',
      'CV reviewer & enhancer',
      'Job matcher access',
      'Priority support'
    ],
    cta: 'Start free trial'
  },
  {
    name: 'Team',
    price: '49.9 DT',
    period: '/mo',
    highlight: false,
    features: [
      'Unlimited mock interviews',
      'Team analytics & insights',
      'Shared questions & templates',
      'Admin controls'
    ],
    cta: 'Contact sales'
  }
]

export default function Pricing() {
  return (
    <div className="relative min-h-screen">
      {/* Background handled globally by BackgroundShader */}

      <section className="container pt-16 pb-10 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-4xl md:text-5xl font-extrabold tracking-tight"
        >
          <span className="text-gradient">Simple, transparent pricing</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto"
        >
          Choose a plan that scales with your ambition.
        </motion.p>
      </section>

      <section className="container pb-20">
        <div className="grid gap-6 md:grid-cols-3">
          {tiers.map((t, index) => (
            <motion.div
              key={t.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -8 }}
              className={cn(
                "relative rounded-2xl p-8 border transition-all",
                t.highlight
                  ? "border-primary shadow-xl bg-card"
                  : "border-border shadow-md bg-card hover:shadow-2xl"
              )}
            >
              {t.highlight && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 text-xs font-semibold background-gradient text-white px-3 py-1 rounded-full shadow">
                  Most Popular
                </div>
              )}
              <div className="mb-6">
                <div className="text-sm font-semibold text-muted-foreground">{t.name}</div>
                <div className="mt-2 flex items-end justify-center gap-1">
                  <div className="text-4xl font-extrabold text-gradient">{t.price}</div>
                  <div className="text-muted-foreground mb-1">{t.period}</div>
                </div>
              </div>

              <ul className="space-y-3 text-foreground mb-8">
                {t.features.map((f) => (
                  <li key={f} className="flex items-start gap-3">
                    <FaCheck className="text-green-600 mt-1 flex-shrink-0" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>

              <button
                className={cn(
                  "w-full py-3 rounded-xl font-semibold shadow transition-all",
                  t.highlight
                    ? "background-gradient text-white hover:shadow-xl transform hover:scale-105"
                    : "bg-card border-2 border-border hover:border-primary hover:text-primary"
                )}
              >
                {t.cta}
              </button>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  )
}
