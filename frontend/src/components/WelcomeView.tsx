import { Mic, CheckCircle, Target, TrendingUp, ChevronRight } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface FeatureCardProps {
  icon: React.ElementType
  title: string
  description: string
}

function FeatureCard({ icon: Icon, title, description }: FeatureCardProps) {
  return (
    <Card className="border-background/20 bg-background/10 backdrop-blur-md transition-all hover:scale-105 hover:shadow-2xl">
      <CardContent className="p-6">
        <div className="w-12 h-12 rounded-full bg-white/20 flex items-center justify-center mb-4">
          <Icon className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        <p className="text-sm leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  )
}

interface WelcomeViewProps {
  onGetStarted: () => void
}

export default function WelcomeView({ onGetStarted }: WelcomeViewProps) {
  const { t } = useTranslation()

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 relative overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-[-10%] right-[-5%] w-[400px] h-[400px] bg-white/10 rounded-full blur-[100px]" />
      <div className="absolute bottom-[-10%] left-[-5%] w-[400px] h-[400px] bg-white/10 rounded-full blur-[100px]" />

      <div className="max-w-6xl w-full relative z-10 translate-y-20">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="w-30 h-30 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-8 backdrop-blur-md border-2 border-white/30 p-8">
            <Mic className="w-16 h-16" />
          </div>

          <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold mb-4 drop-shadow-lg">
            {t('welcome.title')}
          </h1>

          <p className="text-xl md:text-2xl mb-12 max-w-2xl mx-auto leading-relaxed">
            {t('welcome.subtitle')}
          </p>

          <Button
            onClick={onGetStarted}
            size="lg"
            className="shadow-2xl text-lg px-8 py-6 h-auto rounded-full transition-all hover:scale-105"
          >
            {t('welcome.getStarted')}
            <ChevronRight className="ml-2" />
          </Button>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          <FeatureCard
            icon={Target}
            title={t('welcome.features.personalizedQuestions.title')}
            description={t('welcome.features.personalizedQuestions.description')}
          />
          <FeatureCard
            icon={CheckCircle}
            title={t('welcome.features.realTimeFeedback.title')}
            description={t('welcome.features.realTimeFeedback.description')}
          />
          <FeatureCard
            icon={TrendingUp}
            title={t('welcome.features.detailedReports.title')}
            description={t('welcome.features.detailedReports.description')}
          />
        </div>

        {/* Bottom info */}
        <div className="text-center">
          <p className="text-background/70 text-sm">{t('welcome.footer')}</p>
        </div>
      </div>
    </div>
  )
}
