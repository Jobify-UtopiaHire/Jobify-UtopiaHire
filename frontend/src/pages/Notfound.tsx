import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function NotFoundPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <>
      <div className="flex items-center justify-center min-h-screen p-8">
        <Card className="w-full max-w-2xl shadow-2xl">
          <CardHeader className="text-center">
            <div className="text-7xl mb-4">🔍</div>
            <CardTitle className="text-4xl mb-2">404</CardTitle>
            <CardTitle className="text-2xl text-gray-600 dark:text-gray-400">
              {t("notfound.title")}
            </CardTitle>
            <CardDescription className="text-base mt-2">
              {t("notfound.description")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {t("notfound.message")}
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                onClick={() => navigate("/")}
                variant="outline"
                className="flex-1 h-12 text-lg font-bold"
              >
                🏠 {t("notfound.backHome")}
              </Button>
              <Button
                onClick={() => navigate(-1)}
                variant="outline"
                className="flex-1 h-12 text-lg font-bold"
              >
                ⬅️ {t("notfound.goBack")}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}