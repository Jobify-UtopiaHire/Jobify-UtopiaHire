import React, { useEffect, useRef, useState, Suspense } from "react";
import clsx from "clsx";

// Use React.lazy to dynamically load shader components from @shadergradient/react.
const ShaderGradientCanvas = React.lazy(async () => {
  const mod = await import("@shadergradient/react");
  return { default: (mod as any).ShaderGradientCanvas } as any;
});

const ShaderGradient = React.lazy(async () => {
  const mod = await import("@shadergradient/react");
  return { default: (mod as any).ShaderGradient } as any;
});

export interface BackgroundShaderProps {
  className?: string;
  /**
   * Optional explicit theme. If omitted, the component falls back to the user's
   * system preference (prefers-color-scheme).
   */
  theme?: "light" | "dark" | null;
}

export function BackgroundShader({ className, theme: themeProp = null }: BackgroundShaderProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const preventDefault = (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
    };

    // Prevent a set of touch/gesture/wheel events that would interact with the canvas
    const events: Array<{ name: string; options?: AddEventListenerOptions }> = [
      { name: "touchstart", options: { passive: false } },
      { name: "touchmove", options: { passive: false } },
      { name: "touchend", options: { passive: false } },
      { name: "wheel", options: { passive: false } },
      { name: "gesturestart", options: { passive: false } },
      { name: "gesturechange", options: { passive: false } },
      { name: "gestureend", options: { passive: false } },
      { name: "pinch", options: { passive: false } },
      { name: "pinchstart", options: { passive: false } },
      { name: "pinchend", options: { passive: false } },
      { name: "pinchcancel", options: { passive: false } },
    ];

    events.forEach((ev) => container.addEventListener(ev.name, preventDefault as EventListener, ev.options));

    // Prevent double-tap to zoom
    let lastTap = 0;
    const handleTouchStart = (e: any) => {
      const now = Date.now();
      if (now - lastTap < 300) {
        // best-effort preventDefault; event may be TouchEvent or synthetic
        if (e && typeof e.preventDefault === "function") e.preventDefault();
      }
      lastTap = now;
    };

    container.addEventListener("touchstart", handleTouchStart as EventListener, { passive: false });

    return () => {
      events.forEach((ev) => container.removeEventListener(ev.name, preventDefault as EventListener));
      container.removeEventListener("touchstart", handleTouchStart as EventListener);
    };
  }, []);

  const [detectedTheme, setDetectedTheme] = useState<"light" | "dark">(() => {
    if (themeProp === "light" || themeProp === "dark") return themeProp;
    if (typeof window === "undefined") return "light";
    try {
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    } catch (e) {
      return "light";
    }
  });

  useEffect(() => {
    if (themeProp === "light" || themeProp === "dark") {
      setDetectedTheme(themeProp);
      return;
    }

    if (typeof window === "undefined" || !window.matchMedia) return;

    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const handler = (ev: MediaQueryListEvent) => setDetectedTheme(ev.matches ? "light" : "dark");
    // Older browsers use addListener
    if (typeof mq.addEventListener === "function") mq.addEventListener("change", handler as any);
    else (mq as any).addListener(handler as any);

    return () => {
      if (typeof mq.removeEventListener === "function") mq.removeEventListener("change", handler as any);
      else (mq as any).removeListener(handler as any);
    };
  }, [themeProp]);

  const gradientUrl =
    detectedTheme === "light"
      ? "https://shadergradient-web.vercel.app/customize?animate=on&axesHelper=off&bgColor1=%23000000&bgColor2=%23000000&brightness=1.1&cAzimuthAngle=170&cDistance=4.4&cPolarAngle=70&cameraZoom=1&color1=%23b5e1ff&color2=%23f6d1ff&color3=%23ffffff&destination=onCanvas&embedMode=off&envPreset=city&format=gif&fov=45&frameRate=10&gizmoHelper=hide&grain=on&lightType=3d&pixelDensity=1&positionX=0&positionY=0.9&positionZ=-0.3&range=enabled&rangeEnd=40&rangeStart=0&reflection=0.1&rotationX=45&rotationY=0&rotationZ=0&shader=defaults&type=waterPlane&uAmplitude=0&uDensity=1.2&uFrequency=0&uSpeed=0.1&uStrength=3.4&uTime=0&wireframe=false"
      : "https://www.shadergradient.co/customize?animate=on&axesHelper=off&gizmoHelper=hide&bgColor1=%23000000&bgColor2=%23000000&brightness=1&cAzimuthAngle=180&cDistance=5.7&cPolarAngle=115&cameraZoom=1&color1=%23268c9a&color2=%239224b3&color3=%23000000&destination=onCanvas&embedMode=off&envPreset=city&format=gif&fov=45&frameRate=10&grain=on&lightType=3d&pixelDensity=1&positionX=-0.5&positionY=0.1&positionZ=-1&range=enabled&rangeEnd=40&rangeStart=0&reflection=0.1&rotationX=0&rotationY=0&rotationZ=235&shader=defaults&type=waterPlane&uAmplitude=0&uDensity=1.1&uFrequency=5.5&uSpeed=0.1&uStrength=2.4&uTime=0.2&wireframe=false&zoomOut=false";

  return (
    <div
      ref={containerRef}
      className={clsx("pointer-events-none touch-none select-none", className)}
      style={
        {
          backgroundColor: "#000",
          touchAction: "none",
          userSelect: "none",
          WebkitUserSelect: "none",
          msUserSelect: "none",
          WebkitTouchCallout: "none",
          WebkitTapHighlightColor: "transparent",
          overscrollBehavior: "none",
          overscrollBehaviorY: "none",
          overscrollBehaviorX: "none",
          WebkitOverflowScrolling: "touch",
        } as React.CSSProperties
      }
    >
      <Suspense fallback={null}>
        <ShaderGradientCanvas
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            height: "100%",
            width: "100%",
            // backgroundColor: "var(--gradient-2)",
            pointerEvents: "none",
            touchAction: "none",
          }}
        >
          <ShaderGradient control="query" urlString={gradientUrl} />
        </ShaderGradientCanvas>
      </Suspense>

      <div
        className="absolute inset-x-0 bottom-0 h-64"
        style={{
          background: `linear-gradient(to top,
            rgba(0,0,0,1) 0%,
            rgba(0,0,0,0.95) 30%,
            rgba(0,0,0,0.8) 50%,
            rgba(0,0,0,0.4) 75%,
            rgba(0,0,0,0.1) 90%,
            transparent 100%
          )`,
        }}
      />
    </div>
  );
}

export default BackgroundShader;
