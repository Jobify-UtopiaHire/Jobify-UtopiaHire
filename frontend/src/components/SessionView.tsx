import {
  useVoiceAssistant,
  BarVisualizer,
  VideoTrack,
} from "@livekit/components-react";
import { Phone, Mic, MicOff, Video, VideoOff } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useLocalParticipant } from "@livekit/components-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface SessionViewProps {
  onDisconnect: () => void;
}

export default function SessionView({ onDisconnect }: SessionViewProps) {
  const { t } = useTranslation();
  const { state: agentState, audioTrack: agentAudioTrack } =
    useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const [isMuted, setIsMuted] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraPosition, setCameraPosition] = useState({
    x: 20,
    y: window.innerHeight - 300,
  }); // Bottom-left
  const [cameraSize, setCameraSize] = useState({ width: 480, height: 270 }); // Bigger initial size
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeCorner, setResizeCorner] = useState<
    "ne" | "nw" | "se" | "sw" | null
  >(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    posX: 0,
    posY: 0,
  });

  // Initialize camera
  useEffect(() => {
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
        setIsCameraOn(false);
      }
    };

    if (isCameraOn) {
      startCamera();
    }

    // Cleanup function
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isCameraOn]);

  const handleToggleCamera = () => {
    if (isCameraOn && videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(!isCameraOn);
  };

  // Handle drag/resize start
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    const target = e.target as HTMLElement;
    // Check if clicking on a resize handle (look for data-corner on the element or its parent)
    const resizeHandle = target.closest("[data-corner]") as HTMLElement;
    const corner = resizeHandle?.getAttribute("data-corner") as
      | "ne"
      | "nw"
      | "se"
      | "sw"
      | null;

    if (corner) {
      // Start resizing from corner
      setIsResizing(true);
      setResizeCorner(corner);
      setResizeStart({
        x: e.clientX,
        y: e.clientY,
        width: cameraSize.width,
        height: cameraSize.height,
        posX: cameraPosition.x,
        posY: cameraPosition.y,
      });
    } else if (!target.closest("button")) {
      // Start dragging (but not if clicking buttons)
      setIsDragging(true);
      setDragOffset({
        x: e.clientX - cameraPosition.x,
        y: e.clientY - cameraPosition.y,
      });
    }
  };

  // Handle dragging and resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      e.preventDefault();

      if (isDragging) {
        requestAnimationFrame(() => {
          setCameraPosition({
            x: e.clientX - dragOffset.x,
            y: e.clientY - dragOffset.y,
          });
        });
      } else if (isResizing && resizeCorner) {
        requestAnimationFrame(() => {
          const deltaX = e.clientX - resizeStart.x;
          const deltaY = e.clientY - resizeStart.y;

          let newWidth = resizeStart.width;
          let newHeight = resizeStart.height;
          let newX = resizeStart.posX;
          let newY = resizeStart.posY;

          // Calculate based on which corner is being dragged
          switch (resizeCorner) {
            case "se": // Bottom-right corner
              newWidth = Math.max(240, resizeStart.width + deltaX);
              newHeight = newWidth * (9 / 16);
              break;
            case "sw": // Bottom-left corner
              newWidth = Math.max(240, resizeStart.width - deltaX);
              newHeight = newWidth * (9 / 16);
              newX = resizeStart.posX + (resizeStart.width - newWidth);
              break;
            case "ne": // Top-right corner
              newWidth = Math.max(240, resizeStart.width + deltaX);
              newHeight = newWidth * (9 / 16);
              newY = resizeStart.posY + (resizeStart.height - newHeight);
              break;
            case "nw": // Top-left corner
              newWidth = Math.max(240, resizeStart.width - deltaX);
              newHeight = newWidth * (9 / 16);
              newX = resizeStart.posX + (resizeStart.width - newWidth);
              newY = resizeStart.posY + (resizeStart.height - newHeight);
              break;
          }

          setCameraSize({ width: newWidth, height: newHeight });
          setCameraPosition({ x: newX, y: newY });
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(false);
      setResizeCorner(null);
    };

    if (isDragging || isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.userSelect = "none"; // Prevent text selection while dragging
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.userSelect = "";
    };
  }, [
    isDragging,
    isResizing,
    dragOffset,
    resizeStart,
    resizeCorner,
    cameraSize,
    cameraPosition,
  ]);

  const handleToggleMic = async () => {
    if (localParticipant) {
      await localParticipant.setMicrophoneEnabled(isMuted);
      setIsMuted(!isMuted);
    }
  };

  const getAgentStateDisplay = () => {
    switch (agentState) {
      case "listening":
        return { text: t("session.states.listening"), color: "bg-green-500" };
      case "thinking":
        return { text: t("session.states.thinking"), color: "bg-amber-500" };
      case "speaking":
        return { text: t("session.states.speaking"), color: "bg-blue-500" };
      default:
        return { text: t("session.states.ready"), color: "bg-gray-500" };
    }
  };

  const stateDisplay = getAgentStateDisplay();

  return (
    <div className="min-h-screen relative w-full flex-1 flex flex-col items-center justify-center overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] bg-white/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-[-20%] left-[-10%] w-[600px] h-[600px] bg-white/10 rounded-full blur-[120px]" />

      {/* Floating Camera View */}
      {isCameraOn && (
        <div
          onMouseDown={handleMouseDown}
          style={{
            position: "fixed",
            left: `${cameraPosition.x}px`,
            top: `${cameraPosition.y}px`,
            width: `${cameraSize.width}px`,
            height: `${cameraSize.height}px`,
            zIndex: 50,
            cursor: isDragging ? "grabbing" : "grab",
            userSelect: "none",
          }}
        >
          <Card className="bg-gray-900 backdrop-blur-md shadow-2xl border-2 border-white/20 rounded-xl h-full w-full overflow-hidden relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
              style={{ transform: "scaleX(-1)" }}
            />

            {/* Control buttons overlay */}
            <div className="absolute top-2 right-2 flex gap-2 z-30">
              <Button
                onClick={handleToggleCamera}
                size="icon"
                variant="secondary"
                className="w-8 h-8 rounded-full bg-black/70 hover:bg-black/90 backdrop-blur-sm relative z-30"
              >
                <VideoOff className="w-4 h-4 text-white" />
              </Button>
            </div>

            {/* Drag indicator */}
            <div className="absolute top-2 left-2 pointer-events-none z-10">
              <div className="bg-black/70 backdrop-blur-sm rounded-full px-3 py-1 text-xs text-white font-medium">
                📹 Drag • Resize corners
              </div>
            </div>

            {/* Resize handles on all 4 corners */}
            {/* Top-left corner */}
            <div
              data-corner="nw"
              className="absolute top-0 left-0 w-8 h-8 cursor-nw-resize group z-20"
              style={{ touchAction: "none" }}
            >
              <svg
                className="w-full h-full text-white/40 group-hover:text-white/80 transition-colors pointer-events-none"
                viewBox="0 0 32 32"
              >
                <path
                  d="M 8 2 L 2 2 L 2 8"
                  stroke="currentColor"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M 2 2 L 10 10"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  opacity="0.5"
                />
              </svg>
            </div>

            {/* Top-right corner */}
            <div
              data-corner="ne"
              className="absolute top-0 right-0 w-8 h-8 cursor-ne-resize group z-20"
              style={{ touchAction: "none" }}
            >
              <svg
                className="w-full h-full text-white/40 group-hover:text-white/80 transition-colors pointer-events-none"
                viewBox="0 0 32 32"
              >
                <path
                  d="M 24 2 L 30 2 L 30 8"
                  stroke="currentColor"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M 30 2 L 22 10"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  opacity="0.5"
                />
              </svg>
            </div>

            {/* Bottom-left corner */}
            <div
              data-corner="sw"
              className="absolute bottom-0 left-0 w-8 h-8 cursor-sw-resize group z-20"
              style={{ touchAction: "none" }}
            >
              <svg
                className="w-full h-full text-white/40 group-hover:text-white/80 transition-colors pointer-events-none"
                viewBox="0 0 32 32"
              >
                <path
                  d="M 2 24 L 2 30 L 8 30"
                  stroke="currentColor"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M 2 30 L 10 22"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  opacity="0.5"
                />
              </svg>
            </div>

            {/* Bottom-right corner */}
            <div
              data-corner="se"
              className="absolute bottom-0 right-0 w-8 h-8 cursor-nwse-resize group z-20"
              style={{ touchAction: "none" }}
            >
              <svg
                className="w-full h-full text-white/40 group-hover:text-white/80 transition-colors pointer-events-none"
                viewBox="0 0 32 32"
              >
                <path
                  d="M 30 24 L 30 30 L 24 30"
                  stroke="currentColor"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M 30 30 L 22 22"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  strokeLinecap="round"
                  opacity="0.5"
                />
              </svg>
            </div>
          </Card>
        </div>
      )}

      {/* Camera Off Button - show when camera is off */}
      {!isCameraOn && (
        <div className="fixed top-4 right-4 z-50">
          <Button
            onClick={handleToggleCamera}
            size="icon"
            variant="secondary"
            className="w-12 h-12 rounded-full bg-white/95 dark:bg-gray-900/95 backdrop-blur-md shadow-xl hover:scale-110 transition-all"
          >
            <Video className="w-6 h-6" />
          </Button>
        </div>
      )}

      {/* Main content area */}
      <div className="relative z-10 w-full max-w-2xl px-8">
        {/* Status indicator */}
        <div className="text-center mb-8">
          <Card className="inline-flex items-center gap-3 bg-white/95 dark:bg-gray-900/95 backdrop-blur-md px-6 py-3 rounded-full shadow-xl border-0">
            <div
              className={`w-3 h-3 rounded-full ${stateDisplay.color} ${
                agentState === "listening" || agentState === "speaking"
                  ? "animate-pulse"
                  : ""
              }`}
            />
            <span className="font-semibold text-base">{stateDisplay.text}</span>
          </Card>
        </div>

        {/* Agent Visualizer */}
        <Card className="bg-white/95 dark:bg-gray-900/95 backdrop-blur-md shadow-2xl border-0 mb-8">
          <CardContent className="p-12 flex flex-col items-center justify-center gap-6">
            {agentAudioTrack && (
              <div
                style={{
                  width: "100%",
                  height: "150px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "8px",
                  padding: "16px 0",
                  borderRadius: "8px",
                  background: "transparent",
                }}
              >
                <BarVisualizer
                  trackRef={agentAudioTrack}
                  state={agentState}
                />
              </div>
            )}
            <p className="text-center text-gray-600 dark:text-gray-400 text-sm">
              {t("session.help")}
            </p>
          </CardContent>
        </Card>

        {/* Control buttons */}
        <div className="flex gap-4 justify-center">
          {/* Microphone toggle */}
          <Button
            onClick={handleToggleMic}
            size="icon"
            variant={isMuted ? "destructive" : "secondary"}
            className={`w-16 h-16 rounded-full shadow-xl transition-all hover:scale-110 ${
              isMuted
                ? "bg-red-500 hover:bg-red-600"
                : "bg-background hover:bg-background/90"
            }`}
          >
            {isMuted ? (
              <MicOff className="w-7 h-7" />
            ) : (
              <Mic className="w-7 h-7" />
            )}
          </Button>

          {/* End call button */}
          <Button
            onClick={onDisconnect}
            variant="destructive"
            className="h-16 px-8 rounded-full shadow-xl text-base font-bold bg-red-500 hover:bg-red-600 transition-all hover:scale-105 gap-3"
          >
            <Phone className="w-6 h-6" />
            {t("session.endCall")}
          </Button>
        </div>
      </div>
    </div>
  );
}