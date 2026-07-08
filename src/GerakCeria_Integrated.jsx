import { useState, useEffect, useRef, useCallback } from "react";

// ─── MEDIAPIPE + MOVENET LOADER ───────────────────────────────────────────────
const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

let _poseInitPromise = null;
async function loadMediaPipe() {
  if (_poseInitPromise) return _poseInitPromise;
  _poseInitPromise = (async () => {
    if (!window.FilesetResolver) {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.type = "module";
        s.textContent = `
          import { FilesetResolver, PoseLandmarker }
            from "${MEDIAPIPE_CDN}/vision_bundle.mjs";
          window.FilesetResolver = FilesetResolver;
          window.PoseLandmarker  = PoseLandmarker;
          window.dispatchEvent(new Event('mediapipe_loaded'));
        `;
        document.head.appendChild(s);
        window.addEventListener("mediapipe_loaded", resolve, { once: true });
        setTimeout(reject, 15000);
      });
    }
    const vision = await window.FilesetResolver.forVisionTasks(
      `${MEDIAPIPE_CDN}/wasm`
    );
    const landmarker = await window.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 2,
    });
    return landmarker;
  })();
  return _poseInitPromise;
}

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[24,26],
  [25,27],[26,28],[27,29],[28,30],[29,31],[30,32],
  [15,17],[15,19],[15,21],[16,18],[16,20],[16,22],
];

class PoseDetectorEngine {
  constructor() { this.landmarker = null; this.ready = false; this._lastTs = -1; }
  async init() {
    if (this.ready) return;
    try {
      this.landmarker = await loadMediaPipe();
      this.ready = true;
    } catch(e) {
      console.warn("[PoseDetector] MediaPipe load failed:", e);
    }
  }
  detect(videoEl, timestamp) {
    if (!this.ready || !videoEl || videoEl.readyState < 2) return [];
    if (timestamp <= this._lastTs) return [];
    try {
      const result = this.landmarker.detectForVideo(videoEl, timestamp);
      this._lastTs = timestamp;
      return result.landmarks || [];
    } catch(e) { return []; }
  }
  drawSkeleton(ctx, canvas, landmarks, color = "#22c55e", lineWidth = 3) {
    if (!landmarks || landmarks.length === 0) return;
    const W = canvas.width, H = canvas.height;
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineCap = "round";
    for (const [a, b] of POSE_CONNECTIONS) {
      const la = landmarks[a], lb = landmarks[b];
      if (!la || !lb) continue;
      if ((la.visibility ?? 1) < 0.3 || (lb.visibility ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(la.x * W, la.y * H);
      ctx.lineTo(lb.x * W, lb.y * H);
      ctx.stroke();
    }
    ctx.fillStyle = "#fff";
    for (const lm of landmarks) {
      if ((lm.visibility ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  extractAngles(lm) {
    if (!lm || lm.length < 33) return null;
    const angle = (a, b, c) => {
      const ba = { x: a.x - b.x, y: a.y - b.y };
      const bc = { x: c.x - b.x, y: c.y - b.y };
      const dot = ba.x * bc.x + ba.y * bc.y;
      const mag = Math.sqrt(ba.x**2+ba.y**2) * Math.sqrt(bc.x**2+bc.y**2);
      if (mag === 0) return 0;
      return (Math.acos(Math.max(-1, Math.min(1, dot/mag))) * 180) / Math.PI;
    };
    return {
      leftElbow:    angle(lm[11], lm[13], lm[15]),
      rightElbow:   angle(lm[12], lm[14], lm[16]),
      leftShoulder: angle(lm[13], lm[11], lm[23]),
      rightShoulder:angle(lm[14], lm[12], lm[24]),
      leftHip:      angle(lm[11], lm[23], lm[25]),
      rightHip:     angle(lm[12], lm[24], lm[26]),
      leftKnee:     angle(lm[23], lm[25], lm[27]),
      rightKnee:    angle(lm[24], lm[26], lm[28]),
      leftWristY:   lm[15]?.y ?? 1,
      rightWristY:  lm[16]?.y ?? 1,
      leftShoulderY:lm[11]?.y ?? 0.5,
      rightShoulderY:lm[12]?.y ?? 0.5,
      noseY:        lm[0]?.y ?? 0.5,
      leftHipY:     lm[23]?.y ?? 0.7,
      rightHipY:    lm[24]?.y ?? 0.7,
    };
  }
}
const PoseDetector = new PoseDetectorEngine();

const GestureDetector = (() => {
  const HISTORY_SIZE = 8;
  const histories = {};
  const push = (id, val) => {
    if (!histories[id]) histories[id] = [];
    histories[id].push(val);
    if (histories[id].length > HISTORY_SIZE) histories[id].shift();
  };
  const avg = (id) => {
    const h = histories[id] || [];
    return h.length ? h.reduce((a,b)=>a+b,0)/h.length : 0;
  };
  const detect = (angles, personId = 0) => {
    if (!angles) return {};
    const pid = `p${personId}`;
    const leftRaised  = angles.leftWristY  < angles.leftShoulderY  - 0.05;
    const rightRaised = angles.rightWristY < angles.rightShoulderY - 0.05;
    push(`${pid}_raised`, (leftRaised && rightRaised) ? 1 : 0);
    push(`${pid}_leftHand`, leftRaised ? 1 : 0);
    push(`${pid}_rightHand`, rightRaised ? 1 : 0);
    const kneeAvg = (angles.leftKnee + angles.rightKnee) / 2;
    const hipAvg  = (angles.leftHip  + angles.rightHip)  / 2;
    push(`${pid}_squat`, (kneeAvg < 120 && hipAvg < 120) ? 1 : 0);
    const noseY = angles.noseY;
    push(`${pid}_noseY`, noseY);
    const noseHist = histories[`${pid}_noseY`] || [];
    let jumping = false;
    if (noseHist.length >= 4) {
      const diff = noseHist[0] - noseHist[noseHist.length - 1];
      jumping = diff > 0.04;
    }
    const leftKneeLift  = angles.leftKnee  < 140;
    const rightKneeLift = angles.rightKnee < 140;
    push(`${pid}_runL`, leftKneeLift  ? 1 : 0);
    push(`${pid}_runR`, rightKneeLift ? 1 : 0);
    const runActivity = (avg(`${pid}_runL`) + avg(`${pid}_runR`)) / 2;
    return {
      raisedHands: avg(`${pid}_raised`) > 0.6,
      handLeft:    avg(`${pid}_leftHand`) > 0.6,
      handRight:   avg(`${pid}_rightHand`) > 0.6,
      squat:       avg(`${pid}_squat`) > 0.5,
      jumping,
      running:     runActivity > 0.3,
    };
  };
  const checkMove = (moveName, angles, personId = 0) => {
    const g = detect(angles, personId);
    switch (moveName) {
      case "raise_both":  return g.raisedHands;
      case "raise_right": return g.handRight;
      case "raise_left":  return g.handLeft;
      case "squat":       return g.squat;
      case "jump":        return g.jumping;
      case "run":         return g.running;
      default: return false;
    }
  };
  const resetAll = () => Object.keys(histories).forEach(k => delete histories[k]);
  return { detect, checkMove, resetAll };
})();

const EmotionDetector = (() => {
  const MSGS = {
    happy:   ["Semangat terus! 🎉","Kamu hebat! 🌟","Luar biasa! ⭐"],
    tired:   ["Istirahat sebentar ya!","Tetap semangat! 💪","Kamu bisa!"],
    great:   ["Gerakan sempurna! 🏆","Mantap sekali! 🔥","Terus begitu!"],
    neutral: ["Ayo bergerak! 🏃","Fokus ya!","Kita bisa!"],
  };
  const detect = (angles) => {
    if (!angles) return "neutral";
    const bothUp = angles.leftWristY < angles.leftShoulderY - 0.1 &&
                   angles.rightWristY < angles.rightShoulderY - 0.1;
    if (bothUp) return "happy";
    const deepSquat = (angles.leftKnee + angles.rightKnee) / 2 < 100;
    if (deepSquat) return "great";
    const shoulderAvg = (angles.leftShoulder + angles.rightShoulder) / 2;
    if (shoulderAvg < 20) return "tired";
    return "neutral";
  };
  const getMessage = (emotion) => {
    const msgs = MSGS[emotion] || MSGS.neutral;
    return msgs[Math.floor(Math.random() * msgs.length)];
  };
  return { detect, getMessage };
})();

function useCameraTracking() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const streamRef   = useRef(null);
  const rafRef      = useRef(null);
  const lastGestRef = useRef(false);
  const moveTargetRef  = useRef(null);
  const onDetectRef    = useRef(null);
  const isRunningRef   = useRef(false);
  const hiddenVideoRef = useRef(null);

  const [cameraReady,  setCameraReady]  = useState(false);
  const [cameraError,  setCameraError]  = useState(null);
  const [poseReady,    setPoseReady]    = useState(false);
  const [lastAngles,   setLastAngles]   = useState(null);
  const [lastGesture,  setLastGesture]  = useState({});
  const [poseDetected, setPoseDetected] = useState(false);

  useEffect(() => {
    const vid = document.createElement("video");
    vid.autoplay = true;
    vid.playsInline = true;
    vid.muted = true;
    vid.style.cssText = "position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;top:-9999px;left:-9999px;";
    document.body.appendChild(vid);
    hiddenVideoRef.current = vid;
    videoRef.current = vid;
    return () => {
      document.body.removeChild(vid);
      hiddenVideoRef.current = null;
    };
  }, []);

  const runLoop = useCallback(() => {
    if (isRunningRef.current) return;
    isRunningRef.current = true;
    const loop = () => {
      if (!isRunningRef.current) return;
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !PoseDetector.ready || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      canvas.width  = video.videoWidth  || video.clientWidth  || 640;
      canvas.height = video.videoHeight || video.clientHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const ts = performance.now();
      const allLandmarks = PoseDetector.detect(video, ts);
      const lm = allLandmarks[0] ?? null;
      if (lm) {
        setPoseDetected(true);
        PoseDetector.drawSkeleton(ctx, canvas, lm);
        const angles  = PoseDetector.extractAngles(lm);
        setLastAngles(angles);
        setLastGesture(GestureDetector.detect(angles, 0));
        if (moveTargetRef.current && angles) {
          const detected = GestureDetector.checkMove(moveTargetRef.current, angles, 0);
          if (detected && !lastGestRef.current) {
            onDetectRef.current && onDetectRef.current(angles);
          }
          lastGestRef.current = detected;
        }
      } else {
        setPoseDetected(false);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
    } catch (e) {
      setCameraError(e.message || "Kamera tidak dapat diakses");
      return;
    }
    streamRef.current = stream;
    const vid = videoRef.current;
    vid.srcObject = stream;
    try { await vid.play(); } catch(e) { console.warn("[Camera] play() failed:", e); }
    setCameraReady(true);
    try {
      await PoseDetector.init();
      setPoseReady(true);
    } catch (e) {
      console.warn("[Camera] PoseDetector init failed:", e);
    }
    isRunningRef.current = false;
    runLoop();
  }, [runLoop]);

  const startLoop = useCallback((moveTarget, onDetect) => {
    moveTargetRef.current = moveTarget;
    onDetectRef.current   = onDetect;
    lastGestRef.current   = false;
    if (!rafRef.current) runLoop();
  }, [runLoop]);

  const stopCamera = useCallback(() => {
    isRunningRef.current = false;
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    moveTargetRef.current = null;
    onDetectRef.current   = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) { videoRef.current.srcObject = null; }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setCameraReady(false);
    setPoseDetected(false);
    setPoseReady(false);
    GestureDetector.resetAll();
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  return {
    videoRef, canvasRef,
    cameraReady, cameraError, poseReady,
    startCamera, stopCamera, startLoop,
    lastAngles, lastGesture, poseDetected,
  };
}

function CameraView({ videoRef, canvasRef, cameraReady, cameraError, poseDetected, poseReady, height = 220, style = {} }) {
  const displayVideoRef = useRef(null);

  useEffect(() => {
    const displayVid = displayVideoRef.current;
    const sourceVid  = videoRef?.current;
    if (!displayVid || !sourceVid || !cameraReady) return;
    if (displayVid.srcObject !== sourceVid.srcObject) {
      displayVid.srcObject = sourceVid.srcObject;
      displayVid.play().catch(() => {});
    }
  }, [cameraReady, videoRef]);

  return (
    <div style={{
      position: "relative", width: "100%", height,
      background: "rgba(0,0,0,0.6)", borderRadius: 18,
      overflow: "hidden", border: "1px solid rgba(255,255,255,0.1)",
      ...style,
    }}>
      <video
        ref={displayVideoRef}
        autoPlay playsInline muted
        style={{
          position: "absolute", inset: 0,
          width: "100%", height: "100%",
          objectFit: "cover",
          transform: "scaleX(-1)",
          opacity: cameraReady ? 1 : 0,
          transition: "opacity 0.4s",
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute", inset: 0,
          width: "100%", height: "100%",
          transform: "scaleX(-1)",
          pointerEvents: "none",
        }}
      />
      {cameraReady && (
        <div style={{
          position: "absolute", top: 10, left: 10,
          display: "flex", gap: 6, flexDirection: "column",
        }}>
          <div style={{
            background: "rgba(0,0,0,0.6)", borderRadius: 50,
            padding: "3px 10px", fontSize: 10, fontWeight: 800,
            color: poseDetected ? "#22c55e" : "#94a3b8",
            display: "flex", alignItems: "center", gap: 5,
            backdropFilter: "blur(8px)",
          }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: poseDetected ? "#22c55e" : "#64748b",
              animation: poseDetected ? "pulse 1s ease-in-out infinite" : "none",
            }} />
            {poseDetected ? "POSE TERDETEKSI" : poseReady ? "Tunggu gerakan…" : "Loading AI…"}
          </div>
          {poseDetected && (
            <div style={{
              background: "rgba(124,58,237,0.7)", borderRadius: 50,
              padding: "3px 10px", fontSize: 10, fontWeight: 800, color: "#fff",
              backdropFilter: "blur(8px)",
            }}>
              📡 MediaPipe Active
            </div>
          )}
        </div>
      )}
      {!cameraReady && (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: 8,
        }}>
          <div style={{ fontSize: 36 }}>📷</div>
          <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700, textAlign: "center", padding: "0 16px" }}>
            {cameraError ? `❌ ${cameraError}` : "Kamera belum aktif"}
          </div>
        </div>
      )}
    </div>
  );
}

const MascotSVG = ({ size = 120, mood = "happy", glow = false }) => (
  <svg width={size} height={size} viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg"
    style={{ filter: glow ? "drop-shadow(0 0 16px #a78bfa) drop-shadow(0 0 32px #7c3aed)" : "drop-shadow(0 4px 12px rgba(124,58,237,0.4))" }}>
    <ellipse cx="60" cy="80" rx="28" ry="32" fill="url(#bodyGrad)" />
    <circle cx="60" cy="45" r="30" fill="url(#headGrad)" />
    <rect x="34" y="33" width="52" height="26" rx="10" fill="url(#visorGrad)" opacity="0.9" />
    <ellipse cx="48" cy="46" rx="7" ry="8" fill="#fff" />
    <ellipse cx="72" cy="46" rx="7" ry="8" fill="#fff" />
    <circle cx={mood === "happy" ? 50 : 48} cy="46" r="4" fill="#1e1b4b" />
    <circle cx={mood === "happy" ? 74 : 72} cy="46" r="4" fill="#1e1b4b" />
    <circle cx={mood === "happy" ? 51 : 49} cy="44" r="1.5" fill="#fff" />
    <circle cx={mood === "happy" ? 75 : 73} cy="44" r="1.5" fill="#fff" />
    {mood === "happy"
      ? <path d="M50 58 Q60 66 70 58" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" fill="none" />
      : <path d="M50 60 Q60 55 70 60" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" fill="none" />}
    <line x1="60" y1="15" x2="60" y2="0" stroke="#c4b5fd" strokeWidth="3" strokeLinecap="round" />
    <circle cx="60" cy="0" r="5" fill="#a78bfa">
      <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
    </circle>
    <rect x="18" y="70" width="12" height="28" rx="6" fill="url(#bodyGrad)" transform="rotate(-15 18 70)" />
    <rect x="90" y="70" width="12" height="28" rx="6" fill="url(#bodyGrad)" transform="rotate(15 90 70)" />
    <rect x="44" y="106" width="12" height="14" rx="6" fill="#6d28d9" />
    <rect x="64" y="106" width="12" height="14" rx="6" fill="#6d28d9" />
    <ellipse cx="50" cy="119" rx="10" ry="5" fill="#4c1d95" />
    <ellipse cx="70" cy="119" rx="10" ry="5" fill="#4c1d95" />
    <text x="96" y="28" fontSize="12" fill="#fbbf24">✦</text>
    <text x="10" y="35" fontSize="10" fill="#34d399">✦</text>
    <defs>
      <radialGradient id="headGrad" cx="40%" cy="35%">
        <stop stopColor="#c4b5fd" /><stop offset="1" stopColor="#7c3aed" />
      </radialGradient>
      <radialGradient id="bodyGrad" cx="40%" cy="30%">
        <stop stopColor="#a78bfa" /><stop offset="1" stopColor="#5b21b6" />
      </radialGradient>
      <linearGradient id="visorGrad" x1="0" y1="0" x2="0" y2="1">
        <stop stopColor="#e0f2fe" stopOpacity="0.9" /><stop offset="1" stopColor="#bae6fd" stopOpacity="0.6" />
      </linearGradient>
    </defs>
  </svg>
);

const StarIcon = ({ filled, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={filled ? "#fbbf24" : "none"} stroke={filled ? "#fbbf24" : "#94a3b8"} strokeWidth="2">
    <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" />
  </svg>
);

const Confetti = ({ active }) => {
  const pieces = useRef([...Array(32)].map((_, i) => ({
    id: i,
    x: Math.random() * 100,
    color: ["#fbbf24","#f472b6","#34d399","#60a5fa","#c084fc","#fb923c"][i % 6],
    delay: Math.random() * 0.8,
    size: 6 + Math.random() * 8,
    rotate: Math.random() * 360,
  })));
  if (!active) return null;
  return (
    <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:999, overflow:"hidden" }}>
      {pieces.current.map(p => (
        <div key={p.id} style={{
          position:"absolute", left:`${p.x}%`, top:"-20px",
          width:p.size, height:p.size, background:p.color,
          borderRadius: p.id % 3 === 0 ? "50%" : "2px",
          transform:`rotate(${p.rotate}deg)`,
          animation:`confettiFall 2.5s ${p.delay}s ease-in forwards`,
        }} />
      ))}
      <style>{`
        @keyframes confettiFall {
          0% { top:-20px; opacity:1; transform:rotate(0deg) translateX(0); }
          100% { top:110vh; opacity:0; transform:rotate(720deg) translateX(${Math.random() > 0.5 ? 80 : -80}px); }
        }
      `}</style>
    </div>
  );
};

const FloatingParticles = () => {
  const particles = useRef([...Array(18)].map((_, i) => ({
    id: i,
    x: Math.random() * 100,
    size: 4 + Math.random() * 8,
    dur: 8 + Math.random() * 12,
    delay: Math.random() * 8,
    color: ["rgba(167,139,250,0.4)","rgba(52,211,153,0.3)","rgba(251,191,36,0.3)","rgba(96,165,250,0.3)"][i % 4],
  })));
  return (
    <div style={{ position:"absolute", inset:0, overflow:"hidden", pointerEvents:"none" }}>
      {particles.current.map(p => (
        <div key={p.id} style={{
          position:"absolute",
          left:`${p.x}%`, bottom:"-20px",
          width:p.size, height:p.size,
          borderRadius:"50%",
          background:p.color,
          animation:`floatUp ${p.dur}s ${p.delay}s ease-in-out infinite`,
        }} />
      ))}
    </div>
  );
};

const KID_BOY_IMG = "data:image/webp;base64,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";
const KID_GIRL_IMG = "data:image/webp;base64,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";

const KidBoySVG = ({ size = 200 }) => (
  <div style={{
    width: size,
    height: `calc(${typeof size === "number" ? `${size}px` : size} * 1.85)`,
    display: "flex", alignItems: "flex-end", justifyContent: "center",
  }}>
    <img
      src={KID_BOY_IMG}
      alt="Karakter anak laki-laki"
      draggable={false}
      style={{
        maxWidth: "80%", maxHeight: "80%", width: "auto", height: "auto",
        objectFit: "contain", display: "block",
        filter: "drop-shadow(0 12px 24px rgba(0,0,0,0.35))",
        userSelect: "none",
      }}
    />
  </div>
);

const KidGirlSVG = ({ size = 200 }) => (
  <div style={{
    width: size,
    height: `calc(${typeof size === "number" ? `${size}px` : size} * 1.85)`,
    display: "flex", alignItems: "flex-end", justifyContent: "center",
  }}>
    <img
      src={KID_GIRL_IMG}
      alt="Karakter anak perempuan"
      draggable={false}
      style={{
        maxWidth: "100%", maxHeight: "100%", width: "auto", height: "auto",
        objectFit: "contain", display: "block",
        filter: "drop-shadow(0 12px 24px rgba(0,0,0,0.35))",
        userSelect: "none",
      }}
    />
  </div>
);

const WelcomeScreen = ({ onStart }) => {
  const [pressed, setPressed] = useState(false);
  const [musicOn, setMusicOn] = useState(true);
  const [hoverSettings, setHoverSettings] = useState(false);
  const [hoverMusic, setHoverMusic] = useState(false);
  const [hoverPlay, setHoverPlay] = useState(false);

  return (
    <div style={{
      minHeight:"100vh", width:"100%", position:"relative", overflow:"hidden",
      background:"radial-gradient(ellipse at 50% 0%, #2a1262 0%, #1a0533 45%, #0d1224 100%)",
      display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center",
      padding:"32px 24px",
    }}>
      <FloatingParticles />
      <div style={{ position:"absolute", top:"8%", left:"6%", fontSize:28, opacity:0.5, animation:"pulse 3s ease-in-out infinite" }}>✦</div>
      <div style={{ position:"absolute", top:"22%", right:"10%", fontSize:18, opacity:0.4, animation:"pulse 4s ease-in-out infinite" }}>✦</div>
      <div style={{ position:"absolute", bottom:"30%", left:"4%", fontSize:16, opacity:0.35, animation:"pulse 3.5s ease-in-out infinite" }}>✦</div>
      <div style={{
        position:"absolute", top:"6%", right:"4%", width:90, height:90, borderRadius:"50%",
        background:"linear-gradient(135deg,#7c3aed,#4338ca)", opacity:0.18, filter:"blur(2px)",
      }} />
      <div style={{
        position:"absolute", top:20, left:20, right:20,
        display:"flex", justifyContent:"space-between", alignItems:"center", zIndex:5,
      }}>
        <button
          onMouseEnter={() => setHoverMusic(true)} onMouseLeave={() => setHoverMusic(false)}
          onClick={() => { AudioSystem.click(); setMusicOn(m => !m); }}
          style={{
            display:"flex", alignItems:"center", gap:8, padding:"10px 18px", borderRadius:999,
            background: hoverMusic ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.06)",
            border:"1px solid rgba(255,255,255,0.15)", color:"#fff", fontWeight:800, fontSize:13,
            cursor:"pointer", backdropFilter:"blur(12px)", transition:"all 0.2s ease",
          }}>
          <span style={{ fontSize:15 }}>🎵</span> Musik
          <span style={{ color: musicOn ? "#34d399" : "#94a3b8" }}>{musicOn ? "● ON" : "○ OFF"}</span>
        </button>
        <button
          onMouseEnter={() => setHoverSettings(true)} onMouseLeave={() => setHoverSettings(false)}
          onClick={() => AudioSystem.click()}
          style={{
            display:"flex", alignItems:"center", gap:8, padding:"10px 18px", borderRadius:999,
            background: hoverSettings ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.06)",
            border:"1px solid rgba(255,255,255,0.15)", color:"#fff", fontWeight:800, fontSize:13,
            cursor:"pointer", backdropFilter:"blur(12px)", transition:"all 0.2s ease",
          }}>
          <span style={{ fontSize:15 }}>⚙️</span> Pengaturan
        </button>
      </div>
      <div style={{ animation:"mascotBob 3s ease-in-out infinite", marginBottom:-6, zIndex:3 }}>
        <MascotSVG size={110} mood="happy" glow />
      </div>
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"center", gap:8,
        width:"100%", maxWidth:1000, position:"relative", zIndex:2,
      }}>
        <div style={{ textAlign:"center", flex:1, minWidth:0 }}>
          <h1 style={{
            fontSize:"clamp(40px, 8vw, 76px)", fontWeight:900, lineHeight:1.02, margin:0,
            background:"linear-gradient(135deg,#ffffff 0%, #c4b5fd 60%, #93c5fd 100%)",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text",
            letterSpacing:-1, filter:"drop-shadow(0 0 24px rgba(167,139,250,0.4))",
          }}>GERAK</h1>
          <h1 style={{
            fontSize:"clamp(44px, 9vw, 84px)", fontWeight:900, lineHeight:1.0, margin:0,
            background:"linear-gradient(135deg,#fbbf24 0%, #fb923c 60%, #f59e0b 100%)",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text",
            letterSpacing:-1, filter:"drop-shadow(0 0 28px rgba(251,191,36,0.45))",
          }}>CERIA</h1>
          <div style={{
            display:"inline-block", marginTop:10, padding:"7px 26px", borderRadius:999,
            background:"linear-gradient(135deg,#7c3aed,#4338ca)",
            boxShadow:"0 6px 24px rgba(124,58,237,0.5)",
          }}>
            <span style={{ color:"#fff", fontWeight:900, fontSize:"clamp(13px,2.2vw,18px)", letterSpacing:1.5 }}>AI ADVENTURE</span>
          </div>
        </div>
      </div>
      <div className="welcome-kid welcome-kid-boy" style={{
        position:"absolute", left:"clamp(-10px, 1vw, 24px)", top:"17%", zIndex:2,
        animation:"mascotBob 4s ease-in-out infinite",
      }}>
        <KidBoySVG size="clamp(120px, 19vw, 300px)" />
      </div>
      <div className="welcome-kid welcome-kid-girl" style={{
        position:"absolute", right:"clamp(-10px, 1vw, 24px)", top:"20%", zIndex:2,
        animation:"mascotBob 4.5s ease-in-out infinite 0.3s",
      }}>
        <KidGirlSVG size="clamp(120px, 19vw, 300px)" />
      </div>
      <p style={{
        color:"#e2e8f0", fontSize:"clamp(14px,2.4vw,19px)", fontWeight:700, margin:"18px 0 28px",
        textAlign:"center", maxWidth:560,
      }}>
        <span style={{ color:"#fbbf24" }}>✨</span> Ayo bergerak, ayo sehat, ayo <span style={{ color:"#fbbf24" }}>ceria!</span> <span>🤸</span><span style={{ color:"#fbbf24" }}>✨</span>
      </p>
      <button
        onMouseEnter={() => setHoverPlay(true)}
        onMouseLeave={() => setHoverPlay(false)}
        onMouseDown={() => setPressed(true)}
        onMouseUp={() => setPressed(false)}
        onClick={onStart}
        style={{
          display:"flex", alignItems:"center", justifyContent:"center", gap:14,
          padding:"20px 56px", borderRadius:20, border:"none", cursor:"pointer",
          background:"linear-gradient(135deg,#7c3aed 0%, #ec4899 100%)",
          color:"#fff", fontWeight:900, fontSize:"clamp(17px,2.6vw,24px)", letterSpacing:1,
          boxShadow: hoverPlay
            ? "0 12px 48px rgba(124,58,237,0.65), 0 0 0 6px rgba(236,72,153,0.15)"
            : "0 8px 32px rgba(124,58,237,0.5)",
          transform: pressed ? "scale(0.96)" : hoverPlay ? "scale(1.03) translateY(-2px)" : "scale(1)",
          transition:"all 0.2s cubic-bezier(0.34,1.56,0.64,1)",
          animation:"playPulse 2.5s ease-in-out infinite",
        }}>
        <span style={{
          width:0, height:0,
          borderTop:"11px solid transparent", borderBottom:"11px solid transparent",
          borderLeft:"18px solid #fff",
        }} />
        MULAI BERMAIN
      </button>
      <div style={{
        display:"grid", gridTemplateColumns:"repeat(4, minmax(90px,1fr))", gap:14,
        width:"100%", maxWidth:680, marginTop:40, zIndex:2,
      }}>
        {[
          { icon:"🏆", label:"Badge Saya"},
          { icon:"🎯", label:"Misi" },
          { icon:"⚔️", label:"Battle" },
          { icon:"📊", label:"Statistik" },
        ].map((c, i) => (
          <div key={i} className="welcome-quick-card" style={{
            display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:8,
            padding:"18px 8px", borderRadius:18,
            background:"rgba(255,255,255,0.05)", border:"1px solid rgba(167,139,250,0.25)",
            backdropFilter:"blur(12px)", cursor:"pointer", transition:"all 0.2s ease",
          }}>
            <span style={{ fontSize:26 }}>{c.icon}</span>
            <span style={{ color:"#e2e8f0", fontWeight:800, fontSize:12, textAlign:"center" }}>{c.label}</span>
          </div>
        ))}
      </div>
      <div style={{
        position:"absolute", bottom:16, left:24, right:24,
        display:"flex", justifyContent:"space-between", color:"#64748b", fontSize:12, fontWeight:700,
      }}>
        <span>Versi 1.0.0</span>
        <span>Gerak Ceria AI Adventure © 2026</span>
      </div>
      <style>{`
        .welcome-quick-card:hover { transform: translateY(-4px) scale(1.04); background: rgba(124,58,237,0.18) !important; border-color: rgba(167,139,250,0.5) !important; }
        @media (max-width: 640px) {
          .welcome-kid-boy { left: -18px !important; top: 20% !important; }
          .welcome-kid-girl { right: -18px !important; top: 22% !important; }
          .welcome-kid img { width: 110px !important; }
        }
      `}</style>
    </div>
  );
};

const AudioSystem = (() => {
  let ctx = null;
  let soundOn = true;
  function getCtx() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }
  function beep(freq, duration, type = "sine", vol = 0.25) {
    if (!soundOn) return;
    try {
      const ac = getCtx();
      const osc = ac.createOscillator();
      const gain = ac.createGain();
      osc.connect(gain); gain.connect(ac.destination);
      osc.type = type;
      osc.frequency.setValueAtTime(freq, ac.currentTime);
      gain.gain.setValueAtTime(vol, ac.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + duration);
      osc.start(ac.currentTime); osc.stop(ac.currentTime + duration);
    } catch(e) {}
  }
  function speak(text, rate = 0.95, pitch = 1.1) {
    if (!soundOn || !window.speechSynthesis) return;
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = "id-ID"; utt.rate = rate; utt.pitch = pitch; utt.volume = 1.0;
    speechSynthesis.cancel();
    speechSynthesis.speak(utt);
  }
  return {
    click:    () => beep(800, 0.08, "square", 0.15),
    success:  () => { beep(523, 0.15); setTimeout(() => beep(659, 0.15), 100); setTimeout(() => beep(784, 0.25), 200); },
    fail:     () => { beep(300, 0.2, "sawtooth"); setTimeout(() => beep(220, 0.3, "sawtooth"), 150); },
    rep:      () => beep(660, 0.1, "sine", 0.2),
    combo:    () => { beep(880, 0.08); setTimeout(() => beep(1100, 0.12), 80); },
    levelup:  () => { [523,587,659,698,784].forEach((f, i) => setTimeout(() => beep(f, 0.15), i * 80)); },
    countdown:(n) => {
      if (n === 0) { [523,659,784,1047].forEach((f,i) => setTimeout(() => beep(f,0.2,"sine",0.22), i*60)); }
      else { beep(n === 1 ? 660 : 440, 0.18, "sine", 0.25); }
    },
    win:      () => { [523,659,784,1047].forEach((f,i) => setTimeout(() => beep(f,0.2), i*120)); },
    badge:    () => { [523,659,784].forEach(f=>beep(f,0.2)); setTimeout(()=>[659,784,1047].forEach(f=>beep(f,0.35)),200); },
    energy:   () => beep(440, 0.1, "sine", 0.15),
    moveDone: () => { [523,659,784].forEach(f=>beep(f,0.18,"sine",0.2)); setTimeout(()=>[659,784,1047].forEach(f=>beep(f,0.28,"sine",0.2)),200); },
    speak,
    voiceStart:    () => speak("Bersiap untuk pemanasan! Ikuti gerakan di layar kiri."),
    voiceMove:     (name, instr) => speak(`${name}! ${instr}`, 0.92),
    voiceRep:      (n, max) => { if (n === max) speak("Sempurna! Gerakan selesai!", 1.05); else if (n % 2 === 0) speak(`Bagus! ${n}`, 1.1); },
    voiceCountdown:(n) => speak(n === 0 ? "Mulai! Ayo bergerak!" : String(n), 0.9, 1.2),
    voiceDone:     () => speak("Selamat! Pemanasan selesai! Kerja yang luar biasa!", 0.9),
    voiceStop:     () => { if (window.speechSynthesis) speechSynthesis.cancel(); speak("Sesi dihentikan."); },
    toggleSound:   () => { soundOn = !soundOn; if (!soundOn && window.speechSynthesis) speechSynthesis.cancel(); return soundOn; },
    isOn:          () => soundOn,
  };
})();

// ─── GAME STATE ────────────────────────────────────────────────────────────────
const DEFAULT_STATE = {
  playerName: "Atlet Cilik",
  level: 0,
  exp: 0,
  energy: 100,
  totalStars: 0,
  totalMoves: 0,
  streak: 0,
  lastPlayDate: null,
  levelStars: {},
  levelProgress: {},
  levelUnlocked: { 1: true },
  badges: {},
  activityLog: [],
  weeklyActivity: [0, 0, 0, 0, 0, 0, 0],
  totalCalories: 0,
  warmupCount: 0,
  challengeHighScore: 0,
  miniGameHighScores: {},
  redTeamWins: 0,
  blueTeamWins: 0,
  settings: { sound: true, music: true, tracking: true },
  // ── Tracking untuk Daily Challenges & Weekly Boss ──────────────────────────
  // Semua counter harian direset setiap hari baru (lastDailyReset)
  // weeklyMoves direset setiap Senin (lastWeeklyReset)
  lastDailyReset: null,   // toDateString() hari terakhir reset harian
  lastWeeklyReset: null,  // toDateString() Senin terakhir reset mingguan
  dailyJumps: 0,          // Lompatan hari ini (target: 10)
  dailySquats: 0,         // Squat hari ini (target: 15)
  weeklyMoves: 0,         // Total gerakan minggu ini (target: 100)
  // Klaim challenge harian — direset tiap hari
  dailyClaimed: {},       // { 1: true, 2: true, 3: true }
};

function useGameState() {
  const [state, setStateRaw] = useState(() => {
    try {
      const saved = localStorage.getItem("gerakCeriaState");
      if (saved) return { ...DEFAULT_STATE, ...JSON.parse(saved) };
    } catch(e) {}
    return { ...DEFAULT_STATE };
  });

  const setState = useCallback((updater) => {
    setStateRaw(prev => {
      const next = typeof updater === "function" ? updater(prev) : { ...prev, ...updater };
      try { localStorage.setItem("gerakCeriaState", JSON.stringify(next)); } catch(e) {}
      return next;
    });
  }, []);

  const addXP = useCallback((amount) => {
    setState(prev => {
      let newExp = prev.exp + amount;
      let newLevel = prev.level;
      while (newExp >= 2000) { newExp -= 2000; newLevel++; }
      return { ...prev, exp: newExp, level: newLevel };
    });
  }, [setState]);

  // ── Helper: cek apakah perlu reset harian / mingguan ──────────────────────
  const _checkResets = (prev) => {
    const todayStr = new Date().toDateString();
    // Cari Senin minggu ini
    const now = new Date();
    const day = now.getDay(); // 0=Min, 1=Sen,...
    const diffToMonday = (day === 0 ? -6 : 1 - day);
    const monday = new Date(now);
    monday.setDate(now.getDate() + diffToMonday);
    monday.setHours(0,0,0,0);
    const mondayStr = monday.toDateString();

    let patch = {};
    // Reset harian
    if (prev.lastDailyReset !== todayStr) {
      patch = { ...patch, dailyJumps: 0, dailySquats: 0, lastDailyReset: todayStr, dailyClaimed: {} };
    }
    // Reset mingguan
    if (prev.lastWeeklyReset !== mondayStr) {
      patch = { ...patch, weeklyMoves: 0, lastWeeklyReset: mondayStr };
    }
    return patch;
  };

  const addActivity = useCallback((entry) => {
    setState(prev => {
      const resets = _checkResets(prev);
      const merged = { ...prev, ...resets };
      const today = new Date().toLocaleDateString("id-ID");
      const full = { ...entry, date: today, timestamp: Date.now() };
      const newLog = [full, ...merged.activityLog].slice(0, 20);
      const dayIdx = new Date().getDay();
      const newWeekly = [...merged.weeklyActivity];
      newWeekly[dayIdx] = (newWeekly[dayIdx] || 0) + 1;
      const isNewDay = merged.lastPlayDate !== new Date().toDateString();
      return {
        ...merged,
        activityLog: newLog,
        weeklyActivity: newWeekly,
        totalMoves: merged.totalMoves + 1,
        weeklyMoves: (merged.weeklyMoves || 0) + 1,
        totalCalories: merged.totalCalories + Math.floor(Math.random() * 8) + 3,
        streak: isNewDay ? merged.streak + 1 : merged.streak,
        lastPlayDate: isNewDay ? new Date().toDateString() : merged.lastPlayDate,
      };
    });
  }, [setState]);

  // ── Tambah lompatan harian ────────────────────────────────────────────────
  const addDailyJumps = useCallback((count = 1) => {
    setState(prev => {
      const resets = _checkResets(prev);
      const merged = { ...prev, ...resets };
      return { ...merged, dailyJumps: (merged.dailyJumps || 0) + count };
    });
  }, [setState]);

  // ── Tambah squat harian ───────────────────────────────────────────────────
  const addDailySquats = useCallback((count = 1) => {
    setState(prev => {
      const resets = _checkResets(prev);
      const merged = { ...prev, ...resets };
      return { ...merged, dailySquats: (merged.dailySquats || 0) + count };
    });
  }, [setState]);

  // ── Tandai challenge harian sudah diklaim ─────────────────────────────────
  const claimDailyChallenge = useCallback((challengeId, xpReward, addXPFn) => {
    setState(prev => {
      const resets = _checkResets(prev);
      const merged = { ...prev, ...resets };
      if (merged.dailyClaimed?.[challengeId]) return merged; // sudah diklaim hari ini
      const newClaimed = { ...(merged.dailyClaimed || {}), [challengeId]: true };
      if (addXPFn) addXPFn(xpReward);
      return { ...merged, dailyClaimed: newClaimed };
    });
  }, [setState]);

  // ── Sync reset harian saat state dibaca ──────────────────────────────────
  const syncDailyResets = useCallback(() => {
    setState(prev => {
      const resets = _checkResets(prev);
      if (Object.keys(resets).length === 0) return prev;
      return { ...prev, ...resets };
    });
  }, [setState]);

  const unlockBadge = useCallback((badgeId) => {
    setState(prev => {
      if (prev.badges[badgeId]?.unlocked) return prev;
      return { ...prev, badges: { ...prev.badges, [badgeId]: { unlocked: true, date: new Date().toISOString() } } };
    });
    return true;
  }, [setState]);

  const setLevelStars = useCallback((levelId, stars) => {
    setState(prev => {
      const prevStars = prev.levelStars[levelId] || 0;
      const newStars = Math.max(prevStars, stars);
      const newLevelStars = { ...prev.levelStars, [levelId]: newStars };
      const newUnlocked = { ...prev.levelUnlocked };
      if (stars >= 1 && levelId < 5) newUnlocked[levelId + 1] = true;
      const total = Object.values(newLevelStars).reduce((a, b) => a + b, 0);
      return { ...prev, levelStars: newLevelStars, levelUnlocked: newUnlocked, totalStars: total };
    });
  }, [setState]);

  // ── BARU: simpan progres misi (persentase reps/target × 100) ──────────────
  // Hanya update jika nilai baru lebih tinggi dari yang tersimpan sebelumnya.
  const setLevelProgress = useCallback((levelId, progressPercent) => {
    setState(prev => {
      const prevProgress = prev.levelProgress?.[levelId] ?? 0;
      const newProgress = Math.max(prevProgress, Math.min(100, Math.round(progressPercent)));
      return {
        ...prev,
        levelProgress: {
          ...(prev.levelProgress || {}),
          [levelId]: newProgress,
        },
      };
    });
  }, [setState]);

  const resetState = useCallback(() => {
    try { localStorage.removeItem("gerakCeriaState"); } catch(e) {}
    setStateRaw({ ...DEFAULT_STATE });
  }, []);

  return { state, setState, addXP, addActivity, addDailyJumps, addDailySquats, claimDailyChallenge, syncDailyResets, unlockBadge, setLevelStars, setLevelProgress, resetState };
}

const BADGE_DEFS = [
  { id:"warmup_king",    icon:"🔥", name:"Raja Pemanasan",     desc:"Selesaikan AI Warm-Up pertamamu!",        trigger:"warmup_complete" },
  { id:"sporty_kid",     icon:"🏃", name:"Atlet Sportif",      desc:"Main 5 sesi game apapun.",                trigger:"play_5" },
  { id:"jump_master",    icon:"🏆", name:"Master Lompat",      desc:"Lakukan 20 lompatan dalam satu sesi.",    trigger:"jump_20" },
  { id:"team_player",    icon:"🤝", name:"Atlet Gotong Royong", desc:"Menangkan estafet tim.",                  trigger:"relay_win" },
  { id:"active_kid",     icon:"⚡", name:"Anak Aktif",         desc:"Bermain 3 hari berturut-turut.",          trigger:"streak_3" },
  { id:"star_collector", icon:"⭐", name:"Kolektor Bintang",   desc:"Kumpulkan 10 bintang dari mission.",      trigger:"stars_10" },
  { id:"challenge_hero", icon:"⚔️", name:"Hero Challenge",     desc:"Capai combo x5 di Motion Challenge.",     trigger:"combo_5" },
  { id:"perfect_mover",  icon:"💎", name:"Gerak Sempurna",     desc:"Raih skor 100% di salah satu mini game.", trigger:"perfect_score" },
];

const MISSIONS_DATA = [
  { id:1, name:"Latihan Dasar",   icon:"🌱", desc:"Gerakan dasar olahraga",  color:"#22c55e", glow:"rgba(34,197,94,0.4)",   moves:["raise_both","squat"], target:5 },
  { id:2, name:"Lompatan Ceria",  icon:"⬆️", desc:"Tantangan melompat seru", color:"#3b82f6", glow:"rgba(59,130,246,0.4)",  moves:["jump","raise_both"],  target:8 },
  { id:3, name:"Sprint Mini",     icon:"🏃", desc:"Lari kencang di tempat",  color:"#f97316", glow:"rgba(249,115,22,0.4)",  moves:["run","squat"],        target:10 },
  { id:4, name:"Combo Gerak",     icon:"⚡", desc:"Kombinasi gerakan ajaib", color:"#a855f7", glow:"rgba(168,85,247,0.4)",  moves:["raise_both","squat","jump"], target:8 },
  { id:5, name:"Master Olahraga", icon:"🏆", desc:"Tantangan akhir BOSS!",  color:"#ef4444", glow:"rgba(239,68,68,0.4)",   moves:["raise_both","squat","jump","run"], target:12 },
];

// ── DAILY_CHALLENGES dihitung dinamis dari state ──────────────────────────────
// Setiap challenge punya: target, getValue(state), xpReward
const DAILY_CHALLENGE_DEFS = [
  {
    id: 1,
    title: "Lompat 10x Hari Ini",
    icon: "⬆️",
    color: "#f472b6",
    xp: 30,
    target: 10,
    unit: "lompatan",
    getValue: (s) => Math.min(s.dailyJumps || 0, 10),
  },
  {
    id: 2,
    title: "Squat 15x Hari Ini",
    icon: "🦵",
    color: "#34d399",
    xp: 25,
    target: 15,
    unit: "squat",
    getValue: (s) => Math.min(s.dailySquats || 0, 15),
  },
  {
    id: 3,
    title: "Streak 3 Hari",
    icon: "🔥",
    color: "#fb923c",
    xp: 50,
    target: 3,
    unit: "hari",
    getValue: (s) => Math.min(s.streak || 0, 3),
  },
];

// Fungsi helper: hasilkan array challenge dengan progress nyata dari state
function buildDailyChallenges(state) {
  return DAILY_CHALLENGE_DEFS.map(def => {
    const current = def.getValue(state);
    const pct = Math.min(100, Math.round((current / def.target) * 100));
    const isDone = (state.dailyClaimed?.[def.id] === true) || pct >= 100;
    return {
      ...def,
      reward: `+${def.xp} XP`,
      progress: pct,
      current,
      done: isDone,
    };
  });
}

const WARMUP_MOVES = [
  { id:"raise_both", name:"Angkat Kedua Tangan", icon:"🙌", reps:5, instruction:"Angkat kedua tangan ke atas kepala!" },
  { id:"squat",      name:"Squat",               icon:"🦵", reps:5, instruction:"Tekuk lutut seperti mau duduk, lalu berdiri lagi!" },
  { id:"jump",       name:"Lompat Kecil",         icon:"⬆️", reps:5, instruction:"Lompat-lompat kecil di tempat!" },
  { id:"run",        name:"Lari di Tempat",       icon:"🏃", reps:8, instruction:"Angkat kaki bergantian seperti lari!" },
];

const COACH_MSGS = [
  "Ayo bergerak! Kamu bisa jadi juara! 💪",
  "Bagus sekali! Terus semangat! 🔥",
  "Gerakan kamu keren banget hari ini! ⚡",
  "Level up menanti! Kamu hampir sampai! 🌟",
  "Streak-mu makin panjang, luar biasa! 🏆",
];

const MOTION_CHALLENGES = [
  { text:"Angkat kedua tangan! 🙌", move:"raise_both", icon:"🙌" },
  { text:"Squat sekarang! 🦵",       move:"squat",      icon:"🦵" },
  { text:"Lompat 1x! ⬆️",            move:"jump",       icon:"⬆️" },
  { text:"Lari di tempat! 🏃",       move:"run",        icon:"🏃" },
  { text:"Angkat tangan kanan! ✋",  move:"raise_right",icon:"✋" },
  { text:"Angkat tangan kiri! 🤚",  move:"raise_left", icon:"🤚" },
];

export default function GerakCeria() {
  const [screen, setScreen] = useState("welcome");
  const [confetti, setConfetti] = useState(false);
  const [coachIdx, setCoachIdx] = useState(0);
  const [hoveredNav, setHoveredNav] = useState(null);
  const [pressedPlay, setPressedPlay] = useState(false);
  const [toast, setToast] = useState(null);
  const { state, setState, addXP, addActivity, addDailyJumps, addDailySquats, claimDailyChallenge, syncDailyResets, unlockBadge, setLevelStars, setLevelProgress, resetState } = useGameState();

  // Sync reset harian saat app dibuka
  useEffect(() => { syncDailyResets(); }, [syncDailyResets]);

  useEffect(() => {
    const t = setInterval(() => setCoachIdx(i => (i + 1) % COACH_MSGS.length), 4000);
    return () => clearInterval(t);
  }, []);

  const triggerConfetti = () => {
    setConfetti(true);
    setTimeout(() => setConfetti(false), 3000);
  };

  const showToast = useCallback((msg, duration = 3000) => {
    setToast(msg);
    setTimeout(() => setToast(null), duration);
  }, []);

  const tryAwardBadge = useCallback((trigger) => {
    const badge = BADGE_DEFS.find(b => b.trigger === trigger);
    if (!badge) return;
    if (!state.badges[badge.id]?.unlocked) {
      unlockBadge(badge.id);
      showToast(`🏅 Badge baru: ${badge.name} ${badge.icon}`, 3500);
      triggerConfetti();
      AudioSystem.badge();
    }
  }, [state.badges, unlockBadge, showToast]);

  const nav = [
    { id:"home",      icon: <HomeIcon />,     label:"Home" },
    { id:"mission",   icon: <MapIcon />,      label:"Misi" },
    { id:"challenge", icon: <ZapIcon />,      label:"Challenge" },
    { id:"warmup",    icon: <FireIcon />,     label:"Warmup" },
    { id:"minibattle",icon: <SwordIcon />,    label:"Battle" },
    { id:"badges",    icon: <TrophyIcon />,   label:"Badge" },
    { id:"stats",     icon: <ChartIcon />,    label:"Statistik" },
    { id:"settings",  icon: <SettingsIcon />, label:"Setting" },
  ];

  if (screen === "welcome") {
    return (
      <div style={{ fontFamily:"'Nunito', sans-serif", minHeight:"100vh", position:"relative", overflow:"hidden" }}>
        <style>{CSS}</style>
        <WelcomeScreen onStart={() => { AudioSystem.click(); setScreen("home"); }} />
      </div>
    );
  }

  return (
    <div style={{ fontFamily:"'Nunito', sans-serif", minHeight:"100vh", background:"linear-gradient(135deg, #0f0c29 0%, #1a0533 40%, #0d2137 100%)", display:"flex", position:"relative", overflow:"hidden" }}>
      <style>{CSS}</style>
      <Confetti active={confetti} />
      <FloatingParticles />
      {toast && (
        <div style={{
          position:"fixed", top:20, left:"50%", transform:"translateX(-50%)",
          background:"rgba(20,20,40,0.95)", border:"1px solid rgba(167,139,250,0.4)",
          borderRadius:14, padding:"12px 24px", color:"#fff", fontSize:14, fontWeight:800,
          zIndex:1000, backdropFilter:"blur(20px)", boxShadow:"0 8px 32px rgba(0,0,0,0.4)",
          animation:"coachFadeIn 0.3s ease",
        }}>{toast}</div>
      )}
      <aside style={{
        width: 88, minHeight:"100vh", background:"rgba(255,255,255,0.04)",
        backdropFilter:"blur(20px)", borderRight:"1px solid rgba(255,255,255,0.08)",
        display:"flex", flexDirection:"column", alignItems:"center",
        padding:"20px 0", gap:4, position:"relative", zIndex:10, flexShrink:0,
      }}>
        <div style={{ marginBottom:16, textAlign:"center" }}>
          <div style={{
            width:52, height:52, borderRadius:16, background:"linear-gradient(135deg,#7c3aed,#2563eb)",
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:26, boxShadow:"0 4px 20px rgba(124,58,237,0.5)",
            animation:"logoFloat 3s ease-in-out infinite",
          }}>🏃</div>
        </div>
        {nav.map(n => (
          <button key={n.id}
            onMouseEnter={() => setHoveredNav(n.id)}
            onMouseLeave={() => setHoveredNav(null)}
            onClick={() => { AudioSystem.click(); setScreen(n.id); }}
            style={{
              width:64, height:56, borderRadius:18, border:"none",
              background: screen === n.id
                ? "linear-gradient(135deg,#7c3aed,#2563eb)"
                : hoveredNav === n.id ? "rgba(124,58,237,0.2)" : "transparent",
              cursor:"pointer", display:"flex", flexDirection:"column",
              alignItems:"center", justifyContent:"center", gap:3,
              transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
              transform: screen === n.id ? "scale(1.08)" : hoveredNav === n.id ? "scale(1.04)" : "scale(1)",
              boxShadow: screen === n.id ? "0 4px 20px rgba(124,58,237,0.5)" : "none",
              color: screen === n.id ? "#fff" : hoveredNav === n.id ? "#c4b5fd" : "#64748b",
              position:"relative",
            }}>
            <div style={{ fontSize:20 }}>{n.icon}</div>
            <div style={{ fontSize:8, fontWeight:800, letterSpacing:0.5 }}>{n.label}</div>
            {screen === n.id && (
              <div style={{
                position:"absolute", right:-2, top:"50%", transform:"translateY(-50%)",
                width:4, height:28, borderRadius:4, background:"linear-gradient(#7c3aed,#2563eb)",
              }} />
            )}
          </button>
        ))}
        <div style={{ marginTop:"auto", cursor:"pointer" }} onClick={() => setScreen("home")}>
          <MascotSVG size={60} mood="happy" />
        </div>
      </aside>
      <main style={{ flex:1, overflow:"auto", position:"relative", zIndex:5, display:"flex", flexDirection:"column" }}>
        {screen === "home" && (
          <HomeScreen
            state={state}
            coachMsg={COACH_MSGS[coachIdx]}
            onPlay={() => { setPressedPlay(true); setTimeout(() => { setPressedPlay(false); setScreen("warmup"); }, 200); }}
            pressedPlay={pressedPlay}
            triggerConfetti={triggerConfetti}
            setScreen={setScreen}
          />
        )}
        {screen === "mission" && (
          <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>
            <MissionScreen
              state={state}
              setLevelStars={setLevelStars}
              setLevelProgress={setLevelProgress}
              addXP={addXP}
              addActivity={addActivity}
              addDailyJumps={addDailyJumps}
              addDailySquats={addDailySquats}
              tryAwardBadge={tryAwardBadge}
              triggerConfetti={triggerConfetti}
              showToast={showToast}
            />
          </div>
        )}
        {screen === "challenge" && (
          <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>
            <ChallengeScreen
              state={state}
              claimDailyChallenge={claimDailyChallenge}
              triggerConfetti={triggerConfetti}
              addXP={addXP}
              addActivity={addActivity}
            />
          </div>
        )}
        {screen === "warmup" && (
          <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>
            <WarmupScreen
              state={state}
              setState={setState}
              addXP={addXP}
              addActivity={addActivity}
              addDailyJumps={addDailyJumps}
              addDailySquats={addDailySquats}
              tryAwardBadge={tryAwardBadge}
              triggerConfetti={triggerConfetti}
              showToast={showToast}
            />
          </div>
        )}
        {screen === "minibattle" && (
          <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>
            <MiniBattleScreen
              state={state}
              setState={setState}
              addActivity={addActivity}
              tryAwardBadge={tryAwardBadge}
              triggerConfetti={triggerConfetti}
              showToast={showToast}
            />
          </div>
        )}
        {screen === "badges" && <BadgesScreen state={state} triggerConfetti={triggerConfetti} />}
        {screen === "stats"  && <StatsScreen state={state} addActivity={addActivity} />}
        {screen === "settings" && (
          <SettingsScreen state={state} setState={setState} resetState={resetState} showToast={showToast} />
        )}
      </main>
    </div>
  );
}

// ─── HOME SCREEN ──────────────────────────────────────────────────────────────
function HomeScreen({ state, coachMsg, onPlay, pressedPlay, triggerConfetti, setScreen }) {
  const xpToNext = 2000;
  const xpPercent = Math.round((state.exp / xpToNext) * 100);
  return (
    <div style={{ padding:"28px 32px", display:"flex", flexDirection:"column", gap:24, minHeight:"100vh" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div>
          <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Selamat Datang!</div>
          <div style={{ fontSize:28, fontWeight:900, color:"#fff", lineHeight:1.2 }}>Hai, {state.playerName}! 👋</div>
        </div>
        <div style={{ display:"flex", gap:10 }}>
          <div style={{ background:"linear-gradient(135deg,#fb923c,#ef4444)", borderRadius:14, padding:"8px 16px", display:"flex", alignItems:"center", gap:6, boxShadow:"0 4px 16px rgba(239,68,68,0.4)" }}>
            <span style={{ fontSize:18 }}>🔥</span>
            <div>
              <div style={{ fontSize:18, fontWeight:900, color:"#fff", lineHeight:1 }}>{state.streak}</div>
              <div style={{ fontSize:10, color:"rgba(255,255,255,0.8)", fontWeight:700 }}>STREAK</div>
            </div>
          </div>
          <div style={{ background:"linear-gradient(135deg,#7c3aed,#2563eb)", borderRadius:14, padding:"8px 16px", display:"flex", alignItems:"center", gap:6, boxShadow:"0 4px 16px rgba(124,58,237,0.4)" }}>
            <span style={{ fontSize:18 }}>⭐</span>
            <div>
              <div style={{ fontSize:18, fontWeight:900, color:"#fff", lineHeight:1 }}>{state.exp}</div>
              <div style={{ fontSize:10, color:"rgba(255,255,255,0.8)", fontWeight:700 }}>XP TOTAL</div>
            </div>
          </div>
        </div>
      </div>
      <div style={{
        background:"linear-gradient(135deg, rgba(124,58,237,0.3) 0%, rgba(37,99,235,0.2) 50%, rgba(16,185,129,0.15) 100%)",
        border:"1px solid rgba(167,139,250,0.25)", borderRadius:28, padding:"28px 32px",
        display:"flex", alignItems:"center", gap:28, position:"relative", overflow:"hidden", backdropFilter:"blur(10px)",
      }}>
        <div style={{ position:"absolute", top:-60, right:-60, width:200, height:200, borderRadius:"50%", background:"radial-gradient(circle, rgba(124,58,237,0.3) 0%, transparent 70%)", pointerEvents:"none" }} />
        <div style={{ animation:"mascotBob 3s ease-in-out infinite", flexShrink:0 }}>
          <MascotSVG size={140} mood="happy" glow />
        </div>
        <div style={{ flex:1 }}>
          <div style={{ display:"inline-flex", alignItems:"center", gap:8, background:"rgba(52,211,153,0.15)", border:"1px solid rgba(52,211,153,0.3)", borderRadius:50, padding:"4px 14px", marginBottom:12 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#34d399", animation:"pulse 1.5s ease-in-out infinite" }} />
            <span style={{ fontSize:11, color:"#34d399", fontWeight:800, letterSpacing:1 }}>AI COACH AKTIF</span>
          </div>
          <div style={{ background:"rgba(255,255,255,0.06)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:18, padding:"16px 20px", marginBottom:16, backdropFilter:"blur(8px)", minHeight:64, animation:"coachFadeIn 0.5s ease" }}>
            <p style={{ color:"#e2e8f0", fontSize:16, fontWeight:700, lineHeight:1.5, margin:0 }}>{coachMsg}</p>
          </div>
          <div style={{ marginBottom:8 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
              <span style={{ fontSize:12, color:"#94a3b8", fontWeight:700 }}>Level {state.level}</span>
              <span style={{ fontSize:12, color:"#a78bfa", fontWeight:700 }}>{state.exp} / {2000} XP</span>
            </div>
            <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:10, overflow:"hidden" }}>
              <div style={{ height:"100%", width:`${xpPercent}%`, background:"linear-gradient(90deg, #7c3aed, #2563eb, #34d399)", borderRadius:50, boxShadow:"0 0 10px rgba(124,58,237,0.6)", transition:"width 1s ease" }} />
            </div>
          </div>
          <div style={{ fontSize:12, color:"#64748b", fontWeight:600 }}>{2000 - state.exp} XP lagi menuju Level {state.level + 1} 🚀</div>
        </div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:14 }}>
        {[
          { label:"Gerakan Hari Ini", val:state.totalMoves, unit:"rep",   color:"#22c55e", glow:"rgba(34,197,94,0.3)",    icon:"💪" },
          { label:"Kalori Terbakar",  val:state.totalCalories, unit:"kkal", color:"#f97316", glow:"rgba(249,115,22,0.3)", icon:"🔥" },
          { label:"Total Bintang",    val:state.totalStars, unit:"bintang",color:"#3b82f6", glow:"rgba(59,130,246,0.3)",  icon:"⭐" },
          { label:"Badge Diraih",     val:Object.values(state.badges).filter(b=>b.unlocked).length, unit:"badge", color:"#a855f7", glow:"rgba(168,85,247,0.3)", icon:"🏅" },
        ].map((s, i) => (
          <div key={i} className="stat-card" style={{
            background:`linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03))`,
            border:`1px solid ${s.color}33`, borderRadius:20, padding:"18px 16px",
            backdropFilter:"blur(10px)", boxShadow:`0 4px 24px ${s.glow}`,
            animation:`cardEntrance 0.5s ${i * 0.1}s both ease`, cursor:"default",
          }}>
            <div style={{ fontSize:28, marginBottom:8 }}>{s.icon}</div>
            <div style={{ fontSize:26, fontWeight:900, color:s.color, lineHeight:1 }}>{s.val}</div>
            <div style={{ fontSize:11, color:"rgba(255,255,255,0.5)", fontWeight:700, textTransform:"uppercase", letterSpacing:0.5 }}>{s.unit}</div>
            <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:600, marginTop:4 }}>{s.label}</div>
          </div>
        ))}
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14 }}>
        {[
          { label:"AI Warm Up",   desc:"Pemanasan cerdas",    icon:"🤸", color:"#fb923c", glow:"rgba(251,146,60,0.35)",  screen:"warmup" },
          { label:"Mission Map",  desc:"Petualangan misi",    icon:"🗺️", color:"#3b82f6", glow:"rgba(59,130,246,0.35)", screen:"mission" },
          { label:"Mini Battle",  desc:"Lawan temanmu",       icon:"⚔️", color:"#ef4444", glow:"rgba(239,68,68,0.35)",  screen:"minibattle" },
        ].map((f, i) => (
          <button key={i} className="feature-btn" onClick={() => { AudioSystem.click(); setScreen(f.screen); }} style={{
            background:`linear-gradient(135deg, ${f.color}22, ${f.color}11)`,
            border:`1px solid ${f.color}44`, borderRadius:20, padding:"20px 16px",
            display:"flex", flexDirection:"column", alignItems:"flex-start", gap:8,
            cursor:"pointer", textAlign:"left", boxShadow:`0 4px 20px ${f.glow}`,
            transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
          }}>
            <div style={{ fontSize:32 }}>{f.icon}</div>
            <div style={{ fontSize:15, fontWeight:900, color:"#fff" }}>{f.label}</div>
            <div style={{ fontSize:12, color:"rgba(255,255,255,0.5)", fontWeight:600 }}>{f.desc}</div>
          </button>
        ))}
      </div>
      <button onClick={onPlay} style={{
        width:"100%", padding:"20px",
        background: pressedPlay ? "linear-gradient(135deg,#16a34a,#1d4ed8)" : "linear-gradient(135deg,#22c55e 0%,#16a34a 40%,#2563eb 100%)",
        border:"none", borderRadius:24, cursor:"pointer",
        fontSize:22, fontWeight:900, color:"#fff", letterSpacing:2, textTransform:"uppercase",
        boxShadow: pressedPlay ? "0 2px 10px rgba(34,197,94,0.3)" : "0 8px 40px rgba(34,197,94,0.5), 0 4px 12px rgba(37,99,235,0.3)",
        transform: pressedPlay ? "scale(0.97)" : "scale(1)", transition:"all 0.15s ease",
        animation:"playPulse 2.5s ease-in-out infinite",
        display:"flex", alignItems:"center", justifyContent:"center", gap:12,
        fontFamily:"'Nunito', sans-serif",
      }}>
        <span style={{ fontSize:28 }}>▶</span>
        MULAI BERMAIN SEKARANG!
        <span style={{ fontSize:28 }}>🚀</span>
      </button>
    </div>
  );
}

// ─── MISSION SCREEN ───────────────────────────────────────────────────────────
function MissionScreen({ state, setLevelStars, setLevelProgress, addXP, addActivity, addDailyJumps, addDailySquats, tryAwardBadge, triggerConfetti, showToast }) {
  const [selected, setSelected] = useState(null);
  const [activeMission, setActiveMission] = useState(null);
  const [gameState, setGameState] = useState(null);
  const timerRef = useRef(null);
  const activeMissionRef = useRef(null);
  activeMissionRef.current = activeMission;

  const startMissionGame = (m) => {
    setActiveMission(m);
    setGameState({ reps:0, score:0, combo:1, timeLeft:30, phase:"countdown", count:3, currentMoveIdx:0 });
    AudioSystem.click();
    let c = 3;
    const cd = setInterval(() => {
      c--;
      AudioSystem.countdown(c === 0 ? 0 : c);
      setGameState(prev => ({ ...prev, count: c }));
      if (c <= 0) {
        clearInterval(cd);
        setGameState(prev => ({ ...prev, phase:"playing" }));
        startTimer();
      }
    }, 900);
  };

  const startTimer = () => {
    timerRef.current = setInterval(() => {
      setGameState(prev => {
        if (!prev || prev.phase !== "playing") { clearInterval(timerRef.current); return prev; }
        if (prev.timeLeft <= 1) {
          clearInterval(timerRef.current);
          return { ...prev, timeLeft:0, phase:"result" };
        }
        return { ...prev, timeLeft: prev.timeLeft - 1 };
      });
    }, 1000);
  };

  const doRep = () => {
    if (!gameState || gameState.phase !== "playing" || !activeMission) return;
    AudioSystem.rep();

    // ── Catat jenis gerakan untuk Daily Challenge tracker ──────────────────
    const currentMoveId = activeMission.moves[gameState.currentMoveIdx % activeMission.moves.length];
    if (currentMoveId === "jump") addDailyJumps(1);
    if (currentMoveId === "squat") addDailySquats(1);

    setGameState(prev => {
      const newReps = prev.reps + 1;
      const newCombo = newReps % 3 === 0 ? Math.min(5, prev.combo + 1) : prev.combo;
      const pts = 10 * newCombo;
      const newScore = prev.score + pts;
      if (newCombo > prev.combo) AudioSystem.combo();
      if (newReps >= activeMission.target) {
        clearInterval(timerRef.current);
        return { ...prev, reps: newReps, score: newScore, combo: newCombo, phase:"result" };
      }
      const nextMoveIdx = (prev.currentMoveIdx + 1) % activeMission.moves.length;
      return { ...prev, reps: newReps, score: newScore, combo: newCombo, currentMoveIdx: nextMoveIdx };
    });
  };

  useEffect(() => {
    if (gameState?.phase === "result" && activeMission) {
      const reps = gameState.reps;
      const target = activeMission.target;
      const completed = reps >= target;

      // ── Hitung progres nyata: (reps / target) × 100, max 100 ──────────────
      const progressPercent = Math.min(100, Math.round((reps / target) * 100));

      // Simpan progres (hanya naik, tidak pernah turun)
      setLevelProgress(activeMission.id, progressPercent);

      // Bintang hanya diberikan jika benar-benar selesai (100%)
      const stars = completed
        ? (gameState.score >= target * 25 ? 3 : gameState.score >= target * 15 ? 2 : 1)
        : 0; // belum 100% → belum dapat bintang

      if (stars > 0) setLevelStars(activeMission.id, stars);
      addXP(Math.floor(progressPercent / 10) + Math.floor(gameState.score / 4));
      addActivity({ type:"mission", name: activeMission.name, score: gameState.score, icon: activeMission.icon });
      if (stars === 3) tryAwardBadge("perfect_score");
      if (completed) { triggerConfetti(); AudioSystem.win(); }
      else if (reps > 0) { AudioSystem.fail(); }
    }
  }, [gameState?.phase]);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  if (activeMission && gameState) {
    return (
      <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden", position:"relative" }}>
        <MiniGameView
          mission={activeMission}
          gameState={gameState}
          onRep={doRep}
          onBack={() => { clearInterval(timerRef.current); setActiveMission(null); setGameState(null); }}
          onReplay={() => startMissionGame(activeMission)}
          savedProgress={state.levelProgress?.[activeMission.id] ?? 0}
        />
      </div>
    );
  }

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Petualangan</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Mission Map 🗺️</div>
        <div style={{ fontSize:13, color:"rgba(255,255,255,0.4)", marginTop:4 }}>
          ⭐ Total Bintang: {state.totalStars} / 15
        </div>
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:0, position:"relative" }}>
        {MISSIONS_DATA.map((m, i) => {
          const isLeft = i % 2 === 0;
          const unlocked = i === 0 ? true : (state.levelUnlocked[m.id] ?? false);
          const stars = state.levelStars[m.id] ?? 0;
          // ── Ambil progres tersimpan ──
          const savedPct = state.levelProgress?.[m.id] ?? 0;
          const isSelected = selected === m.id;
          return (
            <div key={m.id} style={{ display:"flex", flexDirection:"column", alignItems: isLeft ? "flex-start" : "flex-end", marginBottom:8 }}>
              {i > 0 && (
                <div style={{
                  width:4, height:40, background:`linear-gradient(${MISSIONS_DATA[i-1].color}, ${m.color})`,
                  borderRadius:4, marginLeft: isLeft ? 68 : "auto",
                  marginRight: isLeft ? "auto" : 68,
                  opacity: unlocked ? 1 : 0.3,
                }} />
              )}
              <button
                disabled={!unlocked}
                onClick={() => { if (unlocked) { setSelected(isSelected ? null : m.id); AudioSystem.click(); } }}
                className="mission-node"
                style={{
                  display:"flex", alignItems:"center", gap:16,
                  background:`linear-gradient(135deg, ${m.color}22, ${m.color}11)`,
                  border: isSelected ? `2px solid ${m.color}` : `1px solid ${m.color}44`,
                  borderRadius:24, padding:"14px 20px",
                  cursor: unlocked ? "pointer" : "not-allowed",
                  opacity: unlocked ? 1 : 0.5,
                  boxShadow: isSelected ? `0 0 30px ${m.glow}` : `0 4px 16px ${m.glow}`,
                  width:"85%",
                  transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
                  transform: isSelected ? "scale(1.03)" : "scale(1)",
                }}>
                <div style={{
                  width:64, height:64, borderRadius:18, flexShrink:0,
                  background:`linear-gradient(135deg, ${m.color}, ${m.color}aa)`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontSize:28, boxShadow:`0 4px 16px ${m.glow}`,
                }}>
                  {unlocked ? m.icon : "🔒"}
                </div>
                <div style={{ flex:1, textAlign:"left" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:4 }}>
                    <span style={{ fontSize:16, fontWeight:900, color:"#fff" }}>{m.name}</span>
                    {i === MISSIONS_DATA.length - 1 && <span style={{ fontSize:10, background:"#ef4444", color:"#fff", borderRadius:50, padding:"2px 8px", fontWeight:800 }}>BOSS</span>}
                  </div>
                  <div style={{ fontSize:12, color:"rgba(255,255,255,0.5)", fontWeight:600, marginBottom:6 }}>{m.desc}</div>

                  {/* ── Bintang (hanya muncul jika selesai 100%) ── */}
                  <div style={{ display:"flex", gap:4, marginBottom: savedPct > 0 ? 6 : 0 }}>
                    {[0,1,2].map(s => <StarIcon key={s} filled={s < stars} size={16} />)}
                  </div>

                  {/* ── Bar progres (selalu tampil jika pernah dimainkan) ── */}
                  {unlocked && savedPct > 0 && (
                    <div>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                        <span style={{ fontSize:10, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>Progres</span>
                        <span style={{
                          fontSize:10, fontWeight:900,
                          color: savedPct >= 100 ? "#22c55e" : m.color,
                        }}>
                          {savedPct}% {savedPct >= 100 ? "✅" : ""}
                        </span>
                      </div>
                      <div style={{ background:"rgba(255,255,255,0.1)", borderRadius:50, height:5, overflow:"hidden" }}>
                        <div style={{
                          height:"100%",
                          width:`${savedPct}%`,
                          background: savedPct >= 100
                            ? "linear-gradient(90deg,#22c55e,#16a34a)"
                            : `linear-gradient(90deg, ${m.color}, ${m.color}aa)`,
                          borderRadius:50,
                          transition:"width 0.6s ease",
                          boxShadow: savedPct >= 100 ? "0 0 6px rgba(34,197,94,0.6)" : `0 0 4px ${m.color}88`,
                        }} />
                      </div>
                    </div>
                  )}
                </div>
                <div style={{ textAlign:"right", flexShrink:0 }}>
                  <div style={{ fontSize:18, fontWeight:900, color:m.color }}>+{m.target * 15}</div>
                  <div style={{ fontSize:10, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>XP</div>
                </div>
              </button>

              {isSelected && (
                <div style={{
                  width:"85%", background:"rgba(255,255,255,0.05)",
                  border:`1px solid ${m.color}33`, borderRadius:20, padding:20, marginTop:8,
                  backdropFilter:"blur(10px)", animation:"expandIn 0.3s ease",
                }}>
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
                    <div style={{ fontSize:14, fontWeight:700, color:"rgba(255,255,255,0.7)" }}>Target: {m.target} gerakan</div>
                    <div style={{ fontSize:14, fontWeight:700, color:m.color }}>🎯 Reward: {m.target * 15} XP</div>
                  </div>
                  <div style={{ fontSize:12, color:"rgba(255,255,255,0.5)", marginBottom:8 }}>
                    Gerakan: {m.moves.map(mv => ({ raise_both:"🙌", squat:"🦵", jump:"⬆️", run:"🏃", raise_right:"✋", raise_left:"🤚" })[mv]).join(" → ")}
                  </div>

                  {/* ── Info progres sebelumnya di panel detail ── */}
                  {savedPct > 0 && savedPct < 100 && (
                    <div style={{
                      background:`rgba(255,255,255,0.05)`, border:`1px solid ${m.color}33`,
                      borderRadius:12, padding:"10px 14px", marginBottom:12,
                      fontSize:12, color:"rgba(255,255,255,0.6)", fontWeight:700,
                    }}>
                      📊 Progres tersimpan: <span style={{ color: m.color, fontWeight:900 }}>{savedPct}%</span>
                      &nbsp;({Math.round(savedPct * m.target / 100)} dari {m.target} gerakan)
                      <br/>
                      <span style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:600 }}>
                        Lanjutkan untuk mencapai 100%!
                      </span>
                    </div>
                  )}
                  {savedPct >= 100 && (
                    <div style={{
                      background:"rgba(34,197,94,0.1)", border:"1px solid rgba(34,197,94,0.3)",
                      borderRadius:12, padding:"10px 14px", marginBottom:12,
                      fontSize:12, color:"#34d399", fontWeight:700,
                    }}>
                      ✅ Kamu sudah menyelesaikan misi ini! Main lagi untuk skor lebih tinggi.
                    </div>
                  )}

                  <button
                    onClick={() => { setSelected(null); startMissionGame(m); }}
                    style={{
                      width:"100%", padding:"14px",
                      background:`linear-gradient(135deg, ${m.color}, ${m.color}bb)`,
                      border:"none", borderRadius:14, cursor:"pointer",
                      fontSize:15, fontWeight:900, color:"#fff",
                      boxShadow:`0 4px 20px ${m.glow}`,
                      fontFamily:"'Nunito', sans-serif",
                      transition:"transform 0.2s",
                    }}>
                    {savedPct >= 100 ? "🔄 Main Lagi" : savedPct > 0 ? "▶ Lanjutkan Misi" : `▶ Mulai ${m.name}!`}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── MINI GAME VIEW ──────────────────────────────────────────────────────────
function MiniGameView({ mission, gameState, onRep, onBack, onReplay, savedProgress = 0 }) {
  const MOVE_LABELS = {
    raise_both:"🙌 Angkat Kedua Tangan!", raise_right:"✋ Angkat Tangan Kanan!",
    raise_left:"🤚 Angkat Tangan Kiri!", squat:"🦵 Squat!",
    jump:"⬆️ Lompat!", run:"🏃 Lari di Tempat!"
  };
  const MOVE_ICON_MAP = { raise_both:"🙌", squat:"🦵", jump:"⬆️", run:"🏃", raise_right:"✋", raise_left:"🤚" };

  const currentMove = mission.moves[gameState.currentMoveIdx % mission.moves.length];

  // ── Hitung progres sesi ini: (reps / target) × 100 ──────────────────────
  const sessionProgressPct = Math.min(100, Math.round((gameState.reps / mission.target) * 100));
  const completed = gameState.reps >= mission.target;

  // Bintang hanya jika selesai 100%
  const stars = gameState.phase === "result" && completed
    ? (gameState.score >= mission.target * 25 ? 3 : gameState.score >= mission.target * 15 ? 2 : 1)
    : 0;

  const cam = useCameraTracking();
  const [cameraOn, setCameraOn] = useState(false);
  const [aiRep, setAiRep] = useState(false);

  const toggleCamera = async () => {
    if (cameraOn) { cam.stopCamera(); setCameraOn(false); }
    else { setCameraOn(true); await cam.startCamera(); }
  };

  useEffect(() => {
    if (!cameraOn || gameState.phase !== "playing") return;
    cam.startLoop(currentMove, () => {
      setAiRep(true);
      setTimeout(() => setAiRep(false), 400);
      onRep();
    });
  }, [cameraOn, cam.cameraReady, gameState.phase, currentMove]);

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", overflow:"hidden" }}>

      {/* ── COUNTDOWN OVERLAY ── */}
      {gameState.phase === "countdown" && (
        <div style={{
          position:"absolute", inset:0, zIndex:100,
          display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center",
          background:"rgba(15,23,42,0.95)",
        }}>
          <div style={{ fontSize:120, fontWeight:900, color:"#a78bfa", animation:"mascotBob 0.5s ease-in-out infinite" }}>
            {gameState.count > 0 ? gameState.count : "GO!"}
          </div>
          <div style={{ fontSize:20, color:"#fff", fontWeight:800, marginTop:24 }}>{mission.icon} {mission.name}</div>
          <div style={{ position:"absolute", width:1, height:1, overflow:"hidden", opacity:0, pointerEvents:"none" }}>
            <CameraView videoRef={cam.videoRef} canvasRef={cam.canvasRef} cameraReady={cam.cameraReady} cameraError={cam.cameraError} poseDetected={cam.poseDetected} poseReady={cam.poseReady} height={1} />
          </div>
        </div>
      )}

      {/* ── RESULT OVERLAY ── */}
      {gameState.phase === "result" && (
        <div style={{
          position:"absolute", inset:0, zIndex:100,
          display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:20,
          background:"rgba(15,23,42,0.97)",
        }}>
          <div style={{ fontSize:80 }}>{completed ? "🎉" : sessionProgressPct >= 50 ? "💪" : "😅"}</div>
          <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>
            {completed ? "Berhasil! 🏆" : "Waktu Habis!"}
          </div>

          {/* ── Tampilan progres aktual ── */}
          <div style={{
            background:"rgba(255,255,255,0.06)", border:`1px solid ${mission.color}44`,
            borderRadius:20, padding:"20px 32px", textAlign:"center", minWidth:260,
          }}>
            {/* Persentase besar */}
            <div style={{
              fontSize:56, fontWeight:900,
              color: completed ? "#22c55e" : sessionProgressPct >= 50 ? "#fbbf24" : "#f97316",
              lineHeight:1,
            }}>
              {sessionProgressPct}%
            </div>
            <div style={{ fontSize:13, color:"rgba(255,255,255,0.5)", fontWeight:700, marginTop:4, marginBottom:12 }}>
              {gameState.reps} dari {mission.target} gerakan benar
            </div>

            {/* Bar progres sesi ini */}
            <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:10, overflow:"hidden", marginBottom:12 }}>
              <div style={{
                height:"100%",
                width:`${sessionProgressPct}%`,
                background: completed
                  ? "linear-gradient(90deg,#22c55e,#16a34a)"
                  : `linear-gradient(90deg, ${mission.color}, ${mission.color}aa)`,
                borderRadius:50, transition:"width 0.8s ease",
                boxShadow: completed ? "0 0 10px rgba(34,197,94,0.5)" : `0 0 8px ${mission.color}66`,
              }} />
            </div>

            {/* Bintang hanya tampil jika 100% */}
            {completed && (
              <div style={{ fontSize:28, fontWeight:900, color:"#fbbf24", marginBottom:4 }}>
                {[...Array(3)].map((_, i) => i < stars ? "⭐" : "☆").join("")}
              </div>
            )}

            {/* Skor */}
            <div style={{ fontSize:20, fontWeight:900, color: completed ? "#22c55e" : "#94a3b8" }}>
              {gameState.score} pts
            </div>

            {/* Pesan kontekstual */}
            <div style={{ marginTop:10, fontSize:12, color:"rgba(255,255,255,0.5)", fontWeight:600 }}>
              {completed
                ? "🎊 Sempurna! Semua gerakan selesai!"
                : sessionProgressPct >= 80
                  ? `Hampir! Kamu hanya perlu ${mission.target - gameState.reps} gerakan lagi.`
                  : sessionProgressPct >= 50
                    ? `Sudah setengah jalan! Terus latihan ya! 💪`
                    : `Jangan menyerah! Coba lagi dan tingkatkan progresmu.`}
            </div>

            {/* Progres tersimpan sebelumnya */}
            {savedProgress > 0 && !completed && (
              <div style={{ marginTop:8, fontSize:11, color:"rgba(255,255,255,0.35)", fontWeight:600 }}>
                Progres terbaik sebelumnya: {Math.max(savedProgress, sessionProgressPct)}%
              </div>
            )}
          </div>

          <div style={{ display:"flex", gap:14 }}>
            <button onClick={onReplay} style={{
              padding:"14px 28px", background:"linear-gradient(135deg,#22c55e,#16a34a)",
              border:"none", borderRadius:16, color:"#fff", fontWeight:900, fontSize:16,
              cursor:"pointer", fontFamily:"'Nunito', sans-serif",
            }}>
              🔄 Coba Lagi
            </button>
            <button onClick={onBack} style={{
              padding:"14px 28px", background:"rgba(255,255,255,0.1)",
              border:"1px solid rgba(255,255,255,0.2)", borderRadius:16, color:"#fff",
              fontWeight:800, fontSize:16, cursor:"pointer", fontFamily:"'Nunito', sans-serif",
            }}>
              🗺️ Kembali
            </button>
          </div>
        </div>
      )}

      {/* ── TOP BAR ── */}
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"12px 20px", borderBottom:"1px solid rgba(255,255,255,0.08)",
        background:"rgba(15,23,42,0.7)", flexShrink:0,
      }}>
        <div style={{ fontSize:18, fontWeight:900, color:"#fff" }}>{mission.icon} {mission.name}</div>
        <div style={{ display:"flex", gap:8 }}>
          <button onClick={toggleCamera} style={{
            background: cameraOn ? "linear-gradient(135deg,#22c55e,#16a34a)" : "rgba(255,255,255,0.08)",
            border: cameraOn ? "none" : "1px solid rgba(255,255,255,0.15)",
            borderRadius:12, padding:"8px 14px", color:"#fff", fontSize:12, fontWeight:800,
            cursor:"pointer", fontFamily:"'Nunito', sans-serif",
            display:"flex", alignItems:"center", gap:6,
          }}>
            📷 {cameraOn ? "AI ON" : "Aktifkan AI"}
          </button>
          <button onClick={onBack} style={{
            background:"rgba(255,255,255,0.1)", border:"1px solid rgba(255,255,255,0.2)",
            borderRadius:12, padding:"8px 16px", color:"#94a3b8", fontSize:13, fontWeight:700,
            cursor:"pointer", fontFamily:"'Nunito', sans-serif",
          }}>← Keluar</button>
        </div>
      </div>

      {/* ── SCROLLABLE BODY ── */}
      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden", minHeight:0 }}>
        <div style={{ padding:"20px 24px", display:"flex", flexDirection:"column", gap:16, maxWidth:980, margin:"0 auto", width:"100%" }}>

          {aiRep && (
            <div style={{
              background:"linear-gradient(135deg,#22c55e,#16a34a)", borderRadius:14,
              padding:"10px 20px", textAlign:"center", fontSize:16, fontWeight:900, color:"#fff",
              animation:"coachFadeIn 0.2s ease",
            }}>
              🤖 AI Mendeteksi Gerakan! +1 Rep
            </div>
          )}

          {/* Statistik */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12 }}>
            {[
              { label:"Waktu",  val:gameState.timeLeft, icon:"⏱️", color: gameState.timeLeft < 10 ? "#ef4444" : "#22c55e" },
              { label:"Skor",   val:gameState.score,    icon:"⭐",  color:"#fbbf24" },
              { label:"Combo",  val:`x${gameState.combo}`, icon:"🔥", color:"#f97316" },
            ].map((s,i) => (
              <div key={i} style={{
                background:"rgba(255,255,255,0.05)", border:`1px solid ${s.color}33`,
                borderRadius:18, padding:"16px", textAlign:"center",
              }}>
                <div style={{ fontSize:20, marginBottom:4 }}>{s.icon}</div>
                <div style={{ fontSize:28, fontWeight:900, color:s.color }}>{s.val}</div>
                <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{s.label}</div>
              </div>
            ))}
          </div>

          {/* AI Camera */}
          <div style={{ position:"relative", borderRadius:24, overflow:"hidden", boxShadow:"0 16px 50px rgba(0,0,0,0.45)", flexShrink:0 }}>
            <div style={{ position:"absolute", top:12, left:12, zIndex:5, display:"flex", alignItems:"center", gap:6 }}>
              <div style={{
                background:"rgba(0,0,0,0.6)", border:"1px solid rgba(255,255,255,0.15)",
                borderRadius:50, padding:"4px 12px", fontSize:11, fontWeight:700, color:"#fff",
                display:"flex", alignItems:"center", gap:6, backdropFilter:"blur(8px)",
              }}>
                <div style={{ width:7, height:7, borderRadius:"50%", background: cameraOn ? "#22c55e" : "#64748b", animation: cameraOn ? "pulse 1.5s ease-in-out infinite" : "none" }} />
                Kamu — Live
              </div>
            </div>
            <CameraView
              videoRef={cam.videoRef} canvasRef={cam.canvasRef}
              cameraReady={cam.cameraReady} cameraError={cam.cameraError}
              poseDetected={cam.poseDetected} poseReady={cam.poseReady}
              height={"min(58vh, 540px)"} style={{ borderRadius:24 }}
            />
            {cameraOn && (
              <div style={{
                position:"absolute", bottom:0, left:0, right:0, padding:"10px 14px",
                background:"linear-gradient(to top, rgba(0,0,0,0.85), transparent)",
                fontSize:11, fontWeight:700, color:"#34d399",
              }}>
                📡 AI mendeteksi gerakan otomatis
              </div>
            )}
          </div>

          {/* Instruksi gerakan + progres bar ── DIPERBARUI ── */}
          <div style={{
            background:`linear-gradient(135deg, ${mission.color}22, ${mission.color}11)`,
            border:`1px solid ${mission.color}44`, borderRadius:24, padding:"28px 24px", textAlign:"center",
            animation:"coachFadeIn 0.3s ease",
          }}>
            <div style={{ fontSize:48, marginBottom:12 }}>{MOVE_ICON_MAP[currentMove] || "✋"}</div>
            <div style={{ fontSize:22, fontWeight:900, color:"#fff", marginBottom:8 }}>
              {MOVE_LABELS[currentMove] || currentMove}
            </div>

            {/* Hitungan reps */}
            <div style={{ fontSize:14, color:"rgba(255,255,255,0.5)", fontWeight:600, marginBottom:4 }}>
              {gameState.reps} / {mission.target} gerakan benar
            </div>

            {/* Persentase progres real-time */}
            <div style={{
              fontSize:18, fontWeight:900, marginBottom:12,
              color: sessionProgressPct >= 100 ? "#22c55e" : sessionProgressPct >= 50 ? "#fbbf24" : mission.color,
            }}>
              {sessionProgressPct}% selesai
            </div>

            {/* Bar progres */}
            <div style={{ marginTop:4, background:"rgba(255,255,255,0.08)", borderRadius:50, height:12, overflow:"hidden" }}>
              <div style={{
                height:"100%",
                width:`${sessionProgressPct}%`,
                background: sessionProgressPct >= 100
                  ? "linear-gradient(90deg,#22c55e,#16a34a)"
                  : `linear-gradient(90deg, ${mission.color}, ${mission.color}aa)`,
                borderRadius:50,
                transition:"width 0.3s ease",
                boxShadow: sessionProgressPct >= 100 ? "0 0 10px rgba(34,197,94,0.5)" : `0 0 6px ${mission.color}66`,
              }} />
            </div>

            {cameraOn && (
              <div style={{ marginTop:10, fontSize:12, color:"#34d399", fontWeight:700 }}>
                📡 Gerakan terdeteksi otomatis via AI Camera
              </div>
            )}
          </div>

          {/* TAP button — manual fallback */}
          {!cameraOn && (
            <button
              onClick={onRep}
              style={{
                width:"100%", padding:"28px",
                background:`linear-gradient(135deg, ${mission.color}, ${mission.color}bb)`,
                border:"none", borderRadius:24, cursor:"pointer",
                fontSize:24, fontWeight:900, color:"#fff",
                boxShadow:`0 8px 40px ${mission.glow}`,
                animation:"playPulse 1.5s ease-in-out infinite",
                fontFamily:"'Nunito', sans-serif", transition:"transform 0.1s",
              }}
              onMouseDown={e => e.currentTarget.style.transform = "scale(0.97)"}
              onMouseUp={e => e.currentTarget.style.transform = "scale(1)"}
            >
              👆 TAP = 1 Gerakan Benar!
            </button>
          )}

          <div style={{ fontSize:12, color:"rgba(255,255,255,0.3)", textAlign:"center", fontWeight:600, paddingBottom:8 }}>
            {cameraOn
              ? "📡 AI Camera aktif — lakukan gerakan di depan kamera!"
              : `Tap tombol setiap kali gerakan selesai. Target: ${mission.target} gerakan (100%)`}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── CHALLENGE SCREEN ─────────────────────────────────────────────────────────
function ChallengeScreen({ state, claimDailyChallenge, triggerConfetti, addXP, addActivity }) {
  // Bangun challenges dari state nyata (progres real-time)
  const challenges = buildDailyChallenges(state);
  const [timeStr, setTimeStr] = useState(() => {
    // Hitung sisa waktu sampai tengah malam
    const now = new Date();
    const midnight = new Date(now); midnight.setHours(24,0,0,0);
    const secs = Math.floor((midnight - now) / 1000);
    const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60), s = secs%60;
    return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
  });

  useEffect(() => {
    const t = setInterval(() => {
      setTimeStr(prev => {
        const [h,m,s] = prev.split(":").map(Number);
        let total = h*3600+m*60+s-1;
        if (total < 0) total = 86399;
        return `${String(Math.floor(total/3600)).padStart(2,"0")}:${String(Math.floor((total%3600)/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`;
      });
    }, 1000);
    return () => clearInterval(t);
  }, []);

  const [challengeActive, setChallengeActive] = useState(false);
  const [round, setRound] = useState(0);
  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(1);
  const [currentChallenge, setCurrentChallenge] = useState(null);
  const [timeLeft, setTimeLeft] = useState(5);
  const [feedback, setFeedback] = useState("Siap-siap!");
  const [history, setHistory] = useState([]);
  const [gameOver, setGameOver] = useState(false);
  const timerRef = useRef(null);
  const TOTAL_ROUNDS = 10;

  const cam = useCameraTracking();
  const [cameraOn, setCameraOn] = useState(false);
  const [emotionMsg, setEmotionMsg] = useState("");

  const toggleCamera = async () => {
    if (cameraOn) { cam.stopCamera(); setCameraOn(false); setEmotionMsg(""); }
    else { setCameraOn(true); await cam.startCamera(); }
  };

  useEffect(() => {
    if (!cameraOn) return;
    const emotionInterval = setInterval(() => {
      if (cam.lastAngles) {
        const emotion = EmotionDetector.detect(cam.lastAngles);
        setEmotionMsg(EmotionDetector.getMessage(emotion));
      }
    }, 3000);
    return () => clearInterval(emotionInterval);
  }, [cameraOn, cam.lastAngles]);

  useEffect(() => {
    if (!cameraOn || !challengeActive || !currentChallenge || gameOver) return;
    cam.startLoop(currentChallenge.move, () => handleCorrect());
  }, [cameraOn, challengeActive, currentChallenge?.move, gameOver]);

  const startMotionChallenge = () => {
    setScore(0); setCombo(1); setRound(0); setHistory([]); setGameOver(false);
    setChallengeActive(true);
    AudioSystem.click();
    nextRound(0, 0, 1, []);
  };

  const nextRound = (currentRound, currentScore, currentCombo, currentHistory) => {
    const nr = currentRound + 1;
    if (nr > TOTAL_ROUNDS) { endGame(currentScore, currentHistory); return; }
    const c = MOTION_CHALLENGES[Math.floor(Math.random() * MOTION_CHALLENGES.length)];
    setRound(nr); setCurrentChallenge(c); setTimeLeft(5); setFeedback("Siap?");
    AudioSystem.countdown(3);
    let tl = 5;
    clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      tl--;
      setTimeLeft(tl);
      AudioSystem.countdown(tl);
      if (tl <= 0) {
        clearInterval(timerRef.current);
        setCombo(1);
        setFeedback("😢 Waktu habis!");
        setHistory(prev => [{ challenge: c, correct: false }, ...prev].slice(0, 8));
        setTimeout(() => nextRound(nr, currentScore, 1, [{ challenge: c, correct: false }, ...currentHistory].slice(0, 8)), 1200);
      }
    }, 1000);
  };

  const handleCorrect = useCallback(() => {
    if (!currentChallenge || gameOver) return;
    clearInterval(timerRef.current);
    const pts = Math.max(10, timeLeft * 20) * combo;
    const newScore = score + pts;
    const newCombo = combo + 1;
    const newHistory = [{ challenge: currentChallenge, correct: true }, ...history].slice(0, 8);
    setScore(newScore); setCombo(newCombo); setHistory(newHistory);
    setFeedback(newCombo > 3 ? `🔥 COMBO x${newCombo}! +${pts}pts!` : `✅ Bagus! +${pts}pts!`);
    if (newCombo > 3) AudioSystem.combo(); else AudioSystem.rep();
    triggerConfetti();
    setTimeout(() => nextRound(round, newScore, newCombo, newHistory), 900);
  }, [currentChallenge, gameOver, timeLeft, combo, score, history, round]);

  const endGame = (finalScore, finalHistory) => {
    clearInterval(timerRef.current);
    setGameOver(true);
    setFeedback(`🎉 Selesai! Skor: ${finalScore}`);
    addXP(Math.floor(finalScore / 10));
    addActivity({ type:"challenge", name:"Motion Challenge", score: finalScore, icon:"⚡" });
    AudioSystem.win(); triggerConfetti();
  };

  const timerRatio = timeLeft / 5;
  const timerColor = timerRatio > 0.5 ? "#4ade80" : timerRatio > 0.25 ? "#eab308" : "#ef4444";

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", overflow:"hidden" }}>
      <div style={{
        padding:"16px 28px", borderBottom:"1px solid rgba(255,255,255,0.08)",
        background:"rgba(15,23,42,0.7)", flexShrink:0,
        display:"flex", alignItems:"center", justifyContent:"space-between",
      }}>
        <div>
          <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Harian</div>
          <div style={{ fontSize:22, fontWeight:900, color:"#fff" }}>Daily Challenges ⚡</div>
        </div>
        <button onClick={toggleCamera} style={{
          background: cameraOn ? "linear-gradient(135deg,#22c55e,#16a34a)" : "rgba(255,255,255,0.08)",
          border: cameraOn ? "none" : "1px solid rgba(255,255,255,0.15)",
          borderRadius:12, padding:"9px 16px", color:"#fff", fontSize:12, fontWeight:800,
          cursor:"pointer", fontFamily:"'Nunito', sans-serif",
          display:"flex", alignItems:"center", gap:6,
        }}>
          📷 {cameraOn ? "AI Camera ON" : "Aktifkan AI Camera"}
        </button>
      </div>

      <div style={{ flex:1, overflowY:"auto", overflowX:"hidden", minHeight:0 }}>
        <div style={{ padding:"20px 28px", maxWidth:980, margin:"0 auto", width:"100%" }}>
          <div style={{
            background:"linear-gradient(135deg, rgba(239,68,68,0.2), rgba(249,115,22,0.1))",
            border:"1px solid rgba(239,68,68,0.3)", borderRadius:20, padding:"14px 20px",
            display:"flex", alignItems:"center", gap:12, marginBottom:20,
          }}>
            <span style={{ fontSize:24 }}>⏰</span>
            <div>
              <div style={{ fontSize:14, fontWeight:900, color:"#fca5a5" }}>Reset dalam {timeStr}</div>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:600 }}>Selesaikan sebelum waktu habis!</div>
            </div>
          </div>

          <div style={{ position:"relative", borderRadius:24, overflow:"hidden", boxShadow:"0 16px 50px rgba(0,0,0,0.45)", marginBottom:28 }}>
            <div style={{ position:"absolute", top:12, left:12, zIndex:5, display:"flex", alignItems:"center", gap:6 }}>
              <div style={{
                background:"rgba(0,0,0,0.6)", border:"1px solid rgba(255,255,255,0.15)",
                borderRadius:50, padding:"4px 12px", fontSize:11, fontWeight:700, color:"#fff",
                display:"flex", alignItems:"center", gap:6, backdropFilter:"blur(8px)",
              }}>
                <div style={{ width:7, height:7, borderRadius:"50%", background: cameraOn ? "#22c55e" : "#64748b", animation: cameraOn ? "pulse 1.5s ease-in-out infinite" : "none" }} />
                Kamu — Live
              </div>
            </div>
            <CameraView
              videoRef={cam.videoRef} canvasRef={cam.canvasRef}
              cameraReady={cam.cameraReady} cameraError={cam.cameraError}
              poseDetected={cam.poseDetected} poseReady={cam.poseReady}
              height={"min(58vh, 540px)"} style={{ borderRadius:24 }}
            />
            {cameraOn && (
              <div style={{ position:"absolute", bottom:0, left:0, right:0, padding:"10px 14px", background:"linear-gradient(to top, rgba(0,0,0,0.85), transparent)", fontSize:11, fontWeight:700, color:"#34d399" }}>
                📡 AI mendeteksi gerakan otomatis
              </div>
            )}
          </div>

          <div style={{ display:"flex", flexDirection:"column", gap:16, marginBottom:32 }}>
            {challenges.map((c, i) => {
              const isDone = c.done;
              const canClaim = c.progress >= 100 && !isDone;
              return (
                <div key={c.id} className="challenge-card" style={{
                  background:`linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03))`,
                  border:`1px solid ${isDone ? c.color + "66" : c.color + "22"}`,
                  borderRadius:24, padding:"20px 24px", backdropFilter:"blur(10px)",
                  boxShadow: isDone ? `0 4px 24px ${c.color}44` : "none",
                  animation:`cardEntrance 0.4s ${i * 0.1}s both ease`, opacity: isDone ? 1 : 0.88,
                }}>
                  <div style={{ display:"flex", alignItems:"center", gap:14, marginBottom:14 }}>
                    <div style={{
                      width:52, height:52, borderRadius:16, flexShrink:0,
                      background:`linear-gradient(135deg, ${c.color}44, ${c.color}22)`,
                      display:"flex", alignItems:"center", justifyContent:"center",
                      fontSize:24, border:`1px solid ${c.color}44`,
                    }}>
                      {isDone ? "✅" : c.icon}
                    </div>
                    <div style={{ flex:1 }}>
                      <div style={{ fontSize:16, fontWeight:900, color:isDone ? "#34d399" : "#fff", marginBottom:2 }}>{c.title}</div>
                      <div style={{ display:"flex", alignItems:"center", gap:8, flexWrap:"wrap" }}>
                        <div style={{ background:`${c.color}22`, border:`1px solid ${c.color}44`, borderRadius:50, padding:"2px 10px" }}>
                          <span style={{ fontSize:11, color:c.color, fontWeight:800 }}>{c.reward}</span>
                        </div>
                        {/* Hitungan nyata */}
                        <span style={{ fontSize:11, color:"rgba(255,255,255,0.45)", fontWeight:700 }}>
                          {c.current} / {c.target} {c.unit}
                        </span>
                      </div>
                    </div>
                    {/* Tombol KLAIM — hanya aktif bila 100% dan belum diklaim */}
                    {canClaim && (
                      <button
                        onClick={() => {
                          claimDailyChallenge(c.id, c.xp, addXP);
                          triggerConfetti();
                          AudioSystem.success();
                        }}
                        style={{
                          padding:"8px 16px", borderRadius:12, border:"none",
                          background:`linear-gradient(135deg, ${c.color}, ${c.color}aa)`,
                          color:"#fff", fontWeight:800, fontSize:12, cursor:"pointer",
                          fontFamily:"'Nunito', sans-serif", boxShadow:`0 4px 12px ${c.color}44`,
                          animation:"playPulse 1.5s ease-in-out infinite",
                        }}>
                        🎁 KLAIM
                      </button>
                    )}
                    {isDone && !canClaim && (
                      <div style={{ fontSize:11, color:"#34d399", fontWeight:800, padding:"6px 12px", background:"rgba(34,197,94,0.1)", border:"1px solid rgba(34,197,94,0.3)", borderRadius:10 }}>
                        ✓ Selesai
                      </div>
                    )}
                    {/* Progress belum 100% dan belum selesai — tampilkan lock */}
                    {!canClaim && !isDone && c.progress < 100 && (
                      <div style={{ fontSize:11, color:"rgba(255,255,255,0.3)", fontWeight:700, padding:"6px 12px", background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:10, whiteSpace:"nowrap" }}>
                        🔒 {c.target - c.current} lagi
                      </div>
                    )}
                  </div>
                  <div>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                      <span style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>Progres</span>
                      <span style={{
                        fontSize:11, fontWeight:900,
                        color: isDone ? "#34d399" : c.progress >= 100 ? c.color : c.color,
                      }}>
                        {c.progress}% {isDone ? "✅" : c.progress >= 100 ? "— Siap diklaim!" : ""}
                      </span>
                    </div>
                    <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:8, overflow:"hidden" }}>
                      <div style={{
                        height:"100%",
                        width:`${c.progress}%`,
                        background: isDone
                          ? "linear-gradient(90deg,#22c55e,#16a34a)"
                          : `linear-gradient(90deg, ${c.color}, ${c.color}aa)`,
                        borderRadius:50, transition:"width 0.6s ease",
                        boxShadow:`0 0 8px ${c.color}66`,
                      }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{
            background:"linear-gradient(135deg, rgba(124,58,237,0.25), rgba(37,99,235,0.15))",
            border:"1px solid rgba(167,139,250,0.3)", borderRadius:24, padding:"24px",
            position:"relative", overflow:"hidden",
          }}>
            <div style={{ position:"absolute", top:-30, right:-30, width:120, height:120, borderRadius:"50%", background:"radial-gradient(rgba(124,58,237,0.3), transparent)" }} />
            <div style={{ marginBottom:8 }}>
              <div style={{ fontSize:13, color:"#a78bfa", fontWeight:800, letterSpacing:1, textTransform:"uppercase" }}>⚡ MOTION CHALLENGE</div>
              <div style={{ fontSize:20, fontWeight:900, color:"#fff", marginTop:4 }}>Tebak Gerakan!</div>
              <div style={{ fontSize:13, color:"rgba(255,255,255,0.5)", marginTop:2 }}>10 ronde, ikuti perintah gerakan sebelum waktu habis!</div>
            </div>
            {emotionMsg && cameraOn && (
              <div style={{ background:"rgba(124,58,237,0.2)", border:"1px solid rgba(167,139,250,0.4)", borderRadius:12, padding:"8px 16px", fontSize:12, color:"#c4b5fd", fontWeight:700, marginBottom:10 }}>
                🤖 AI Coach: {emotionMsg}
              </div>
            )}
            {!challengeActive ? (
              <button onClick={startMotionChallenge} style={{ padding:"14px 28px", background:"linear-gradient(135deg,#7c3aed,#2563eb)", border:"none", borderRadius:16, color:"#fff", fontWeight:900, fontSize:16, cursor:"pointer", fontFamily:"'Nunito', sans-serif" }}>
                ▶ Mulai Motion Challenge
              </button>
            ) : (
              <div>
                <div style={{ display:"flex", gap:12, marginBottom:16 }}>
                  {[
                    { l:"Ronde", v:`${round}/${TOTAL_ROUNDS}`, c:"#a78bfa" },
                    { l:"Skor",  v:score,                      c:"#fbbf24" },
                    { l:"Combo", v:`x${combo}`,                c:"#f97316" },
                  ].map((s,i) => (
                    <div key={i} style={{ flex:1, background:"rgba(255,255,255,0.05)", borderRadius:14, padding:"10px", textAlign:"center" }}>
                      <div style={{ fontSize:18, fontWeight:900, color:s.c }}>{s.v}</div>
                      <div style={{ fontSize:10, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{s.l}</div>
                    </div>
                  ))}
                </div>
                {gameOver ? (
                  <div style={{ textAlign:"center" }}>
                    <div style={{ fontSize:24, fontWeight:900, color:"#fff", marginBottom:8 }}>🎉 Selesai! Skor: {score}</div>
                    <button onClick={startMotionChallenge} style={{ padding:"12px 24px", background:"linear-gradient(135deg,#7c3aed,#2563eb)", border:"none", borderRadius:14, color:"#fff", fontWeight:900, fontSize:15, cursor:"pointer", fontFamily:"'Nunito', sans-serif" }}>
                      🔄 Main Lagi
                    </button>
                  </div>
                ) : currentChallenge && (
                  <div>
                    <div style={{ display:"flex", alignItems:"center", gap:16, marginBottom:16 }}>
                      <svg width="80" height="80" viewBox="0 0 80 80">
                        <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />
                        <circle cx="40" cy="40" r="34" fill="none" stroke={timerColor} strokeWidth="8"
                          strokeDasharray={`${2*Math.PI*34}`}
                          strokeDashoffset={`${2*Math.PI*34*(1-timerRatio)}`}
                          strokeLinecap="round" transform="rotate(-90 40 40)"
                          style={{ transition:"stroke-dashoffset 0.3s, stroke 0.3s" }} />
                        <text x="40" y="46" textAnchor="middle" fill={timerColor} fontSize="22" fontWeight="900">{timeLeft}</text>
                      </svg>
                      <div style={{ flex:1 }}>
                        <div style={{ fontSize:20, fontWeight:900, color:"#fff", marginBottom:4 }}>{currentChallenge.text}</div>
                        <div style={{ fontSize:14, color:"rgba(255,255,255,0.5)" }}>{feedback}</div>
                        {cameraOn && <div style={{ fontSize:11, color:"#34d399", fontWeight:700, marginTop:4 }}>📡 AI mendeteksi gerakan otomatis</div>}
                      </div>
                    </div>
                    {!cameraOn && (
                      <button onClick={handleCorrect} style={{
                        width:"100%", padding:"18px", background:"linear-gradient(135deg,#22c55e,#16a34a)",
                        border:"none", borderRadius:16, color:"#fff", fontWeight:900, fontSize:18, cursor:"pointer",
                        fontFamily:"'Nunito', sans-serif", animation:"playPulse 1s ease-in-out infinite",
                        boxShadow:"0 4px 20px rgba(34,197,94,0.4)",
                      }}>
                        {currentChallenge.icon} Saya Sudah Lakukan!
                      </button>
                    )}
                    {history.length > 0 && (
                      <div style={{ marginTop:12, display:"flex", flexDirection:"column", gap:4 }}>
                        {history.slice(0,4).map((h,i) => (
                          <div key={i} style={{ display:"flex", alignItems:"center", gap:8, fontSize:11, color: h.correct ? "#34d399" : "#f87171", fontWeight:700 }}>
                            {h.correct ? "✅" : "❌"} {h.challenge.icon} {h.challenge.text}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {(() => {
            const WEEKLY_TARGET = 100;
            const weeklyCount = Math.min(state.weeklyMoves || 0, WEEKLY_TARGET);
            const weeklyPct = Math.min(100, Math.round((weeklyCount / WEEKLY_TARGET) * 100));
            const weeklyDone = weeklyPct >= 100;
            return (
              <div style={{
                marginTop:24, marginBottom:24, background:"linear-gradient(135deg, rgba(124,58,237,0.25), rgba(37,99,235,0.15))",
                border:`1px solid ${weeklyDone ? "rgba(251,191,36,0.5)" : "rgba(167,139,250,0.3)"}`, borderRadius:24, padding:"24px",
                position:"relative", overflow:"hidden",
                boxShadow: weeklyDone ? "0 0 24px rgba(251,191,36,0.2)" : "none",
              }}>
                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:8 }}>
                  <div style={{ fontSize:13, color: weeklyDone ? "#fbbf24" : "#a78bfa", fontWeight:800, letterSpacing:1, textTransform:"uppercase" }}>
                    {weeklyDone ? "🏆 WEEKLY BOSS" : "🌟 WEEKLY BOSS"}
                  </div>
                  {weeklyDone && (
                    <div style={{ fontSize:11, color:"#fbbf24", fontWeight:900, background:"rgba(251,191,36,0.15)", border:"1px solid rgba(251,191,36,0.4)", borderRadius:50, padding:"3px 12px" }}>
                      ✅ SELESAI MINGGU INI!
                    </div>
                  )}
                </div>
                <div style={{ fontSize:20, fontWeight:900, color:"#fff", marginBottom:4 }}>100 Gerakan Minggu Ini</div>
                <div style={{ fontSize:13, color:"rgba(255,255,255,0.5)", fontWeight:600, marginBottom:16 }}>
                  {weeklyDone ? "Luar biasa! Kamu sudah selesaikan tantangan mingguan! 🎉" : "Selesaikan untuk dapat badge eksklusif Diamond!"}
                </div>
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
                  <span style={{ fontSize:13, color:"rgba(255,255,255,0.7)", fontWeight:800 }}>
                    {weeklyCount} / {WEEKLY_TARGET} gerakan
                  </span>
                  <span style={{
                    fontSize:13, fontWeight:900,
                    color: weeklyDone ? "#fbbf24" : weeklyPct >= 50 ? "#a78bfa" : "#64748b",
                  }}>
                    {weeklyPct}% {weeklyDone ? "🏆" : weeklyPct >= 80 ? "🔥" : ""}
                  </span>
                </div>
                <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:14, overflow:"hidden" }}>
                  <div style={{
                    height:"100%",
                    width:`${weeklyPct}%`,
                    background: weeklyDone
                      ? "linear-gradient(90deg,#fbbf24,#f59e0b)"
                      : weeklyPct >= 50
                        ? "linear-gradient(90deg, #7c3aed, #2563eb)"
                        : "linear-gradient(90deg, #4c1d95, #1e1b4b)",
                    borderRadius:50,
                    transition:"width 0.8s ease",
                    boxShadow: weeklyDone ? "0 0 14px rgba(251,191,36,0.6)" : "0 0 12px rgba(124,58,237,0.6)",
                  }} />
                </div>
                {!weeklyDone && (
                  <div style={{ marginTop:10, fontSize:11, color:"rgba(255,255,255,0.35)", fontWeight:600 }}>
                    Sisa {WEEKLY_TARGET - weeklyCount} gerakan lagi untuk menyelesaikan Weekly Boss!
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}

// ─── WARMUP SCREEN ────────────────────────────────────────────────────────────
function WarmupScreen({ state, setState, addXP, addActivity, addDailyJumps, addDailySquats, tryAwardBadge, triggerConfetti, showToast }) {
  const [phase, setPhase]       = useState("idle");
  const [countdown, setCountdown] = useState(3);
  const [moveIdx, setMoveIdx]   = useState(0);
  const [reps, setReps]         = useState(0);
  const [energy, setEnergy]     = useState(0);
  const [feedback, setFeedback] = useState("Tekan tombol untuk mulai pemanasan!");
  const [soundEnabled, setSoundEnabled] = useState(true);
  const phaseRef   = useRef("idle");
  const moveIdxRef = useRef(0);
  const repsRef    = useRef(0);
  phaseRef.current   = phase;
  moveIdxRef.current = moveIdx;
  repsRef.current    = reps;

  const cam = useCameraTracking();
  const [cameraOn, setCameraOn] = useState(false);
  const [refFile, setRefFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const type = file.type.startsWith("video") ? "video" : "image";
    setRefFile({ url, type, name: file.name });
    setFeedback("📁 Referensi gerakan berhasil diupload!");
  };
  const removeRefFile = () => {
    if (refFile) URL.revokeObjectURL(refFile.url);
    setRefFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const toggleCamera = async () => {
    if (cameraOn) { cam.stopCamera(); setCameraOn(false); }
    else { setCameraOn(true); await cam.startCamera(); }
  };

  const toggleSound = () => {
    const on = AudioSystem.toggleSound();
    setSoundEnabled(on);
  };

  useEffect(() => {
    if (!cameraOn || phase !== "playing") return;
    const move = WARMUP_MOVES[moveIdxRef.current];
    if (!move) return;
    cam.startLoop(move.id, () => doRep());
  }, [cameraOn, phase, moveIdx]);

  const startWarmup = () => {
    AudioSystem.click();
    AudioSystem.voiceStart();
    setPhase("countdown"); setCountdown(3); setMoveIdx(0); setReps(0); setEnergy(0);
    let c = 3;
    const cd = setInterval(() => {
      c--;
      AudioSystem.countdown(c === 0 ? 0 : c);
      AudioSystem.voiceCountdown(c === 0 ? 0 : c);
      setCountdown(c);
      if (c <= 0) {
        clearInterval(cd);
        setPhase("playing");
        const firstMove = WARMUP_MOVES[0];
        setFeedback(firstMove.instruction);
        setTimeout(() => AudioSystem.voiceMove(firstMove.name, firstMove.instruction), 400);
      }
    }, 900);
  };

  const doRep = useCallback(() => {
    if (phaseRef.current !== "playing") return;
    const move = WARMUP_MOVES[moveIdxRef.current];
    AudioSystem.rep();

    // ── Catat jenis gerakan untuk Daily Challenge tracker ──────────────────
    if (move.id === "jump") addDailyJumps(1);
    if (move.id === "squat") addDailySquats(1);

    setReps(prev => {
      const newReps = prev + 1;
      const totalTarget = WARMUP_MOVES.reduce((a, m) => a + m.reps, 0);
      setEnergy(e => Math.min(100, e + (100 / totalTarget)));
      setFeedback(`${move.icon} Bagus! Rep ke-${newReps}!`);
      AudioSystem.voiceRep(newReps, move.reps);
      if (newReps >= move.reps) {
        const nextIdx = moveIdxRef.current + 1;
        if (nextIdx >= WARMUP_MOVES.length) {
          setPhase("done"); setEnergy(100);
          AudioSystem.win();
          setTimeout(() => AudioSystem.voiceDone(), 300);
          triggerConfetti();
          addXP(20);
          addActivity({ type:"warmup", name:"AI Warm-Up", score:100, icon:"🔥" });
          const warmupCount = (state.warmupCount || 0) + 1;
          setState(p => ({ ...p, warmupCount, energy: 100 }));
          tryAwardBadge("warmup_complete");
          if (warmupCount >= 5) tryAwardBadge("play_5");
          setFeedback("🎉 Pemanasan selesai! Energy penuh!");
        } else {
          AudioSystem.moveDone();
          setMoveIdx(nextIdx);
          const next = WARMUP_MOVES[nextIdx];
          setFeedback(`✅ ${move.name} selesai! Lanjut: ${next.name}`);
          setTimeout(() => AudioSystem.voiceMove(next.name, next.instruction), 600);
          return 0;
        }
      }
      return newReps;
    });
  }, [addDailyJumps, addDailySquats]);

  const resetWarmup = () => {
    setPhase("idle"); setEnergy(0); setMoveIdx(0); setReps(0);
    setFeedback("Tekan tombol untuk mulai pemanasan!");
  };

  const currentMove = WARMUP_MOVES[moveIdx];

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", overflow:"hidden" }}>
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"12px 20px", borderBottom:"1px solid rgba(255,255,255,0.08)",
        background:"rgba(15,23,42,0.7)", flexShrink:0, gap:12,
      }}>
        <div>
          <div style={{ fontSize:11, color:"#64748b", fontWeight:700, textTransform:"uppercase", letterSpacing:1 }}>Pemanasan</div>
          <div style={{ fontSize:20, fontWeight:900, color:"#fff" }}>AI Warm Up 🤸</div>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ background:"rgba(255,255,255,0.07)", border:"1px solid rgba(255,255,255,0.12)", borderRadius:50, padding:"5px 14px", display:"flex", alignItems:"center", gap:8 }}>
            <span style={{ fontSize:12, fontWeight:800, color:"#fb923c" }}>⚡ {Math.round(energy)}%</span>
            <div style={{ width:60, height:5, background:"rgba(255,255,255,0.1)", borderRadius:50, overflow:"hidden" }}>
              <div style={{ height:"100%", width:`${energy}%`, background:"linear-gradient(90deg,#fb923c,#ef4444)", borderRadius:50, transition:"width 0.4s" }} />
            </div>
          </div>
          <button onClick={toggleSound} style={{
            background: soundEnabled ? "rgba(99,102,241,0.2)" : "rgba(255,255,255,0.06)",
            border: soundEnabled ? "1px solid rgba(99,102,241,0.5)" : "1px solid rgba(255,255,255,0.1)",
            borderRadius:10, padding:"6px 12px", color: soundEnabled ? "#a78bfa" : "#64748b",
            fontSize:12, fontWeight:700, cursor:"pointer",
          }}>
            {soundEnabled ? "🔊 Suara" : "🔇 Mute"}
          </button>
        </div>
      </div>

      <div style={{ display:"flex", flex:1, overflow:"hidden", minHeight:0 }}>
        {/* LEFT PANEL */}
        <div style={{
          width:"50%", flexShrink:0, display:"flex", flexDirection:"column",
          borderRight:"1px solid rgba(255,255,255,0.07)",
          background:"rgba(15,23,42,0.6)", overflow:"hidden",
        }}>
          <div style={{ padding:"10px 16px 8px", borderBottom:"1px solid rgba(255,255,255,0.06)", display:"flex", alignItems:"center", gap:8 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#6366f1" }} />
            <span style={{ fontSize:11, fontWeight:700, color:"#64748b", textTransform:"uppercase", letterSpacing:1 }}>Gerakan Referensi</span>
          </div>
          <div style={{ flex:1, display:"flex", alignItems:"center", justifyContent:"center", background:"linear-gradient(160deg, #0f172a 0%, #1e1b4b 100%)", position:"relative", overflow:"hidden" }}>
            {phase === "playing" && (
              <>
                <div style={{ position:"absolute", width:140, height:140, borderRadius:"50%", border:"2px solid rgba(99,102,241,0.4)", animation:"warmupRing 2s ease-out infinite" }} />
                <div style={{ position:"absolute", width:140, height:140, borderRadius:"50%", border:"2px solid rgba(99,102,241,0.3)", animation:"warmupRing 2s 0.6s ease-out infinite" }} />
              </>
            )}
            {refFile ? (
              refFile.type === "image"
                ? <img src={refFile.url} alt="Ref" style={{ width:"100%", height:"100%", objectFit:"cover" }} />
                : <video src={refFile.url} autoPlay loop muted playsInline style={{ width:"100%", height:"100%", objectFit:"cover" }} />
            ) : (
              <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:10, textAlign:"center", padding:16 }}>
                <div style={{ fontSize:72, lineHeight:1, animation: phase === "playing" ? "warmupBounce 0.9s ease-in-out infinite" : "none", transition:"all 0.3s" }}>
                  {phase === "idle" ? "🔥" : phase === "countdown" ? (countdown > 0 ? countdown : "🚀") : phase === "done" ? "🎉" : (currentMove?.icon ?? "✅")}
                </div>
                <div style={{ fontSize:18, fontWeight:900, color:"#fff" }}>
                  {phase === "idle" && "Siap Bergerak!"}
                  {phase === "countdown" && (countdown > 0 ? "Bersiap..." : "Ayo!")}
                  {phase === "playing" && (currentMove?.name ?? "Selesai!")}
                  {phase === "done" && "Pemanasan Selesai!"}
                </div>
                <div style={{ fontSize:12, color:"#94a3b8", lineHeight:1.5, maxWidth:200 }}>
                  {phase === "idle" && "Tekan MULAI untuk memulai pemanasan"}
                  {phase === "countdown" && "Hitungan mundur..."}
                  {phase === "playing" && (currentMove?.instruction ?? "")}
                  {phase === "done" && "Energy penuh! Kerja bagus! 💪"}
                </div>
                {phase === "playing" && currentMove && (
                  <div style={{ background:"rgba(99,102,241,0.25)", border:"1px solid rgba(99,102,241,0.5)", borderRadius:50, padding:"6px 18px", fontSize:14, fontWeight:900, color:"#a78bfa" }}>
                    {reps} / {currentMove.reps}
                  </div>
                )}
              </div>
            )}
            {refFile && (
              <div style={{ position:"absolute", bottom:10, right:10, display:"flex", gap:6 }}>
                <button onClick={() => fileInputRef.current?.click()} style={{ background:"rgba(0,0,0,0.65)", border:"1px solid rgba(255,255,255,0.2)", borderRadius:8, padding:"4px 10px", fontSize:10, color:"#a78bfa", fontWeight:700, cursor:"pointer" }}>Ganti</button>
                <button onClick={removeRefFile} style={{ background:"rgba(0,0,0,0.65)", border:"1px solid rgba(239,68,68,0.4)", borderRadius:8, padding:"4px 10px", fontSize:10, color:"#f87171", fontWeight:700, cursor:"pointer" }}>Hapus</button>
              </div>
            )}
          </div>
          {!refFile && (
            <div onClick={() => fileInputRef.current?.click()} style={{ margin:"8px 10px", border:"1.5px dashed rgba(124,58,237,0.4)", borderRadius:12, background:"rgba(124,58,237,0.07)", padding:"12px", textAlign:"center", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:8, fontSize:12, fontWeight:700, color:"#7c3aed", flexShrink:0 }}
              onMouseEnter={e => e.currentTarget.style.background="rgba(124,58,237,0.14)"}
              onMouseLeave={e => e.currentTarget.style.background="rgba(124,58,237,0.07)"}>
              📁 Upload Referensi Gerakan (gambar / video)
            </div>
          )}
          <input ref={fileInputRef} type="file" accept="image/*,video/*" style={{ display:"none" }} onChange={handleFileUpload} />
          <div style={{ padding:"8px 10px", flexShrink:0, borderTop:"1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ fontSize:10, fontWeight:700, color:"#475569", textTransform:"uppercase", letterSpacing:1, marginBottom:6 }}>Daftar Gerakan</div>
            <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
              {WARMUP_MOVES.map((m, i) => {
                const isDone   = i < moveIdx || phase === "done";
                const isActive = i === moveIdx && phase === "playing";
                return (
                  <div key={m.id} style={{
                    display:"flex", alignItems:"center", gap:8, padding:"7px 10px", borderRadius:10,
                    background: isActive ? "rgba(251,146,60,0.12)" : isDone ? "rgba(34,197,94,0.08)" : "rgba(255,255,255,0.03)",
                    border: isActive ? "1px solid rgba(251,146,60,0.35)" : isDone ? "1px solid rgba(34,197,94,0.25)" : "1px solid rgba(255,255,255,0.05)",
                    fontSize:12, fontWeight:700,
                    color: isActive ? "#fb923c" : isDone ? "#34d399" : "#64748b",
                  }}>
                    <span style={{ fontSize:14 }}>{isDone ? "✅" : isActive ? "👉" : m.icon}</span>
                    <span style={{ flex:1 }}>{m.name} — {m.reps}x</span>
                    {isActive && <span style={{ fontWeight:900, color:"#fb923c" }}>{reps}/{m.reps}</span>}
                    {isDone && <span>Selesai!</span>}
                  </div>
                );
              })}
            </div>
          </div>
          <div style={{ padding:"10px", borderTop:"1px solid rgba(255,255,255,0.06)", display:"flex", flexDirection:"column", gap:7, flexShrink:0 }}>
            {phase === "idle" && (
              <button onClick={startWarmup} style={{ width:"100%", padding:"14px", background:"linear-gradient(135deg,#fb923c,#ef4444)", border:"none", borderRadius:50, color:"#fff", fontWeight:900, fontSize:15, cursor:"pointer", boxShadow:"0 4px 16px rgba(251,146,60,0.35)", display:"flex", alignItems:"center", justifyContent:"center", gap:8, animation:"playPulse 2s ease-in-out infinite" }}>
                🔥 MULAI PEMANASAN!
              </button>
            )}
            {phase === "playing" && !cameraOn && (
              <button onClick={doRep} style={{ width:"100%", padding:"14px", background:"linear-gradient(135deg,#fb923c,#ef4444)", border:"none", borderRadius:50, color:"#fff", fontWeight:900, fontSize:15, cursor:"pointer", animation:"playPulse 1.5s ease-in-out infinite" }}
                onMouseDown={e => e.currentTarget.style.transform="scale(0.97)"}
                onMouseUp={e => e.currentTarget.style.transform="scale(1)"}>
                👆 TAP = 1 {currentMove?.name}!
              </button>
            )}
            {phase === "playing" && cameraOn && (
              <div style={{ textAlign:"center", padding:"10px", background:"rgba(34,197,94,0.1)", border:"1px solid rgba(34,197,94,0.25)", borderRadius:12, fontSize:12, fontWeight:800, color:"#34d399" }}>
                📡 AI aktif — lakukan gerakan di depan kamera!
              </div>
            )}
            {phase === "done" && (
              <button onClick={resetWarmup} style={{ width:"100%", padding:"12px", background:"linear-gradient(135deg,#fb923c,#ef4444)", border:"none", borderRadius:50, color:"#fff", fontWeight:900, fontSize:14, cursor:"pointer" }}>
                🔄 Ulangi Pemanasan
              </button>
            )}
            <button onClick={toggleCamera} style={{ width:"100%", padding:"9px", background: cameraOn ? "rgba(34,197,94,0.15)" : "rgba(255,255,255,0.05)", border: cameraOn ? "1px solid rgba(34,197,94,0.4)" : "1px solid rgba(255,255,255,0.1)", borderRadius:50, color: cameraOn ? "#34d399" : "#94a3b8", fontSize:12, fontWeight:700, cursor:"pointer" }}>
              {cameraOn ? "📡 AI Camera ON — Klik Nonaktif" : "🎥 Aktifkan AI Camera"}
            </button>
          </div>
        </div>

        {/* RIGHT PANEL — Camera */}
        <div style={{ flex:1, display:"flex", flexDirection:"column", background:"#000", position:"relative", overflow:"hidden" }}>
          <div style={{ position:"absolute", top:0, left:0, right:0, zIndex:5, padding:"10px 14px", background:"linear-gradient(to bottom, rgba(0,0,0,0.75), transparent)", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
            <div style={{ background:"rgba(0,0,0,0.6)", border:"1px solid rgba(255,255,255,0.15)", borderRadius:50, padding:"4px 12px", fontSize:11, fontWeight:700, color:"#fff", display:"flex", alignItems:"center", gap:6, backdropFilter:"blur(8px)" }}>
              {cameraOn ? <><div style={{ width:7, height:7, borderRadius:"50%", background:"#22c55e", animation:"pulse 1.5s ease-in-out infinite" }} /> Kamu — Live</> : <><span>📷</span> Kamu</>}
            </div>
            {phase === "playing" && currentMove && (
              <div style={{ background:"rgba(99,102,241,0.85)", border:"1px solid rgba(167,139,250,0.4)", borderRadius:50, padding:"4px 14px", fontSize:12, fontWeight:900, color:"#fff", backdropFilter:"blur(8px)" }}>
                {reps} / {currentMove.reps} rep
              </div>
            )}
          </div>
          <div style={{ flex:1, position:"relative", display:"flex", alignItems:"center", justifyContent:"center" }}>
            <CameraView videoRef={cam.videoRef} canvasRef={cam.canvasRef} cameraReady={cam.cameraReady} cameraError={cam.cameraError} poseDetected={cam.poseDetected} poseReady={cam.poseReady} height={"100%"} style={{ borderRadius:0, height:"100%", border:"none" }} />
            {phase === "countdown" && (
              <div style={{ position:"absolute", inset:0, zIndex:10, background:"rgba(0,0,0,0.65)", backdropFilter:"blur(4px)", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:8 }}>
                <div style={{ fontSize:100, fontWeight:900, color:"#fff", textShadow:"0 0 40px rgba(99,102,241,0.8)", animation:"warmupPop 0.4s ease", lineHeight:1 }}>
                  {countdown > 0 ? countdown : "GO!"}
                </div>
                <div style={{ fontSize:15, fontWeight:700, color:"#94a3b8", textTransform:"uppercase", letterSpacing:2 }}>
                  {countdown > 0 ? "Bersiap..." : "Ayo Bergerak! 🔥"}
                </div>
              </div>
            )}
            {phase === "done" && (
              <div style={{ position:"absolute", inset:0, zIndex:10, background:"rgba(0,0,0,0.8)", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:10 }}>
                <div style={{ fontSize:52, animation:"warmupBounce 1s infinite" }}>⭐⭐⭐</div>
                <div style={{ fontSize:26, fontWeight:900, color:"#fff" }}>Pemanasan Selesai!</div>
                <div style={{ fontSize:13, color:"#94a3b8" }}>Kerja bagus! Energy penuh! 🎉</div>
              </div>
            )}
          </div>
          <div style={{ padding:"12px 16px", background:"linear-gradient(to top, rgba(0,0,0,0.9), rgba(0,0,0,0.5))", display:"flex", alignItems:"center", gap:10, flexShrink:0 }}>
            <div style={{ width:9, height:9, borderRadius:"50%", flexShrink:0, background: phase === "playing" ? "#22c55e" : phase === "done" ? "#a78bfa" : "#475569", boxShadow: phase === "playing" ? "0 0 8px #22c55e" : "none", transition:"all 0.3s" }} />
            <span style={{ fontSize:13, fontWeight:700, color:"#e2e8f0", flex:1 }}>{feedback}</span>
            {cameraOn && phase === "playing" && <span style={{ fontSize:10, fontWeight:700, color:"#34d399", flexShrink:0 }}>📡 AI aktif</span>}
          </div>
        </div>
      </div>
      <style>{`
        @keyframes warmupBounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
        @keyframes warmupPop    { 0%{transform:scale(1.5);opacity:0} 100%{transform:scale(1);opacity:1} }
        @keyframes warmupRing   { 0%{transform:scale(0.85);opacity:0.6} 100%{transform:scale(1.6);opacity:0} }
      `}</style>
    </div>
  );
}

// ─── MINI BATTLE SCREEN ───────────────────────────────────────────────────────
const RELAY_CHALLENGES = [
  { text:"Angkat kedua tangan 3x!", move:"raise_both", target:3, icon:"🙌" },
  { text:"Squat 3x!",               move:"squat",      target:3, icon:"🦵" },
  { text:"Lompat 3x!",              move:"jump",       target:3, icon:"⬆️" },
  { text:"Lari 5 langkah!",         move:"run",        target:5, icon:"🏃" },
  { text:"Tangan kanan 3x!",        move:"raise_right",target:3, icon:"✋" },
];

function MiniBattleScreen({ state, setState, addActivity, tryAwardBadge, triggerConfetti, showToast }) {
  const [phase, setPhase] = useState("idle");
  const [redScore, setRedScore] = useState(0);
  const [blueScore, setBlueScore] = useState(0);
  const [activeTeam, setActiveTeam] = useState("red");
  const [round, setRound] = useState(0);
  const [timeLeft, setTimeLeft] = useState(12);
  const [reps, setReps] = useState(0);
  const [currentChallenge, setCurrentChallenge] = useState(null);
  const [winner, setWinner] = useState(null);
  const timerRef = useRef(null);
  const MAX_ROUNDS = 5;

  const cam = useCameraTracking();
  const [cameraOn, setCameraOn] = useState(false);
  const activeTeamRef = useRef("red");
  activeTeamRef.current = activeTeam;
  const roundRef = useRef(0);
  roundRef.current = round;

  const toggleCamera = async () => {
    if (cameraOn) { cam.stopCamera(); setCameraOn(false); }
    else { setCameraOn(true); await cam.startCamera(); }
  };

  useEffect(() => {
    if (!cameraOn || phase !== "playing" || !currentChallenge) return;
    cam.startLoop(currentChallenge.move, () => doRep());
  }, [cameraOn, phase, currentChallenge?.move]);

  const startRelay = () => {
    setRedScore(0); setBlueScore(0); setActiveTeam("red"); setRound(0);
    setWinner(null); setPhase("playing"); AudioSystem.click();
    nextRound(0, 0, 0, "red");
  };

  const nextRound = (currentRound, currentRed, currentBlue, team) => {
    const nr = currentRound + 1;
    if (nr > MAX_ROUNDS) { endRelay(currentRed, currentBlue); return; }
    const c = RELAY_CHALLENGES[Math.floor(Math.random() * RELAY_CHALLENGES.length)];
    setRound(nr); setReps(0); setCurrentChallenge(c); setTimeLeft(12);
    clearInterval(timerRef.current);
    let tl = 12;
    timerRef.current = setInterval(() => {
      tl--;
      setTimeLeft(tl);
      if (tl <= 0) {
        clearInterval(timerRef.current);
        const nextTeam = team === "red" ? "blue" : "red";
        setActiveTeam(nextTeam);
        setTimeout(() => nextRound(nr, currentRed, currentBlue, nextTeam), 800);
      }
    }, 1000);
  };

  const doRep = useCallback(() => {
    if (phase !== "playing" || !currentChallenge) return;
    AudioSystem.energy();
    setReps(prev => {
      const newReps = prev + 1;
      if (newReps >= currentChallenge.target) {
        clearInterval(timerRef.current);
        const pts = Math.max(10, timeLeft * 5);
        if (activeTeamRef.current === "red") setRedScore(r => r + pts);
        else setBlueScore(b => b + pts);
        AudioSystem.rep(); triggerConfetti();
        const nextTeam = activeTeamRef.current === "red" ? "blue" : "red";
        setActiveTeam(nextTeam);
        setTimeout(() => nextRound(roundRef.current, redScore, blueScore, nextTeam), 1200);
      }
      return newReps;
    });
  }, [phase, currentChallenge, timeLeft, redScore, blueScore]);

  const endRelay = (finalRed, finalBlue) => {
    clearInterval(timerRef.current);
    setPhase("result");
    let w = finalRed > finalBlue ? "red" : finalBlue > finalRed ? "blue" : "draw";
    setWinner(w);
    AudioSystem.win(); triggerConfetti();
    addActivity({ type:"relay", name:"Estafet Tim", score: Math.max(finalRed, finalBlue), icon:"👥" });
    if (w !== "draw") tryAwardBadge("relay_win");
    if (w === "red") setState(prev => ({ ...prev, redTeamWins: (prev.redTeamWins||0)+1 }));
    else if (w === "blue") setState(prev => ({ ...prev, blueTeamWins: (prev.blueTeamWins||0)+1 }));
  };

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  const teamColor = activeTeam === "red" ? "#ef4444" : "#3b82f6";
  const teamName = activeTeam === "red" ? "🔴 Tim Merah" : "🔵 Tim Biru";

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", overflow:"hidden" }}>
      <div style={{ padding:"14px 24px", borderBottom:"1px solid rgba(255,255,255,0.08)", background:"rgba(15,23,42,0.7)", flexShrink:0, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div>
          <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Multiplayer</div>
          <div style={{ fontSize:22, fontWeight:900, color:"#fff" }}>Mini Battle ⚔️</div>
        </div>
        <button onClick={toggleCamera} style={{ background: cameraOn ? "linear-gradient(135deg,#22c55e,#16a34a)" : "rgba(255,255,255,0.08)", border: cameraOn ? "none" : "1px solid rgba(255,255,255,0.15)", borderRadius:12, padding:"8px 16px", color:"#fff", fontSize:12, fontWeight:800, cursor:"pointer", fontFamily:"'Nunito', sans-serif", display:"flex", alignItems:"center", gap:8 }}>
          📷 {cameraOn ? "AI Tracking ON" : "Aktifkan AI Camera"}
        </button>
      </div>
      <div style={{ display:"flex", flex:1, overflow:"hidden", minHeight:0 }}>
        <div style={{ flex:1, overflowY:"auto", padding:"20px 28px", borderRight: cameraOn ? "1px solid rgba(255,255,255,0.07)" : "none" }}>
          <div style={{ marginBottom:16, fontSize:12, color:"rgba(255,255,255,0.4)", fontWeight:600 }}>
            🔴 {state.redTeamWins||0} menang vs 🔵 {state.blueTeamWins||0} menang (all-time)
          </div>
          <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:24, padding:"20px", marginBottom:20, display:"grid", gridTemplateColumns:"1fr auto 1fr", gap:12, alignItems:"center" }}>
            <div style={{ textAlign:"center" }}>
              <div style={{ fontSize:40, marginBottom:4 }}>🔴</div>
              <div style={{ fontSize:32, fontWeight:900, color:"#ef4444" }}>{redScore}</div>
              <div style={{ fontSize:12, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>TIM MERAH</div>
            </div>
            <div style={{ fontSize:20, fontWeight:900, color:"rgba(255,255,255,0.3)" }}>VS</div>
            <div style={{ textAlign:"center" }}>
              <div style={{ fontSize:40, marginBottom:4 }}>🔵</div>
              <div style={{ fontSize:32, fontWeight:900, color:"#3b82f6" }}>{blueScore}</div>
              <div style={{ fontSize:12, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>TIM BIRU</div>
            </div>
          </div>
          {phase !== "idle" && (
            <div style={{ marginBottom:20 }}>
              <div style={{ display:"flex", height:12, borderRadius:50, overflow:"hidden", marginBottom:8 }}>
                <div style={{ background:"#ef4444", width:`${redScore/(redScore+blueScore+1)*100}%`, transition:"width 0.5s" }} />
                <div style={{ flex:1, background:"#3b82f6", transition:"width 0.5s" }} />
              </div>
              <div style={{ fontSize:12, color:"rgba(255,255,255,0.4)", textAlign:"center" }}>Ronde {round}/{MAX_ROUNDS}</div>
            </div>
          )}
          {phase === "idle" && (
            <div style={{ textAlign:"center", marginBottom:24 }}>
              <div style={{ fontSize:16, color:"rgba(255,255,255,0.6)", fontWeight:700, marginBottom:20 }}>
                Dua pemain bergiliran melakukan gerakan! Tim yang paling banyak poin menang!
              </div>
              <button onClick={startRelay} style={{ padding:"18px 40px", background:"linear-gradient(135deg,#ef4444,#b91c1c)", border:"none", borderRadius:20, color:"#fff", fontWeight:900, fontSize:20, cursor:"pointer", fontFamily:"'Nunito', sans-serif", boxShadow:"0 8px 32px rgba(239,68,68,0.4)", animation:"playPulse 2s ease-in-out infinite" }}>
                ⚔️ Mulai Pertandingan!
              </button>
            </div>
          )}
          {phase === "playing" && currentChallenge && (
            <div>
              <div style={{ background:`${teamColor}22`, border:`1px solid ${teamColor}44`, borderRadius:14, padding:"10px 20px", display:"inline-flex", alignItems:"center", gap:8, marginBottom:16 }}>
                <div style={{ width:12, height:12, borderRadius:"50%", background:teamColor, animation:"pulse 1s ease-in-out infinite" }} />
                <span style={{ fontSize:14, fontWeight:900, color:teamColor }}>{teamName} Giliran!</span>
              </div>
              <div style={{ background:`linear-gradient(135deg, ${teamColor}22, ${teamColor}11)`, border:`1px solid ${teamColor}44`, borderRadius:24, padding:"24px", marginBottom:16, textAlign:"center" }}>
                <div style={{ fontSize:48, marginBottom:8 }}>{currentChallenge.icon}</div>
                <div style={{ fontSize:22, fontWeight:900, color:"#fff", marginBottom:8 }}>{currentChallenge.text}</div>
                <div style={{ fontSize:16, color:teamColor, fontWeight:800 }}>{reps}/{currentChallenge.target} selesai</div>
                <div style={{ marginTop:12, background:"rgba(255,255,255,0.08)", borderRadius:50, height:10 }}>
                  <div style={{ height:"100%", width:`${(reps/currentChallenge.target)*100}%`, background:`linear-gradient(90deg, ${teamColor}, ${teamColor}aa)`, borderRadius:50, transition:"width 0.3s" }} />
                </div>
                <div style={{ marginTop:12, fontSize:24, fontWeight:900, color: timeLeft <= 5 ? "#ef4444" : "#fbbf24" }}>⏱️ {timeLeft}s</div>
                {cameraOn && <div style={{ marginTop:8, fontSize:12, color:"#34d399", fontWeight:700 }}>📡 AI mendeteksi gerakan otomatis</div>}
              </div>
              {!cameraOn && (
                <button onClick={doRep} style={{ width:"100%", padding:"22px", background:`linear-gradient(135deg, ${teamColor}, ${teamColor}bb)`, border:"none", borderRadius:20, color:"#fff", fontWeight:900, fontSize:22, cursor:"pointer", fontFamily:"'Nunito', sans-serif", boxShadow:`0 8px 32px ${teamColor}44`, animation:"playPulse 1.5s ease-in-out infinite" }}
                  onMouseDown={e => e.currentTarget.style.transform="scale(0.97)"}
                  onMouseUp={e => e.currentTarget.style.transform="scale(1)"}>
                  {currentChallenge.icon} TAP = 1 Gerakan!
                </button>
              )}
            </div>
          )}
          {phase === "result" && (
            <div style={{ textAlign:"center", padding:"32px 0" }}>
              <div style={{ fontSize:80, marginBottom:16 }}>{winner === "draw" ? "🤝" : "🏆"}</div>
              <div style={{ fontSize:28, fontWeight:900, color: winner === "red" ? "#ef4444" : winner === "blue" ? "#3b82f6" : "#fbbf24", marginBottom:8 }}>
                {winner === "red" ? "🔴 TIM MERAH MENANG!" : winner === "blue" ? "🔵 TIM BIRU MENANG!" : "🤝 SERI!"}
              </div>
              <div style={{ fontSize:16, color:"rgba(255,255,255,0.5)", marginBottom:24 }}>Merah: {redScore} pts | Biru: {blueScore} pts</div>
              <button onClick={() => { setPhase("idle"); setRedScore(0); setBlueScore(0); setRound(0); setWinner(null); }} style={{ padding:"14px 32px", background:"linear-gradient(135deg,#ef4444,#b91c1c)", border:"none", borderRadius:16, color:"#fff", fontWeight:900, fontSize:16, cursor:"pointer", fontFamily:"'Nunito', sans-serif" }}>
                🔄 Main Lagi
              </button>
            </div>
          )}
        </div>
        {cameraOn ? (
          <div style={{ width:"42%", flexShrink:0, display:"flex", flexDirection:"column", background:"#000", position:"relative" }}>
            <div style={{ position:"absolute", top:0, left:0, right:0, zIndex:5, padding:"10px 14px", background:"linear-gradient(to bottom, rgba(0,0,0,0.75), transparent)", display:"flex", alignItems:"center", gap:6 }}>
              <div style={{ background:"rgba(0,0,0,0.6)", border:"1px solid rgba(255,255,255,0.15)", borderRadius:50, padding:"4px 12px", fontSize:11, fontWeight:700, color:"#fff", display:"flex", alignItems:"center", gap:6, backdropFilter:"blur(8px)" }}>
                <div style={{ width:7, height:7, borderRadius:"50%", background:"#22c55e", animation:"pulse 1.5s ease-in-out infinite" }} />
                AI Tracking — Live
              </div>
            </div>
            <CameraView videoRef={cam.videoRef} canvasRef={cam.canvasRef} cameraReady={cam.cameraReady} cameraError={cam.cameraError} poseDetected={cam.poseDetected} poseReady={cam.poseReady} height={"100%"} style={{ borderRadius:0, height:"100%", border:"none" }} />
            <div style={{ position:"absolute", bottom:0, left:0, right:0, padding:"10px 14px", background:"linear-gradient(to top, rgba(0,0,0,0.85), transparent)", fontSize:11, fontWeight:700, color:"#34d399" }}>
              📡 AI mendeteksi gerakan otomatis
            </div>
          </div>
        ) : (
          <div style={{ position:"absolute", width:1, height:1, overflow:"hidden", opacity:0, pointerEvents:"none" }}>
            <CameraView videoRef={cam.videoRef} canvasRef={cam.canvasRef} cameraReady={cam.cameraReady} cameraError={cam.cameraError} poseDetected={cam.poseDetected} poseReady={cam.poseReady} height={1} />
          </div>
        )}
      </div>
    </div>
  );
}

// ─── BADGES SCREEN ────────────────────────────────────────────────────────────
function BadgesScreen({ state, triggerConfetti }) {
  const [selected, setSelected] = useState(null);
  const allBadges = BADGE_DEFS.map(b => ({ ...b, owned: state.badges[b.id]?.unlocked ?? false, date: state.badges[b.id]?.date }));
  const ownedCount = allBadges.filter(b => b.owned).length;

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24, display:"flex", alignItems:"flex-end", justifyContent:"space-between", gap:16, flexWrap:"wrap" }}>
        <div>
          <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Koleksi</div>
          <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Badge Collection 🏅</div>
          <div style={{ fontSize:13, color:"rgba(255,255,255,0.4)", marginTop:4 }}>🏅 {ownedCount} / {allBadges.length} Badge diraih</div>
        </div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:14, marginBottom:24 }}>
        {allBadges.map((b, i) => (
          <div key={b.id} className="badge-card" style={{ background: b.owned ? "linear-gradient(135deg, rgba(251,191,36,0.2), rgba(245,158,11,0.1))" : "rgba(255,255,255,0.04)", border: b.owned ? "1px solid rgba(251,191,36,0.4)" : "1px solid rgba(255,255,255,0.06)", borderRadius:20, padding:"20px 12px", display:"flex", flexDirection:"column", alignItems:"center", gap:8, textAlign:"center", cursor:"pointer", boxShadow: b.owned ? "0 4px 24px rgba(251,191,36,0.2)" : "none", animation:`cardEntrance 0.4s ${i * 0.06}s both ease`, transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)" }}
            onClick={() => { setSelected(b.id === selected ? null : b.id); if (b.owned) triggerConfetti(); }}>
            <div style={{ width:60, height:60, borderRadius:"50%", background: b.owned ? "linear-gradient(135deg, #fbbf24, #f59e0b)" : "rgba(255,255,255,0.06)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:28, boxShadow: b.owned ? "0 4px 20px rgba(251,191,36,0.4)" : "none", filter: b.owned ? "none" : "grayscale(1) opacity(0.3)", animation: b.owned ? "badgeShimmer 3s ease-in-out infinite" : "none" }}>
              {b.icon}
            </div>
            <div style={{ fontSize:11, fontWeight:800, color: b.owned ? "#fbbf24" : "#475569" }}>{b.name}</div>
            <div style={{ fontSize:9, fontWeight:700, color: b.owned ? "rgba(251,191,36,0.7)" : "#334155", textTransform:"uppercase", letterSpacing:0.5 }}>{b.owned ? "✓ DIRAIH" : "🔒 TERKUNCI"}</div>
          </div>
        ))}
      </div>
      {selected && (() => {
        const b = allBadges.find(b => b.id === selected);
        if (!b) return null;
        return (
          <div style={{ background:"rgba(255,255,255,0.05)", border:"1px solid rgba(255,255,255,0.1)", borderRadius:24, padding:"24px", marginBottom:20, animation:"expandIn 0.3s ease", display:"flex", gap:20, alignItems:"center" }}>
            <div style={{ fontSize:48 }}>{b.icon}</div>
            <div>
              <div style={{ fontSize:18, fontWeight:900, color:"#fff", marginBottom:4 }}>{b.name}</div>
              <div style={{ fontSize:13, color:"rgba(255,255,255,0.5)", marginBottom:8 }}>{b.desc}</div>
              {b.owned && b.date && <div style={{ fontSize:12, color:"#fbbf24", fontWeight:700 }}>✅ Didapat: {new Date(b.date).toLocaleDateString("id-ID")}</div>}
              {!b.owned && <div style={{ fontSize:12, color:"#94a3b8", fontWeight:700 }}>🔒 Belum didapat. Selesaikan tantangan untuk membuka!</div>}
            </div>
          </div>
        );
      })()}
      <div style={{ background:"linear-gradient(135deg, rgba(251,191,36,0.1), rgba(245,158,11,0.05))", border:"1px solid rgba(251,191,36,0.2)", borderRadius:24, padding:"20px 24px" }}>
        <div style={{ fontSize:14, fontWeight:800, color:"#fbbf24", marginBottom:8 }}>🏆 Progress Koleksi</div>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
          <span style={{ fontSize:13, color:"rgba(255,255,255,0.6)", fontWeight:600 }}>{ownedCount} / {allBadges.length} Badge</span>
          <span style={{ fontSize:13, color:"#fbbf24", fontWeight:800 }}>{Math.round(ownedCount/allBadges.length*100)}%</span>
        </div>
        <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:10 }}>
          <div style={{ height:"100%", width:`${ownedCount/allBadges.length*100}%`, background:"linear-gradient(90deg,#fbbf24,#f59e0b)", borderRadius:50 }} />
        </div>
      </div>
    </div>
  );
}

// ─── STATS SCREEN ─────────────────────────────────────────────────────────────
function StatsScreen({ state, addActivity }) {
  const days = ["Min","Sen","Sel","Rab","Kam","Jum","Sab"];
  const weekly = state.weeklyActivity || [0,0,0,0,0,0,0];
  const maxR = Math.max(...weekly, 1);

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Progres</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Statistik Kamu 📊</div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14, marginBottom:24 }}>
        {[
          { label:"Level Saat Ini", val:state.level, color:"#7c3aed", icon:"⚡" },
          { label:"Total XP",       val:state.exp,   color:"#2563eb", icon:"⭐" },
          { label:"Day Streak",     val:state.streak, color:"#ef4444", icon:"🔥" },
        ].map((s, i) => (
          <div key={i} style={{ background:`linear-gradient(135deg, ${s.color}22, ${s.color}11)`, border:`1px solid ${s.color}33`, borderRadius:20, padding:"18px", textAlign:"center", animation:`cardEntrance 0.4s ${i*0.1}s both` }}>
            <div style={{ fontSize:24, marginBottom:6 }}>{s.icon}</div>
            <div style={{ fontSize:28, fontWeight:900, color:s.color }}>{s.val}</div>
            <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{s.label}</div>
          </div>
        ))}
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:24 }}>
        {[
          { label:"Total Gerakan", val:state.totalMoves, color:"#22c55e", icon:"💪" },
          { label:"Kalori Dibakar", val:`${state.totalCalories} kkal`, color:"#f97316", icon:"🔥" },
          { label:"Pemanasan Selesai", val:state.warmupCount||0, color:"#fb923c", icon:"🤸" },
          { label:"Skor Challenge",  val:state.challengeHighScore||0, color:"#a855f7", icon:"⚡" },
        ].map((s, i) => (
          <div key={i} style={{ background:"rgba(255,255,255,0.04)", border:`1px solid ${s.color}33`, borderRadius:18, padding:"16px", display:"flex", alignItems:"center", gap:14 }}>
            <div style={{ fontSize:28 }}>{s.icon}</div>
            <div>
              <div style={{ fontSize:20, fontWeight:900, color:s.color }}>{s.val}</div>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{s.label}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:24, padding:"24px", marginBottom:16 }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:20 }}>Aktivitas 7 Hari Terakhir</div>
        <div style={{ display:"flex", alignItems:"flex-end", gap:10, height:120 }}>
          {weekly.map((val, i) => (
            <div key={i} style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
              <div style={{ fontSize:11, fontWeight:800, color:"#a78bfa" }}>{val}</div>
              <div style={{ width:"100%", borderRadius:"8px 8px 0 0", height:`${(val/maxR)*80}px`, background: i === new Date().getDay() ? "linear-gradient(180deg,#7c3aed,#2563eb)" : "linear-gradient(180deg,rgba(124,58,237,0.6),rgba(37,99,235,0.3))", boxShadow: i === new Date().getDay() ? "0 0 16px rgba(124,58,237,0.5)" : "none", transition:"height 1s ease", animation:`barGrow 0.6s ${i*0.1}s both ease` }} />
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{days[i]}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:24, padding:"24px", marginBottom:16 }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:16 }}>Riwayat Aktivitas</div>
        {state.activityLog && state.activityLog.length > 0 ? (
          state.activityLog.slice(0,8).map((a, i) => (
            <div key={i} style={{ display:"flex", alignItems:"center", gap:14, padding:"10px 0", borderBottom: i < Math.min(state.activityLog.length,8)-1 ? "1px solid rgba(255,255,255,0.06)" : "none" }}>
              <div style={{ width:42, height:42, borderRadius:14, flexShrink:0, background:`rgba(124,58,237,0.2)`, border:`1px solid rgba(124,58,237,0.3)`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18 }}>{a.icon || "🏃"}</div>
              <div style={{ flex:1 }}>
                <div style={{ fontSize:13, fontWeight:800, color:"#fff" }}>{a.name}</div>
                <div style={{ fontSize:11, color:"rgba(255,255,255,0.35)", fontWeight:600 }}>{a.date}</div>
              </div>
              <div style={{ fontSize:11, color:"#a78bfa", fontWeight:800 }}>{a.score} pts</div>
            </div>
          ))
        ) : (
          <div style={{ fontSize:13, color:"rgba(255,255,255,0.3)", textAlign:"center", fontWeight:600 }}>Belum ada aktivitas. Ayo mulai bermain!</div>
        )}
      </div>
      <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:24, padding:"24px" }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:16 }}>💬 AI Coach Feedback</div>
        {[
          { label:"Yang Bagus", val: state.totalMoves >= 10 ? "Konsistensi bagus! Kamu rajin berolahraga." : "Sudah mulai bergerak, pertahankan ya!", color:"#34d399" },
          { label:"Perlu Ditingkatkan", val: state.totalMoves < 5 ? "Coba tambah frekuensi bermain setiap hari." : state.streak < 3 ? "Coba main 3 hari berturut-turut!" : "Tingkatkan tantangan ke level berikutnya!", color:"#fb923c" },
        ].map((f,i) => (
          <div key={i} style={{ padding:"12px 0", borderBottom: i < 1 ? "1px solid rgba(255,255,255,0.06)" : "none" }}>
            <div style={{ fontSize:11, color:f.color, fontWeight:800, textTransform:"uppercase", letterSpacing:1, marginBottom:4 }}>{f.label}</div>
            <div style={{ fontSize:13, color:"rgba(255,255,255,0.7)", fontWeight:600 }}>{f.val}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── SETTINGS SCREEN ──────────────────────────────────────────────────────────
function SettingsScreen({ state, setState, resetState, showToast }) {
  const [name, setName] = useState(state.playerName || "");
  const [sound, setSound] = useState(state.settings?.sound !== false);
  const [music, setMusic] = useState(state.settings?.music !== false);
  const [tracking, setTracking] = useState(state.settings?.tracking !== false);

  const saveSettings = () => {
    setState(prev => ({ ...prev, playerName: name.trim() || prev.playerName, settings: { sound, music, tracking } }));
    AudioSystem.success();
    showToast("✅ Pengaturan disimpan!");
  };

  const handleReset = () => {
    if (window.confirm("Hapus semua data? Ini tidak bisa dibatalkan!")) {
      resetState();
      showToast("🗑️ Data dihapus. Mulai dari awal!");
    }
  };

  const Toggle = ({ value, onChange, label, icon }) => (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"14px 0", borderBottom:"1px solid rgba(255,255,255,0.06)" }}>
      <div style={{ display:"flex", alignItems:"center", gap:12 }}>
        <span style={{ fontSize:20 }}>{icon}</span>
        <span style={{ fontSize:14, fontWeight:700, color:"#e2e8f0" }}>{label}</span>
      </div>
      <div onClick={() => onChange(!value)} style={{ width:52, height:28, borderRadius:50, cursor:"pointer", background: value ? "linear-gradient(135deg,#7c3aed,#2563eb)" : "rgba(255,255,255,0.1)", position:"relative", transition:"all 0.3s ease", boxShadow: value ? "0 4px 12px rgba(124,58,237,0.4)" : "none" }}>
        <div style={{ width:20, height:20, borderRadius:"50%", background:"#fff", position:"absolute", top:4, left: value ? 28 : 4, transition:"left 0.3s ease", boxShadow:"0 2px 4px rgba(0,0,0,0.3)" }} />
      </div>
    </div>
  );

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Pengaturan</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Settings ⚙️</div>
      </div>
      <div style={{ background:"linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.08))", border:"1px solid rgba(34,197,94,0.3)", borderRadius:24, padding:"20px 24px", marginBottom:20 }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#34d399", marginBottom:8 }}>📡 AI Tracking Info</div>
        <div style={{ fontSize:13, color:"rgba(255,255,255,0.6)", fontWeight:600, lineHeight:1.6 }}>
          Gerak Ceria menggunakan <strong style={{ color:"#a78bfa" }}>MediaPipe PoseLandmarker</strong> (MoveNet-class model) untuk mendeteksi gerakan badanmu secara real-time melalui kamera. Aktifkan di layar Warmup, Challenge, atau Mission untuk pengalaman terbaik.
        </div>
        <div style={{ marginTop:12, display:"flex", gap:8, flexWrap:"wrap" }}>
          {["MediaPipe Tasks Vision","PoseLandmarker Lite","GestureDetector","EmotionDetector","Real-time Skeleton"].map(tag => (
            <span key={tag} style={{ fontSize:10, fontWeight:800, color:"#a78bfa", background:"rgba(124,58,237,0.2)", border:"1px solid rgba(124,58,237,0.3)", borderRadius:50, padding:"2px 10px" }}>{tag}</span>
          ))}
        </div>
      </div>
      <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:24, padding:"24px", marginBottom:20 }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:16 }}>👤 Profil Pemain</div>
        <div style={{ marginBottom:8 }}>
          <label style={{ fontSize:12, color:"#94a3b8", fontWeight:700, display:"block", marginBottom:8 }}>NAMA PEMAIN</label>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="Masukkan nama..." style={{ width:"100%", padding:"12px 16px", borderRadius:14, background:"rgba(255,255,255,0.08)", border:"1px solid rgba(255,255,255,0.12)", color:"#fff", fontSize:16, fontWeight:700, fontFamily:"'Nunito', sans-serif", outline:"none", boxSizing:"border-box" }} />
        </div>
        <div style={{ fontSize:12, color:"rgba(255,255,255,0.3)", fontWeight:600 }}>Level {state.level} · {state.exp} XP · Streak {state.streak} hari</div>
      </div>
      <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:24, padding:"24px", marginBottom:20 }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:4 }}>🔊 Suara & Musik</div>
        <Toggle value={sound} onChange={setSound} label="Efek Suara" icon="🔔" />
        <Toggle value={music} onChange={setMusic} label="Musik Latar" icon="🎵" />
        <Toggle value={tracking} onChange={setTracking} label="Body Tracking (AI Camera)" icon="📡" />
      </div>
      <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:24, padding:"24px", marginBottom:20, textAlign:"center" }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:12 }}>🤖 Maskot AI Coach</div>
        <div style={{ display:"flex", justifyContent:"center", animation:"mascotBob 3s ease-in-out infinite" }}>
          <MascotSVG size={100} mood="happy" glow />
        </div>
        <div style={{ fontSize:13, color:"rgba(255,255,255,0.5)", marginTop:12, fontWeight:600 }}>AI Coach siap menemanimu berolahraga! 💪</div>
      </div>
      <button onClick={saveSettings} style={{ width:"100%", padding:"18px", background:"linear-gradient(135deg,#7c3aed,#2563eb)", border:"none", borderRadius:20, color:"#fff", fontWeight:900, fontSize:18, cursor:"pointer", fontFamily:"'Nunito', sans-serif", boxShadow:"0 8px 32px rgba(124,58,237,0.4)", marginBottom:12 }}>
        💾 Simpan Pengaturan
      </button>
      <button onClick={handleReset} style={{ width:"100%", padding:"14px", background:"rgba(239,68,68,0.15)", border:"1px solid rgba(239,68,68,0.3)", borderRadius:16, color:"#ef4444", fontWeight:800, fontSize:14, cursor:"pointer", fontFamily:"'Nunito', sans-serif" }}>
        🗑️ Reset Semua Data
      </button>
    </div>
  );
}

// ─── NAV ICONS ────────────────────────────────────────────────────────────────
const HomeIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>);
const MapIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>);
const ZapIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>);
const FireIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>);
const SwordIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="14.5 17.5 3 6 3 3 6 3 17.5 14.5"/><line x1="13" y1="19" x2="19" y2="13"/><line x1="16" y1="16" x2="20" y2="20"/><line x1="19" y1="21" x2="21" y2="19"/></svg>);
const TrophyIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="8 19 12 23 16 19"/><line x1="12" y1="23" x2="12" y2="17"/><path d="M20.88 18.09A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.29"/></svg>);
const ChartIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>);
const SettingsIcon = () => (<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>);

// ─── CSS ──────────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 4px; }
  @keyframes logoFloat { 0%,100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-6px) rotate(3deg); } }
  @keyframes mascotBob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
  @keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.6; transform: scale(1.3); } }
  @keyframes playPulse { 0%,100% { box-shadow: 0 8px 40px rgba(34,197,94,0.5), 0 4px 12px rgba(37,99,235,0.3); } 50% { box-shadow: 0 8px 60px rgba(34,197,94,0.7), 0 0 0 12px rgba(34,197,94,0.1), 0 4px 12px rgba(37,99,235,0.4); } }
  @keyframes coachFadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes cardEntrance { from { opacity: 0; transform: translateY(20px) scale(0.95); } to { opacity: 1; transform: translateY(0) scale(1); } }
  @keyframes expandIn { from { opacity: 0; max-height: 0; } to { opacity: 1; max-height: 400px; } }
  @keyframes barGrow { from { transform: scaleY(0); transform-origin: bottom; } to { transform: scaleY(1); transform-origin: bottom; } }
  @keyframes badgeShimmer { 0%,100% { box-shadow: 0 4px 20px rgba(251,191,36,0.4); } 50% { box-shadow: 0 4px 30px rgba(251,191,36,0.7), 0 0 0 8px rgba(251,191,36,0.1); } }
  @keyframes floatUp { 0% { bottom: -20px; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 0.3; } 100% { bottom: 110vh; opacity: 0; transform: translateX(30px); } }
  .stat-card:hover { transform: translateY(-4px) scale(1.02); box-shadow: 0 8px 32px rgba(124,58,237,0.3) !important; }
  .feature-btn:hover { transform: translateY(-4px) scale(1.03) !important; }
  .feature-btn:active { transform: scale(0.97) !important; }
  .mission-node:hover:not(:disabled) { transform: scale(1.02); }
  .challenge-card:hover { transform: translateY(-2px); }
  .badge-card:hover { transform: translateY(-4px) scale(1.04) !important; }
`;