import { useState, useEffect, useRef } from "react";

// ─── SVG MASCOT ──────────────────────────────────────────────────────────────
const MascotSVG = ({ size = 120, mood = "happy", glow = false }) => (
  <svg width={size} height={size} viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg"
    style={{ filter: glow ? "drop-shadow(0 0 16px #a78bfa) drop-shadow(0 0 32px #7c3aed)" : "drop-shadow(0 4px 12px rgba(124,58,237,0.4))" }}>
    {/* Body */}
    <ellipse cx="60" cy="80" rx="28" ry="32" fill="url(#bodyGrad)" />
    {/* Head */}
    <circle cx="60" cy="45" r="30" fill="url(#headGrad)" />
    {/* Face visor */}
    <rect x="34" y="33" width="52" height="26" rx="10" fill="url(#visorGrad)" opacity="0.9" />
    {/* Eyes */}
    <ellipse cx="48" cy="46" rx="7" ry="8" fill="#fff" />
    <ellipse cx="72" cy="46" rx="7" ry="8" fill="#fff" />
    <circle cx={mood === "happy" ? 50 : 48} cy="46" r="4" fill="#1e1b4b" />
    <circle cx={mood === "happy" ? 74 : 72} cy="46" r="4" fill="#1e1b4b" />
    <circle cx={mood === "happy" ? 51 : 49} cy="44" r="1.5" fill="#fff" />
    <circle cx={mood === "happy" ? 75 : 73} cy="44" r="1.5" fill="#fff" />
    {/* Smile */}
    {mood === "happy"
      ? <path d="M50 58 Q60 66 70 58" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" fill="none" />
      : <path d="M50 60 Q60 55 70 60" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" fill="none" />}
    {/* Antenna */}
    <line x1="60" y1="15" x2="60" y2="0" stroke="#c4b5fd" strokeWidth="3" strokeLinecap="round" />
    <circle cx="60" cy="0" r="5" fill="#a78bfa">
      <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
    </circle>
    {/* Arms */}
    <rect x="18" y="70" width="12" height="28" rx="6" fill="url(#bodyGrad)" transform="rotate(-15 18 70)" />
    <rect x="90" y="70" width="12" height="28" rx="6" fill="url(#bodyGrad)" transform="rotate(15 90 70)" />
    {/* Legs */}
    <rect x="44" y="106" width="12" height="14" rx="6" fill="#6d28d9" />
    <rect x="64" y="106" width="12" height="14" rx="6" fill="#6d28d9" />
    {/* Shoes */}
    <ellipse cx="50" cy="119" rx="10" ry="5" fill="#4c1d95" />
    <ellipse cx="70" cy="119" rx="10" ry="5" fill="#4c1d95" />
    {/* Stars/sparkles */}
    <text x="96" y="28" fontSize="12" fill="#fbbf24">✦</text>
    <text x="10" y="35" fontSize="10" fill="#34d399">✦</text>
    <defs>
      <radialGradient id="headGrad" cx="40%" cy="35%">
        <stop stopColor="#c4b5fd" />
        <stop offset="1" stopColor="#7c3aed" />
      </radialGradient>
      <radialGradient id="bodyGrad" cx="40%" cy="30%">
        <stop stopColor="#a78bfa" />
        <stop offset="1" stopColor="#5b21b6" />
      </radialGradient>
      <linearGradient id="visorGrad" x1="0" y1="0" x2="0" y2="1">
        <stop stopColor="#e0f2fe" stopOpacity="0.9" />
        <stop offset="1" stopColor="#bae6fd" stopOpacity="0.6" />
      </linearGradient>
    </defs>
  </svg>
);

// ─── STAR SVG ─────────────────────────────────────────────────────────────────
const StarIcon = ({ filled, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={filled ? "#fbbf24" : "none"} stroke={filled ? "#fbbf24" : "#94a3b8"} strokeWidth="2">
    <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" />
  </svg>
);

// ─── CONFETTI ────────────────────────────────────────────────────────────────
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

// ─── FLOATING PARTICLES ───────────────────────────────────────────────────────
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

// ─── MISSION DATA ──────────────────────────────────────────────────────────────
const MISSIONS = [
  { id:1, name:"Latihan Dasar", icon:"🌱", desc:"Gerakan dasar olahraga", xp:50, stars:3, color:"#22c55e", glow:"rgba(34,197,94,0.4)", unlocked:true },
  { id:2, name:"Lompatan Ceria", icon:"⬆️", desc:"Tantangan melompat seru", xp:80, stars:2, color:"#3b82f6", glow:"rgba(59,130,246,0.4)", unlocked:true },
  { id:3, name:"Sprint Mini", icon:"🏃", desc:"Lari kencang di tempat", xp:100, stars:1, color:"#f97316", glow:"rgba(249,115,22,0.4)", unlocked:true },
  { id:4, name:"Combo Gerak", icon:"⚡", desc:"Kombinasi gerakan ajaib", xp:150, stars:0, color:"#a855f7", glow:"rgba(168,85,247,0.4)", unlocked:false },
  { id:5, name:"Master Olahraga", icon:"🏆", desc:"Tantangan akhir BOSS!", xp:300, stars:0, color:"#ef4444", glow:"rgba(239,68,68,0.4)", unlocked:false },
];

const DAILY_CHALLENGES = [
  { id:1, title:"Lompat 10x Berturut", icon:"⬆️", reward:"+30 XP", progress:60, color:"#f472b6", done:false },
  { id:2, title:"Squat 15x Hari Ini", icon:"🦵", reward:"+25 XP", progress:100, color:"#34d399", done:true },
  { id:3, title:"Streak 3 Hari", icon:"🔥", reward:"+50 XP", progress:33, color:"#fb923c", done:false },
];

const BADGES = [
  { icon:"🥇", name:"Juara", owned:true },
  { icon:"⚡", name:"Kilat", owned:true },
  { icon:"🔥", name:"Api", owned:true },
  { icon:"💎", name:"Diamond", owned:false },
  { icon:"🌟", name:"Bintang", owned:false },
  { icon:"🦁", name:"Singa", owned:false },
];

// ─── COACH MESSAGES ───────────────────────────────────────────────────────────
const COACH_MSGS = [
  "Ayo bergerak! Kamu bisa jadi juara! 💪",
  "Bagus sekali! Terus semangat! 🔥",
  "Gerakan kamu keren banget hari ini! ⚡",
  "Level up menanti! Kamu hampir sampai! 🌟",
  "Streak-mu makin panjang, luar biasa! 🏆",
];

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function GerakCeria() {
  const [screen, setScreen] = useState("home");
  const [confetti, setConfetti] = useState(false);
  const [coachIdx, setCoachIdx] = useState(0);
  const [hoveredNav, setHoveredNav] = useState(null);
  const [pressedPlay, setPressedPlay] = useState(false);
  const [xp] = useState(1240);
  const [level] = useState(7);
  const [streak] = useState(5);

  useEffect(() => {
    const t = setInterval(() => setCoachIdx(i => (i + 1) % COACH_MSGS.length), 4000);
    return () => clearInterval(t);
  }, []);

  const triggerConfetti = () => {
    setConfetti(true);
    setTimeout(() => setConfetti(false), 3000);
  };

  const nav = [
    { id:"home", icon: <HomeIcon />, label:"Home" },
    { id:"mission", icon: <MapIcon />, label:"Misi" },
    { id:"challenge", icon: <ZapIcon />, label:"Challenge" },
    { id:"badges", icon: <TrophyIcon />, label:"Badge" },
    { id:"stats", icon: <ChartIcon />, label:"Statistik" },
  ];

  return (
    <div style={{ fontFamily:"'Nunito', sans-serif", minHeight:"100vh", background:"linear-gradient(135deg, #0f0c29 0%, #1a0533 40%, #0d2137 100%)", display:"flex", position:"relative", overflow:"hidden" }}>
      <style>{CSS}</style>
      <Confetti active={confetti} />
      <FloatingParticles />

      {/* ── SIDEBAR ─────────────────────────────────────────────────────── */}
      <aside style={{
        width: 88, minHeight:"100vh", background:"rgba(255,255,255,0.04)",
        backdropFilter:"blur(20px)", borderRight:"1px solid rgba(255,255,255,0.08)",
        display:"flex", flexDirection:"column", alignItems:"center",
        padding:"20px 0", gap:8, position:"relative", zIndex:10, flexShrink:0,
      }}>
        {/* Logo */}
        <div style={{ marginBottom:20, textAlign:"center" }}>
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
            onClick={() => setScreen(n.id)}
            style={{
              width:64, height:64, borderRadius:18, border:"none",
              background: screen === n.id
                ? "linear-gradient(135deg,#7c3aed,#2563eb)"
                : hoveredNav === n.id
                  ? "rgba(124,58,237,0.2)"
                  : "transparent",
              cursor:"pointer", display:"flex", flexDirection:"column",
              alignItems:"center", justifyContent:"center", gap:4,
              transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
              transform: screen === n.id ? "scale(1.08)" : hoveredNav === n.id ? "scale(1.04)" : "scale(1)",
              boxShadow: screen === n.id ? "0 4px 20px rgba(124,58,237,0.5)" : "none",
              color: screen === n.id ? "#fff" : hoveredNav === n.id ? "#c4b5fd" : "#64748b",
              position:"relative",
            }}>
            <div style={{ fontSize:22 }}>{n.icon}</div>
            <div style={{ fontSize:9, fontWeight:800, letterSpacing:0.5 }}>{n.label}</div>
            {screen === n.id && (
              <div style={{
                position:"absolute", right:-2, top:"50%", transform:"translateY(-50%)",
                width:4, height:28, borderRadius:4, background:"linear-gradient(#7c3aed,#2563eb)",
              }} />
            )}
          </button>
        ))}

        {/* Bottom mascot preview */}
        <div style={{ marginTop:"auto", cursor:"pointer" }} onClick={() => setScreen("home")}>
          <MascotSVG size={60} mood="happy" />
        </div>
      </aside>

      {/* ── MAIN CONTENT ──────────────────────────────────────────────── */}
      <main style={{ flex:1, overflow:"auto", position:"relative", zIndex:5 }}>
        {screen === "home" && <HomeScreen level={level} xp={xp} streak={streak} coachMsg={COACH_MSGS[coachIdx]} onPlay={() => { setPressedPlay(true); setTimeout(() => { setPressedPlay(false); setScreen("mission"); }, 200); }} pressedPlay={pressedPlay} triggerConfetti={triggerConfetti} />}
        {screen === "mission" && <MissionScreen missions={MISSIONS} triggerConfetti={triggerConfetti} />}
        {screen === "challenge" && <ChallengeScreen challenges={DAILY_CHALLENGES} triggerConfetti={triggerConfetti} />}
        {screen === "badges" && <BadgesScreen badges={BADGES} triggerConfetti={triggerConfetti} />}
        {screen === "stats" && <StatsScreen xp={xp} level={level} streak={streak} />}
      </main>
    </div>
  );
}

// ─── HOME SCREEN ──────────────────────────────────────────────────────────────
function HomeScreen({ level, xp, streak, coachMsg, onPlay, pressedPlay, triggerConfetti }) {
  const xpToNext = 2000;
  const xpPercent = Math.round((xp / xpToNext) * 100);

  return (
    <div style={{ padding:"28px 32px", display:"flex", flexDirection:"column", gap:24, minHeight:"100vh" }}>
      {/* Header */}
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <div>
          <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Selamat Datang!</div>
          <div style={{ fontSize:28, fontWeight:900, color:"#fff", lineHeight:1.2 }}>Hai, Atlet Cilik! 👋</div>
        </div>
        <div style={{ display:"flex", gap:10 }}>
          {/* Streak badge */}
          <div style={{ background:"linear-gradient(135deg,#fb923c,#ef4444)", borderRadius:14, padding:"8px 16px", display:"flex", alignItems:"center", gap:6, boxShadow:"0 4px 16px rgba(239,68,68,0.4)" }}>
            <span style={{ fontSize:18 }}>🔥</span>
            <div>
              <div style={{ fontSize:18, fontWeight:900, color:"#fff", lineHeight:1 }}>{streak}</div>
              <div style={{ fontSize:10, color:"rgba(255,255,255,0.8)", fontWeight:700 }}>STREAK</div>
            </div>
          </div>
          {/* XP badge */}
          <div style={{ background:"linear-gradient(135deg,#7c3aed,#2563eb)", borderRadius:14, padding:"8px 16px", display:"flex", alignItems:"center", gap:6, boxShadow:"0 4px 16px rgba(124,58,237,0.4)" }}>
            <span style={{ fontSize:18 }}>⭐</span>
            <div>
              <div style={{ fontSize:18, fontWeight:900, color:"#fff", lineHeight:1 }}>{xp}</div>
              <div style={{ fontSize:10, color:"rgba(255,255,255,0.8)", fontWeight:700 }}>XP TOTAL</div>
            </div>
          </div>
        </div>
      </div>

      {/* HERO: Mascot + AI Coach */}
      <div style={{
        background:"linear-gradient(135deg, rgba(124,58,237,0.3) 0%, rgba(37,99,235,0.2) 50%, rgba(16,185,129,0.15) 100%)",
        border:"1px solid rgba(167,139,250,0.25)",
        borderRadius:28, padding:"28px 32px",
        display:"flex", alignItems:"center", gap:28,
        position:"relative", overflow:"hidden",
        backdropFilter:"blur(10px)",
      }}>
        {/* Glow orb */}
        <div style={{ position:"absolute", top:-60, right:-60, width:200, height:200, borderRadius:"50%", background:"radial-gradient(circle, rgba(124,58,237,0.3) 0%, transparent 70%)", pointerEvents:"none" }} />

        <div style={{ animation:"mascotBob 3s ease-in-out infinite", flexShrink:0 }}>
          <MascotSVG size={140} mood="happy" glow />
        </div>

        <div style={{ flex:1 }}>
          <div style={{ display:"inline-flex", alignItems:"center", gap:8, background:"rgba(52,211,153,0.15)", border:"1px solid rgba(52,211,153,0.3)", borderRadius:50, padding:"4px 14px", marginBottom:12 }}>
            <div style={{ width:8, height:8, borderRadius:"50%", background:"#34d399", animation:"pulse 1.5s ease-in-out infinite" }} />
            <span style={{ fontSize:11, color:"#34d399", fontWeight:800, letterSpacing:1 }}>AI COACH AKTIF</span>
          </div>

          <div style={{
            background:"rgba(255,255,255,0.06)", border:"1px solid rgba(255,255,255,0.1)",
            borderRadius:18, padding:"16px 20px", marginBottom:16,
            backdropFilter:"blur(8px)", minHeight:64,
            animation:"coachFadeIn 0.5s ease",
          }}>
            <p style={{ color:"#e2e8f0", fontSize:16, fontWeight:700, lineHeight:1.5, margin:0 }}>{coachMsg}</p>
          </div>

          {/* XP Progress */}
          <div style={{ marginBottom:8 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
              <span style={{ fontSize:12, color:"#94a3b8", fontWeight:700 }}>Level {level}</span>
              <span style={{ fontSize:12, color:"#a78bfa", fontWeight:700 }}>{xp} / {xpToNext} XP</span>
            </div>
            <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:10, overflow:"hidden" }}>
              <div style={{
                height:"100%", width:`${xpPercent}%`,
                background:"linear-gradient(90deg, #7c3aed, #2563eb, #34d399)",
                borderRadius:50,
                boxShadow:"0 0 10px rgba(124,58,237,0.6)",
                transition:"width 1s ease",
              }} />
            </div>
          </div>
          <div style={{ fontSize:12, color:"#64748b", fontWeight:600 }}>{xpToNext - xp} XP lagi menuju Level {level + 1} 🚀</div>
        </div>
      </div>

      {/* STATS CARDS */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:14 }}>
        {[
          { label:"Gerakan Hari Ini", val:"48", unit:"rep", color:"#22c55e", glow:"rgba(34,197,94,0.3)", icon:"💪" },
          { label:"Kalori Terbakar", val:"186", unit:"kkal", color:"#f97316", glow:"rgba(249,115,22,0.3)", icon:"🔥" },
          { label:"Misi Selesai", val:"3/5", unit:"misi", color:"#3b82f6", glow:"rgba(59,130,246,0.3)", icon:"🗺️" },
          { label:"Badge Diraih", val:"3", unit:"badge", color:"#a855f7", glow:"rgba(168,85,247,0.3)", icon:"🏅" },
        ].map((s, i) => (
          <div key={i} className="stat-card" style={{
            background:`linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03))`,
            border:`1px solid ${s.color}33`,
            borderRadius:20, padding:"18px 16px",
            backdropFilter:"blur(10px)",
            boxShadow:`0 4px 24px ${s.glow}`,
            animation:`cardEntrance 0.5s ${i * 0.1}s both ease`,
            cursor:"default",
          }}>
            <div style={{ fontSize:28, marginBottom:8 }}>{s.icon}</div>
            <div style={{ fontSize:26, fontWeight:900, color:s.color, lineHeight:1 }}>{s.val}</div>
            <div style={{ fontSize:11, color:"rgba(255,255,255,0.5)", fontWeight:700, textTransform:"uppercase", letterSpacing:0.5 }}>{s.unit}</div>
            <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:600, marginTop:4 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* QUICK FEATURES GRID */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14 }}>
        {[
          { label:"AI Warm Up", desc:"Pemanasan cerdas", icon:"🤸", color:"#fb923c", glow:"rgba(251,146,60,0.35)" },
          { label:"Mission Map", desc:"Petualangan misi", icon:"🗺️", color:"#3b82f6", glow:"rgba(59,130,246,0.35)" },
          { label:"Mini Battle", desc:"Lawan temanmu", icon:"⚔️", color:"#ef4444", glow:"rgba(239,68,68,0.35)" },
        ].map((f, i) => (
          <button key={i} className="feature-btn" style={{
            background:`linear-gradient(135deg, ${f.color}22, ${f.color}11)`,
            border:`1px solid ${f.color}44`,
            borderRadius:20, padding:"20px 16px",
            display:"flex", flexDirection:"column", alignItems:"flex-start", gap:8,
            cursor:"pointer", textAlign:"left",
            boxShadow:`0 4px 20px ${f.glow}`,
            transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
          }}>
            <div style={{ fontSize:32 }}>{f.icon}</div>
            <div style={{ fontSize:15, fontWeight:900, color:"#fff" }}>{f.label}</div>
            <div style={{ fontSize:12, color:"rgba(255,255,255,0.5)", fontWeight:600 }}>{f.desc}</div>
          </button>
        ))}
      </div>

      {/* PLAY BUTTON */}
      <button
        onClick={onPlay}
        style={{
          width:"100%", padding:"20px",
          background: pressedPlay
            ? "linear-gradient(135deg,#16a34a,#1d4ed8)"
            : "linear-gradient(135deg,#22c55e 0%,#16a34a 40%,#2563eb 100%)",
          border:"none", borderRadius:24, cursor:"pointer",
          fontSize:22, fontWeight:900, color:"#fff",
          letterSpacing:2, textTransform:"uppercase",
          boxShadow: pressedPlay ? "0 2px 10px rgba(34,197,94,0.3)" : "0 8px 40px rgba(34,197,94,0.5), 0 4px 12px rgba(37,99,235,0.3)",
          transform: pressedPlay ? "scale(0.97)" : "scale(1)",
          transition:"all 0.15s ease",
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
function MissionScreen({ missions, triggerConfetti }) {
  const [selected, setSelected] = useState(null);
  const [completed, setCompleted] = useState([]);

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Petualangan</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Mission Map 🗺️</div>
      </div>

      {/* PATH */}
      <div style={{ display:"flex", flexDirection:"column", gap:0, position:"relative" }}>
        {missions.map((m, i) => {
          const isLeft = i % 2 === 0;
          const isDone = completed.includes(m.id);
          return (
            <div key={m.id} style={{ display:"flex", flexDirection:"column", alignItems: isLeft ? "flex-start" : "flex-end", marginBottom:8 }}>
              {/* Connector line */}
              {i > 0 && (
                <div style={{
                  width:4, height:40, background:`linear-gradient(${missions[i-1].color}, ${m.color})`,
                  borderRadius:4, marginLeft: isLeft ? 68 : "auto",
                  marginRight: isLeft ? "auto" : 68,
                  opacity: m.unlocked ? 1 : 0.3,
                }} />
              )}

              <button
                disabled={!m.unlocked}
                onClick={() => { if (m.unlocked) { setSelected(selected === m.id ? null : m.id); } }}
                className="mission-node"
                style={{
                  display:"flex", alignItems:"center", gap:16,
                  background:`linear-gradient(135deg, ${m.color}22, ${m.color}11)`,
                  border: selected === m.id ? `2px solid ${m.color}` : `1px solid ${m.color}44`,
                  borderRadius:24, padding:"14px 20px",
                  cursor: m.unlocked ? "pointer" : "not-allowed",
                  opacity: m.unlocked ? 1 : 0.5,
                  boxShadow: selected === m.id ? `0 0 30px ${m.glow}` : `0 4px 16px ${m.glow}`,
                  width:"85%",
                  transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
                  transform: selected === m.id ? "scale(1.03)" : "scale(1)",
                }}>
                <div style={{
                  width:64, height:64, borderRadius:18, flexShrink:0,
                  background:`linear-gradient(135deg, ${m.color}, ${m.color}aa)`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontSize:28, boxShadow:`0 4px 16px ${m.glow}`,
                }}>
                  {m.unlocked ? m.icon : "🔒"}
                </div>
                <div style={{ flex:1, textAlign:"left" }}>
                  <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:4 }}>
                    <span style={{ fontSize:16, fontWeight:900, color:"#fff" }}>{m.name}</span>
                    {i === missions.length - 1 && <span style={{ fontSize:10, background:"#ef4444", color:"#fff", borderRadius:50, padding:"2px 8px", fontWeight:800 }}>BOSS</span>}
                  </div>
                  <div style={{ fontSize:12, color:"rgba(255,255,255,0.5)", fontWeight:600, marginBottom:6 }}>{m.desc}</div>
                  <div style={{ display:"flex", gap:4 }}>
                    {[0,1,2].map(s => <StarIcon key={s} filled={s < m.stars} size={16} />)}
                  </div>
                </div>
                <div style={{ textAlign:"right", flexShrink:0 }}>
                  <div style={{ fontSize:18, fontWeight:900, color:m.color }}>+{m.xp}</div>
                  <div style={{ fontSize:10, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>XP</div>
                </div>
              </button>

              {/* Expanded card */}
              {selected === m.id && (
                <div style={{
                  width:"85%", background:"rgba(255,255,255,0.05)",
                  border:`1px solid ${m.color}33`, borderRadius:20, padding:20, marginTop:8,
                  backdropFilter:"blur(10px)", animation:"expandIn 0.3s ease",
                }}>
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
                    <div style={{ fontSize:14, fontWeight:700, color:"rgba(255,255,255,0.7)" }}>Target: 10 gerakan</div>
                    <div style={{ fontSize:14, fontWeight:700, color:m.color }}>🎯 Reward: {m.xp} XP</div>
                  </div>
                  <button
                    onClick={() => { setCompleted(c => [...c, m.id]); triggerConfetti(); setSelected(null); }}
                    style={{
                      width:"100%", padding:"14px",
                      background:`linear-gradient(135deg, ${m.color}, ${m.color}bb)`,
                      border:"none", borderRadius:14, cursor:"pointer",
                      fontSize:15, fontWeight:900, color:"#fff",
                      boxShadow:`0 4px 20px ${m.glow}`,
                      fontFamily:"'Nunito', sans-serif",
                      transition:"transform 0.2s",
                    }}>
                    {isDone ? "✅ Selesai! Mainkan Lagi" : `▶ Mulai ${m.name}!`}
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

// ─── CHALLENGE SCREEN ─────────────────────────────────────────────────────────
function ChallengeScreen({ challenges, triggerConfetti }) {
  const [done, setDone] = useState(challenges.filter(c => c.done).map(c => c.id));

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Harian</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Daily Challenges ⚡</div>
      </div>

      {/* Timer banner */}
      <div style={{
        background:"linear-gradient(135deg, rgba(239,68,68,0.2), rgba(249,115,22,0.1))",
        border:"1px solid rgba(239,68,68,0.3)", borderRadius:20, padding:"14px 20px",
        display:"flex", alignItems:"center", gap:12, marginBottom:24,
      }}>
        <span style={{ fontSize:24 }}>⏰</span>
        <div>
          <div style={{ fontSize:14, fontWeight:900, color:"#fca5a5" }}>Reset dalam 08:24:15</div>
          <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:600 }}>Selesaikan sebelum waktu habis!</div>
        </div>
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
        {challenges.map((c, i) => {
          const isDone = done.includes(c.id);
          return (
            <div key={c.id} className="challenge-card" style={{
              background:`linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03))`,
              border:`1px solid ${isDone ? c.color + "66" : c.color + "22"}`,
              borderRadius:24, padding:"20px 24px",
              backdropFilter:"blur(10px)",
              boxShadow: isDone ? `0 4px 24px ${c.color}44` : "none",
              animation:`cardEntrance 0.4s ${i * 0.1}s both ease`,
              opacity: isDone ? 1 : 0.85,
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
                  <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                    <div style={{ background:`${c.color}22`, border:`1px solid ${c.color}44`, borderRadius:50, padding:"2px 10px" }}>
                      <span style={{ fontSize:11, color:c.color, fontWeight:800 }}>{c.reward}</span>
                    </div>
                  </div>
                </div>
                {!isDone && (
                  <button
                    onClick={() => { setDone(d => [...d, c.id]); triggerConfetti(); }}
                    style={{
                      padding:"8px 16px", borderRadius:12, border:"none",
                      background:`linear-gradient(135deg, ${c.color}, ${c.color}aa)`,
                      color:"#fff", fontWeight:800, fontSize:12, cursor:"pointer",
                      fontFamily:"'Nunito', sans-serif",
                      boxShadow:`0 4px 12px ${c.color}44`,
                    }}>KLAIM</button>
                )}
              </div>

              {/* Progress bar */}
              <div>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                  <span style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>Progress</span>
                  <span style={{ fontSize:11, color:c.color, fontWeight:800 }}>{isDone ? 100 : c.progress}%</span>
                </div>
                <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:8, overflow:"hidden" }}>
                  <div style={{
                    height:"100%", width:`${isDone ? 100 : c.progress}%`,
                    background:`linear-gradient(90deg, ${c.color}, ${c.color}aa)`,
                    borderRadius:50, transition:"width 1s ease",
                    boxShadow:`0 0 8px ${c.color}66`,
                  }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Weekly challenge */}
      <div style={{
        marginTop:24, background:"linear-gradient(135deg, rgba(124,58,237,0.25), rgba(37,99,235,0.15))",
        border:"1px solid rgba(167,139,250,0.3)", borderRadius:24, padding:"24px",
        position:"relative", overflow:"hidden",
      }}>
        <div style={{ position:"absolute", top:-30, right:-30, width:120, height:120, borderRadius:"50%", background:"radial-gradient(rgba(124,58,237,0.3), transparent)" }} />
        <div style={{ fontSize:13, color:"#a78bfa", fontWeight:800, letterSpacing:1, textTransform:"uppercase", marginBottom:8 }}>🌟 WEEKLY BOSS</div>
        <div style={{ fontSize:20, fontWeight:900, color:"#fff", marginBottom:8 }}>100 Gerakan Minggu Ini</div>
        <div style={{ fontSize:13, color:"rgba(255,255,255,0.5)", fontWeight:600, marginBottom:16 }}>Selesaikan untuk dapat badge eksklusif Diamond!</div>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
          <span style={{ fontSize:12, color:"rgba(255,255,255,0.5)", fontWeight:700 }}>48 / 100</span>
          <span style={{ fontSize:12, color:"#a78bfa", fontWeight:800 }}>48%</span>
        </div>
        <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:12, overflow:"hidden" }}>
          <div style={{
            height:"100%", width:"48%",
            background:"linear-gradient(90deg, #7c3aed, #2563eb)",
            borderRadius:50, boxShadow:"0 0 12px rgba(124,58,237,0.6)",
          }} />
        </div>
      </div>
    </div>
  );
}

// ─── BADGES SCREEN ────────────────────────────────────────────────────────────
function BadgesScreen({ badges, triggerConfetti }) {
  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Koleksi</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Badge Collection 🏅</div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:16 }}>
        {badges.map((b, i) => (
          <div key={i} className="badge-card" style={{
            background: b.owned
              ? "linear-gradient(135deg, rgba(251,191,36,0.2), rgba(245,158,11,0.1))"
              : "rgba(255,255,255,0.04)",
            border: b.owned ? "1px solid rgba(251,191,36,0.4)" : "1px solid rgba(255,255,255,0.06)",
            borderRadius:24, padding:"24px 16px",
            display:"flex", flexDirection:"column", alignItems:"center", gap:10,
            textAlign:"center", cursor: b.owned ? "pointer" : "default",
            boxShadow: b.owned ? "0 4px 24px rgba(251,191,36,0.2)" : "none",
            animation:`cardEntrance 0.4s ${i * 0.08}s both ease`,
            transition:"all 0.25s cubic-bezier(0.34,1.56,0.64,1)",
          }}
          onClick={() => b.owned && triggerConfetti()}>
            <div style={{
              width:72, height:72, borderRadius:"50%",
              background: b.owned
                ? "linear-gradient(135deg, #fbbf24, #f59e0b)"
                : "rgba(255,255,255,0.06)",
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:32,
              boxShadow: b.owned ? "0 4px 20px rgba(251,191,36,0.4)" : "none",
              filter: b.owned ? "none" : "grayscale(1) opacity(0.3)",
              animation: b.owned ? "badgeShimmer 3s ease-in-out infinite" : "none",
            }}>
              {b.icon}
            </div>
            <div style={{ fontSize:13, fontWeight:800, color: b.owned ? "#fbbf24" : "#475569" }}>{b.name}</div>
            <div style={{ fontSize:10, fontWeight:700, color: b.owned ? "rgba(251,191,36,0.7)" : "#334155", textTransform:"uppercase", letterSpacing:0.5 }}>
              {b.owned ? "✓ DIRAIH" : "🔒 TERKUNCI"}
            </div>
          </div>
        ))}
      </div>

      {/* Trophy shelf */}
      <div style={{
        marginTop:24, background:"linear-gradient(135deg, rgba(251,191,36,0.1), rgba(245,158,11,0.05))",
        border:"1px solid rgba(251,191,36,0.2)", borderRadius:24, padding:"20px 24px",
      }}>
        <div style={{ fontSize:14, fontWeight:800, color:"#fbbf24", marginBottom:8 }}>🏆 Progress Koleksi</div>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
          <span style={{ fontSize:13, color:"rgba(255,255,255,0.6)", fontWeight:600 }}>{badges.filter(b=>b.owned).length} / {badges.length} Badge</span>
          <span style={{ fontSize:13, color:"#fbbf24", fontWeight:800 }}>{Math.round(badges.filter(b=>b.owned).length/badges.length*100)}%</span>
        </div>
        <div style={{ background:"rgba(255,255,255,0.08)", borderRadius:50, height:10 }}>
          <div style={{ height:"100%", width:`${badges.filter(b=>b.owned).length/badges.length*100}%`, background:"linear-gradient(90deg,#fbbf24,#f59e0b)", borderRadius:50 }} />
        </div>
      </div>
    </div>
  );
}

// ─── STATS SCREEN ─────────────────────────────────────────────────────────────
function StatsScreen({ xp, level, streak }) {
  const days = ["S","S","R","K","J","S","M"];
  const reps = [12, 35, 28, 48, 55, 42, 38];
  const maxR = Math.max(...reps);

  return (
    <div style={{ padding:"28px 32px" }}>
      <div style={{ marginBottom:24 }}>
        <div style={{ fontSize:13, color:"#94a3b8", fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Progres</div>
        <div style={{ fontSize:28, fontWeight:900, color:"#fff" }}>Statistik Kamu 📊</div>
      </div>

      {/* Summary cards */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14, marginBottom:24 }}>
        {[
          { label:"Level Saat Ini", val:level, color:"#7c3aed", icon:"⚡" },
          { label:"Total XP", val:xp, color:"#2563eb", icon:"⭐" },
          { label:"Day Streak", val:streak, color:"#ef4444", icon:"🔥" },
        ].map((s, i) => (
          <div key={i} style={{
            background:`linear-gradient(135deg, ${s.color}22, ${s.color}11)`,
            border:`1px solid ${s.color}33`, borderRadius:20, padding:"18px",
            textAlign:"center", animation:`cardEntrance 0.4s ${i*0.1}s both`,
          }}>
            <div style={{ fontSize:24, marginBottom:6 }}>{s.icon}</div>
            <div style={{ fontSize:28, fontWeight:900, color:s.color }}>{s.val}</div>
            <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Bar chart */}
      <div style={{
        background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)",
        borderRadius:24, padding:"24px", marginBottom:16,
      }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:20 }}>Aktivitas 7 Hari Terakhir</div>
        <div style={{ display:"flex", alignItems:"flex-end", gap:10, height:120 }}>
          {days.map((d, i) => (
            <div key={i} style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
              <div style={{ fontSize:11, fontWeight:800, color:"#a78bfa" }}>{reps[i]}</div>
              <div style={{
                width:"100%", borderRadius:"8px 8px 0 0",
                height:`${(reps[i]/maxR)*80}px`,
                background: i === 4
                  ? "linear-gradient(180deg,#7c3aed,#2563eb)"
                  : "linear-gradient(180deg,rgba(124,58,237,0.6),rgba(37,99,235,0.3))",
                boxShadow: i === 4 ? "0 0 16px rgba(124,58,237,0.5)" : "none",
                transition:"height 1s ease",
                animation:`barGrow 0.6s ${i*0.1}s both ease`,
              }} />
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.4)", fontWeight:700 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Achievements */}
      <div style={{
        background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)",
        borderRadius:24, padding:"24px",
      }}>
        <div style={{ fontSize:14, fontWeight:900, color:"#fff", marginBottom:16 }}>Pencapaian Terbaru</div>
        {[
          { label:"Pertama Kali 50 Rep", time:"2 jam lalu", icon:"🎯", color:"#34d399" },
          { label:"Streak 5 Hari!", time:"Hari ini", icon:"🔥", color:"#fb923c" },
          { label:"Level 7 Tercapai", time:"Kemarin", icon:"⚡", color:"#a78bfa" },
        ].map((a, i) => (
          <div key={i} style={{
            display:"flex", alignItems:"center", gap:14, padding:"10px 0",
            borderBottom: i < 2 ? "1px solid rgba(255,255,255,0.06)" : "none",
          }}>
            <div style={{
              width:42, height:42, borderRadius:14, flexShrink:0,
              background:`${a.color}22`, border:`1px solid ${a.color}33`,
              display:"flex", alignItems:"center", justifyContent:"center", fontSize:18,
            }}>{a.icon}</div>
            <div style={{ flex:1 }}>
              <div style={{ fontSize:13, fontWeight:800, color:"#fff" }}>{a.label}</div>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.35)", fontWeight:600 }}>{a.time}</div>
            </div>
            <div style={{ fontSize:11, color:a.color, fontWeight:800 }}>+XP</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── NAV ICONS (inline SVG) ───────────────────────────────────────────────────
const HomeIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
  </svg>
);
const MapIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>
    <line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/>
  </svg>
);
const ZapIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);
const TrophyIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="8 19 12 23 16 19"/><line x1="12" y1="23" x2="12" y2="17"/>
    <path d="M20.88 18.09A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.29"/>
  </svg>
);
const ChartIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
    <line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>
  </svg>
);

// ─── CSS ──────────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 4px; }

  @keyframes logoFloat {
    0%,100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-6px) rotate(3deg); }
  }
  @keyframes mascotBob {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
  }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.3); }
  }
  @keyframes playPulse {
    0%,100% { box-shadow: 0 8px 40px rgba(34,197,94,0.5), 0 4px 12px rgba(37,99,235,0.3); }
    50% { box-shadow: 0 8px 60px rgba(34,197,94,0.7), 0 0 0 12px rgba(34,197,94,0.1), 0 4px 12px rgba(37,99,235,0.4); }
  }
  @keyframes coachFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes cardEntrance {
    from { opacity: 0; transform: translateY(20px) scale(0.95); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }
  @keyframes expandIn {
    from { opacity: 0; max-height: 0; }
    to { opacity: 1; max-height: 200px; }
  }
  @keyframes barGrow {
    from { transform: scaleY(0); transform-origin: bottom; }
    to { transform: scaleY(1); transform-origin: bottom; }
  }
  @keyframes badgeShimmer {
    0%,100% { box-shadow: 0 4px 20px rgba(251,191,36,0.4); }
    50% { box-shadow: 0 4px 30px rgba(251,191,36,0.7), 0 0 0 8px rgba(251,191,36,0.1); }
  }
  @keyframes floatUp {
    0% { bottom: -20px; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 0.3; }
    100% { bottom: 110vh; opacity: 0; transform: translateX(30px); }
  }

  .stat-card:hover { transform: translateY(-4px) scale(1.02); box-shadow: 0 8px 32px rgba(124,58,237,0.3) !important; }
  .feature-btn:hover { transform: translateY(-4px) scale(1.03) !important; }
  .feature-btn:active { transform: scale(0.97) !important; }
  .mission-node:hover:not(:disabled) { transform: scale(1.02); }
  .challenge-card:hover { transform: translateY(-2px); }
  .badge-card:hover { transform: translateY(-4px) scale(1.04) !important; }
`;
