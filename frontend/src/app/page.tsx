"use client";

import { useEffect, useRef, useState } from "react";
import QueueLengthChart, {
  pushQueueSample,
  type QueueSample,
} from "./QueueLengthChart";

const HTTP_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000/ws";

type FrameMsg = {
  type: string;
  frame_idx?: number;
  queue_count?: number;
  eta_sec?: number | null;
  avg_service_sec?: number | null;
  ewma_service_sec?: number | null;
  throughput_per_min?: number;
  serviced_count?: number;
  fps?: number;
  elapsed?: number;
  count_history?: number[];
};

export default function Home() {
  const [showSplash, setShowSplash] = useState(true);
  const [videos, setVideos] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [inputFile, setInputFile] = useState("");
  const [wsStatus, setWsStatus] = useState<"idle" | "connecting" | "open" | "closed" | "error">("idle");
  const [isRunning, setIsRunning] = useState(false);

  const [queue, setQueue] = useState<number | null>(null);

  const [eta, setEta] = useState<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  const [avgSvc, setAvgSvc] = useState<number | null>(null);
  const [throughput, setThroughput] = useState<number | null>(null);
  const [serviced, setServiced] = useState<number | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [queueHistory, setQueueHistory] = useState<QueueSample[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const stableEtaUpdatedAtRef = useRef(0);
  const [stableEta, setStableEta] = useState<number | null>(null);
  const [stableEtaUpdatedAt, setStableEtaUpdatedAt] = useState<number | null>(null);
  const [demoBufferMinutes, setDemoBufferMinutes] = useState(2);
  const [classTime, setClassTime] = useState("");
  const [currentTime, setCurrentTime] = useState<Date | null>(null);

  useEffect(() => {
    const updateClock = () => setCurrentTime(new Date());
    updateClock();
    const interval = window.setInterval(updateClock, 1_000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    const t = setTimeout(() => setShowSplash(false), 1400);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    fetch(`${HTTP_URL}/videos`)
      .then((r) => r.json())
      .then((d) => {
        const list: string[] = d.videos || [];
        setVideos(list);
        if (list.length && !selected) {
          const pref = list.includes("test2.mp4") ? "test2.mp4" : list[0];
          setSelected(pref);
          setInputFile(pref);
        }
      })
      .catch(() => setVideos(["test2.mp4", "test1.mp4", "testing.mp4"]));
  }, []); // eslint-disable-line



  // ws connect
  useEffect(() => {
    if (showSplash) return;
    let ws: WebSocket;
    let closedByUs = false;
    const connect = () => {
      setWsStatus("connecting");
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setWsStatus("open");
      ws.onclose = () => {
        setWsStatus("closed");
        if (!closedByUs) setTimeout(connect, 2000);
      };
      ws.onerror = () => setWsStatus("error");
      ws.onmessage = (ev) => {
        try {
          const msg: FrameMsg = JSON.parse(ev.data);
          if (msg.type === "hello") return;
          if (msg.type === "started") {
            setIsRunning(true);
            setError(null);
            setLastUpdated(Date.now());
            setQueueHistory([]);
            return;
          }
          if (msg.type === "frame") {
            if (typeof msg.queue_count === "number") setQueue(msg.queue_count);
            if (typeof msg.eta_sec === "number" || msg.eta_sec === null) {
              const nextEta = msg.eta_sec ?? null;
              setEta(nextEta);
              const now = Date.now();
              if (nextEta === null) {
                setStableEta(null);
                setStableEtaUpdatedAt(null);
                stableEtaUpdatedAtRef.current = 0;
              } else if (now - stableEtaUpdatedAtRef.current >= 10_000) {
                // Freeze the ETA for 10 seconds so arrival guidance remains readable.
                setStableEta(nextEta);
                setStableEtaUpdatedAt(now);
                setDemoBufferMinutes(Math.random() < 0.5 ? 2 : 3);
                stableEtaUpdatedAtRef.current = now;
              }
            }
            if (typeof msg.fps === "number") setFps(msg.fps);
            if (typeof msg.avg_service_sec === "number" || msg.avg_service_sec === null) setAvgSvc(msg.avg_service_sec ?? null);
            if (typeof msg.throughput_per_min === "number") setThroughput(msg.throughput_per_min);
            if (typeof msg.serviced_count === "number") setServiced(msg.serviced_count);
            if (typeof msg.queue_count === "number") {
              const t = typeof msg.elapsed === "number" ? msg.elapsed : 0;
              setQueueHistory((prev) => pushQueueSample(prev, t, msg.queue_count as number));
            }
            setLastUpdated(Date.now());
            return;
          }
          if (msg.type === "done" || msg.type === "stopped") {
            setIsRunning(false);
            setLastUpdated(Date.now());
            return;
          }
          if (msg.type === "error") {
            setError((msg as unknown as { message: string }).message || "error");
            setIsRunning(false);
          }
        } catch { }
      };
    };
    connect();
    return () => {
      closedByUs = true;
      ws?.close();
    };
  }, [showSplash]);

  const start = () => {
    const f = (inputFile || selected || "").trim();
    if (!f) {
      setError("Enter a filename (e.g. test2.mp4)");
      return;
    }
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket not connected");
      return;
    }
    setError(null);
    wsRef.current.send(JSON.stringify({ action: "start", filename: f, stream_frames: false, jpeg_quality: 70, target_fps: 15 }));
  };

  const stop = () => {
    wsRef.current?.send(JSON.stringify({ action: "stop" }));
  };

  const updatedAgo = lastUpdated && currentTime ? `${Math.max(0, Math.floor((currentTime.getTime() - lastUpdated) / 1000))}s ago` : "—";
  const etaMinutes = eta === null ? "—" : String(Math.ceil(eta / 60) + demoBufferMinutes);
  const targetClassTime = classTime && currentTime
    ? (() => {
        const [hours, minutes] = classTime.split(":").map(Number);
        const target = new Date(currentTime);
        target.setHours(hours, minutes, 0, 0);
        if (target.getTime() < currentTime.getTime()) target.setDate(target.getDate() + 1);
        return target;
      })()
    : null;
  const plannedEtaSeconds = stableEta !== null ? stableEta + demoBufferMinutes * 60 : null;
  const liftArrivalTime = targetClassTime && plannedEtaSeconds !== null
    ? new Date(targetClassTime.getTime() - plannedEtaSeconds * 1000)
    : null;
  const formatClockTime = (value: Date) => value.toLocaleTimeString("en-IN", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  if (showSplash) {
    return (
      <>
        <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');`}</style>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background:
              "radial-gradient(circle at 18% 20%, rgba(255,180,84,0.10), transparent 40%), radial-gradient(circle at 82% 75%, rgba(61,220,151,0.10), transparent 42%), #05070b",
            fontFamily: "'Manrope', sans-serif",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ display: "flex", gap: 10, marginBottom: 22, justifyContent: "center", alignItems: "center" }}>
              <span style={{ width: 9, height: 9, borderRadius: 999, background: "#9aa0b4" }} />
              <span style={{ width: 9, height: 9, borderRadius: 999, background: "#9aa0b4" }} />
              <span
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 999,
                  border: "2px solid #ffb454",
                  animation: "pulse 1.8s ease-in-out infinite",
                }}
              />
              <span style={{ width: 9, height: 9, borderRadius: 999, background: "#9aa0b4" }} />
              <span style={{ width: 9, height: 9, borderRadius: 999, background: "#9aa0b4" }} />
            </div>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 32, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>
              queuesense
            </div>
            <div style={{ marginTop: 8, fontSize: 14, color: "#cfd3e0" }}>Know before you go.</div>
            <div style={{ marginTop: 28, display: "flex", justifyContent: "center", gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: 999, background: "#9aa0b4" }} />
              <span style={{ width: 6, height: 6, borderRadius: 999, background: "#ffb454" }} />
              <span style={{ width: 6, height: 6, borderRadius: 999, background: "#9aa0b4" }} />
            </div>
          </div>
        </div>
        <style>{`@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,180,84,0.35)}50%{box-shadow:0 0 0 10px rgba(255,180,84,0)}}`}</style>
      </>
    );
  }

  return (
    <>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');`}</style>
      <div
        style={{
          minHeight: "100vh",
          background:
            "radial-gradient(circle at 18% 18%, rgba(255,180,84,0.08), transparent 38%), radial-gradient(circle at 82% 72%, rgba(61,220,151,0.08), transparent 42%), #05070b",
          fontFamily: "'Manrope', sans-serif",
          color: "#fff",
        }}
      >
        {/* header */}
        <header
          style={{
            position: "sticky",
            top: 0,
            zIndex: 20,
            backdropFilter: "blur(12px)",
            background: "rgba(11,14,22,0.85)",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
          }}
        >
          <div style={{ maxWidth: 1200, margin: "0 auto", padding: "14px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 800, fontSize: 16, letterSpacing: "-0.03em" }}>queuesense</div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 12, color: "#9aa0b4", display: "none" }} className="sm:inline">
                {HTTP_URL.replace(/^https?:\/\//, "")}
              </span>
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 12,
                  fontWeight: 700,
                  padding: "6px 12px",
                  borderRadius: 999,
                  background: wsStatus === "open" ? "rgba(61,220,151,0.14)" : "rgba(255,107,107,0.14)",
                  color: wsStatus === "open" ? "#3ddc97" : "#ff6b6b",
                  border: `1px solid ${wsStatus === "open" ? "rgba(61,220,151,0.25)" : "rgba(255,107,107,0.25)"}`,
                }}
              >
                <span style={{ width: 7, height: 7, borderRadius: 999, background: wsStatus === "open" ? "#3ddc97" : "#ff6b6b", boxShadow: wsStatus === "open" ? "0 0 8px rgba(61,220,151,0.6)" : "none" }} />
                {wsStatus === "open" ? (isRunning ? "Live · Streaming" : "Ready") : wsStatus}
              </span>
            </div>
          </div>
        </header>

        <main style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 20px 40px" }}>
          {/* hero stats */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
            <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr" }} className="lg:grid-cols-3">
              {/* queue card */}
              <div
                style={{
                  gridColumn: "span 2 / span 2",
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 24,
                  padding: 24,
                }}
                className="lg:col-span-2"
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#cfd3e0", letterSpacing: "0.04em", textTransform: "uppercase" }}>Live Queue</div>
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      padding: "5px 10px",
                      borderRadius: 999,
                      background: isRunning ? "rgba(61,220,151,0.16)" : "rgba(255,255,255,0.06)",
                      color: isRunning ? "#3ddc97" : "#9aa0b4",
                    }}
                  >
                    {isRunning ? "●  Estimating" : "Idle"}
                  </span>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 12, color: "#9aa0b4", fontWeight: 600, marginBottom: 8 }}>People in queue</div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 56, fontWeight: 800, lineHeight: 1, color: "#ffb454", transition: "all 300ms ease" }}>
                      {queue === null ? "—" : queue}
                    </div>
                    <div style={{ marginTop: 8, fontSize: 11, color: "#9aa0b4" }}>Live estimate · updates instantly</div>
                  </div>
                  <div style={{ borderLeft: "1px solid rgba(255,255,255,0.08)", paddingLeft: 20 }}>
                    <div style={{ fontSize: 12, color: "#9aa0b4", fontWeight: 600, marginBottom: 8 }}>Est. wait</div>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 44, fontWeight: 800, lineHeight: 1 }}>{etaMinutes}</span>
                      <span style={{ fontSize: 14, color: "#cfd3e0", fontWeight: 700 }}>min</span>
                    </div>
                    {/* <div style={{ marginTop: 8, fontSize: 11, color: "#9aa0b4", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {eta === null ? "calculating..." : `${etaLabel} · avg ${avgSvc ?? "—"}s`}
                    </div> */}
                  </div>
                </div>

                <div
                  style={{
                    marginTop: 20,
                    padding: "15px 16px",
                    borderRadius: 16,
                    border: "1px solid rgba(255,180,84,0.26)",
                    background: "rgba(255,180,84,0.07)",
                  }}
                >
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                    <div>
                      <div style={{ fontSize: 12, color: "#ffcf91", fontWeight: 800, letterSpacing: "0.04em", textTransform: "uppercase" }}>Arrival planner</div>
                      <div style={{ marginTop: 4, fontSize: 12, color: "#cfd3e0" }}>Need to reach class by?</div>
                    </div>
                    <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#cfd3e0", fontWeight: 700 }}>
                      Class time
                      <input
                        type="time"
                        value={classTime}
                        onChange={(event) => setClassTime(event.target.value)}
                        aria-label="Class arrival time"
                        style={{
                          colorScheme: "dark",
                          background: "rgba(5,7,11,0.7)",
                          border: "1px solid rgba(255,255,255,0.16)",
                          borderRadius: 10,
                          color: "#fff",
                          fontFamily: "'JetBrains Mono', monospace",
                          fontSize: 13,
                          fontWeight: 700,
                          padding: "8px 10px",
                          outline: "none",
                        }}
                      />
                    </label>
                  </div>

                  {classTime && (
                    <div style={{ marginTop: 14, paddingTop: 13, borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                      {liftArrivalTime ? (
                        <>
                          <div style={{ fontSize: 13, color: "#cfd3e0" }}>To reach class on time, arrive at the lift by</div>
                          <div style={{ marginTop: 3, fontFamily: "'JetBrains Mono', monospace", fontSize: 24, fontWeight: 800, color: "#ffcf91" }}>
                            {formatClockTime(liftArrivalTime)}
                          </div>
                          <div style={{ marginTop: 5, fontSize: 11, color: "#9aa0b4" }}>
                            held stable for 10 seconds{stableEtaUpdatedAt ? ` · refreshed ${new Date(stableEtaUpdatedAt).toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit", hour12: true })}` : ""}.
                          </div>
                        </>
                      ) : (
                        <div style={{ fontSize: 12, color: "#9aa0b4" }}>Waiting for enough queue activity to establish a stable ETA.</div>
                      )}
                    </div>
                  )}
                </div>

                <div
                  style={{
                    marginTop: 18,
                    paddingTop: 14,
                    borderTop: "1px dashed rgba(255,255,255,0.14)",
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 12,
                    color: "#9aa0b4",
                  }}
                >
                  <span>Updated {updatedAgo}</span>
                  <span>{fps !== null ? `${fps.toFixed(1)} FPS` : "— FPS"}</span>
                </div>
              </div>

              {/* controls */}
              <div
                style={{
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 24,
                  padding: 20,
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14 }}>Run inference</div>

                <label style={{ fontSize: 11, color: "#9aa0b4", fontWeight: 600, marginBottom: 6 }}>Video file on server</label>
                <select
                  value={selected}
                  onChange={(e) => {
                    setSelected(e.target.value);
                    setInputFile(e.target.value);
                  }}
                  style={{
                    width: "100%",
                    background: "rgba(0,0,0,0.35)",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 12,
                    color: "#fff",
                    fontSize: 13,
                    padding: "10px 12px",
                    outline: "none",
                  }}
                >
                  {videos.length === 0 && <option value="">loading…</option>}
                  {videos.map((v) => (
                    <option key={v} value={v} style={{ color: "#000" }}>
                      {v}
                    </option>
                  ))}
                </select>

                <input
                  value={inputFile}
                  onChange={(e) => setInputFile(e.target.value)}
                  placeholder="or type filename e.g. test2.mp4"
                  style={{
                    marginTop: 10,
                    width: "100%",
                    background: "rgba(0,0,0,0.35)",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 12,
                    color: "#fff",
                    fontSize: 13,
                    padding: "10px 12px",
                    outline: "none",
                  }}
                />

                <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                  <button
                    onClick={start}
                    disabled={isRunning || wsStatus !== "open"}
                    style={{
                      flex: 1,
                      padding: "11px 14px",
                      borderRadius: 12,
                      border: "none",
                      background: isRunning ? "rgba(255,255,255,0.08)" : "linear-gradient(150deg,#ffb454,#ff9a3d)",
                      color: isRunning ? "#9aa0b4" : "#1a1405",
                      fontWeight: 800,
                      fontSize: 14,
                      cursor: isRunning || wsStatus !== "open" ? "not-allowed" : "pointer",
                      opacity: wsStatus !== "open" ? 0.5 : 1,
                    }}
                  >
                    {isRunning ? "Running…" : "Start"}
                  </button>
                  <button
                    onClick={stop}
                    disabled={!isRunning}
                    style={{
                      padding: "11px 16px",
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.14)",
                      background: "rgba(255,255,255,0.06)",
                      color: "#fff",
                      fontWeight: 700,
                      fontSize: 14,
                      cursor: !isRunning ? "not-allowed" : "pointer",
                      opacity: !isRunning ? 0.5 : 1,
                    }}
                  >
                    Stop
                  </button>
                </div>

                {error && <div style={{ marginTop: 10, fontSize: 12, color: "#ff6b6b", background: "rgba(255,107,107,0.08)", border: "1px solid rgba(255,107,107,0.18)", borderRadius: 10, padding: "8px 10px" }}>{error}</div>}

                <div style={{ marginTop: 12, fontSize: 11, color: "rgba(255,255,255,0.45)" }}>
                  Backend → {HTTP_URL} · no video stream to save bandwidth
                </div>
              </div>
            </div>
          </div>

          <QueueLengthChart samples={queueHistory} isRunning={isRunning} />

          {/* metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 16 }}>
            {[
              { label: "Throughput", value: throughput !== null ? `${throughput}` : "—", sub: "per minute" },
              { label: "Serviced", value: serviced !== null ? String(serviced) : "—", sub: "total count" },
              { label: "Avg service", value: avgSvc !== null ? `${avgSvc}s` : "—", sub: "per person" },
            ].map((m) => (
              <div
                key={m.label}
                style={{
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.10)",
                  borderRadius: 20,
                  padding: "16px 18px",
                }}
              >
                <div style={{ fontSize: 11, color: "#9aa0b4", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}>{m.label}</div>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 22, fontWeight: 800, marginTop: 8 }}>{m.value}</div>
                <div style={{ fontSize: 11, color: "#9aa0b4", marginTop: 4 }}>{m.sub}</div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 20, textAlign: "center", fontSize: 11, color: "rgba(255,255,255,0.35)" }}>QueueSense · EdgeNova</div>
        </main>
      </div>
      <style>{`@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,180,84,0.35)}50%{box-shadow:0 0 0 10px rgba(255,180,84,0)}}`}</style>
    </>
  );
}
