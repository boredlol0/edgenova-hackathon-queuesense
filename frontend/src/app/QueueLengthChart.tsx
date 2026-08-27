"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type QueueSample = { t: number; q: number };

function formatClock(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

type Props = {
  samples: QueueSample[];
  isRunning: boolean;
};

export default function QueueLengthChart({ samples, isRunning }: Props) {
  const empty = samples.length === 0;
  const lastQ = empty ? null : samples[samples.length - 1].q;
  const lastT = empty ? null : samples[samples.length - 1].t;
  const qPeak = empty ? 0 : samples.reduce((m, s) => (s.q > m ? s.q : m), 0);
  const xMax = Math.max(lastT ?? 0, 15);
  const yMax = Math.max(qPeak, 4);

  return (
    <section
      aria-label="Queue length over time"
      style={{
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 24,
        padding: "20px 20px 16px",
        marginTop: 16,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, marginBottom: 8 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#cfd3e0", letterSpacing: "0.04em", textTransform: "uppercase" }}>
            Queue length over time
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: "#9aa0b4" }}>
            Confirmed people in ROI vs elapsed time
          </div>
        </div>
        <div style={{ display: "flex", gap: 18, fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>
          <div>
            <span style={{ color: "#9aa0b4" }}>now </span>
            <span style={{ color: "#ffb454", fontWeight: 700 }}>{lastQ === null ? "—" : lastQ}</span>
          </div>
          <div>
            <span style={{ color: "#9aa0b4" }}>peak </span>
            <span style={{ color: "#fff", fontWeight: 700 }}>{empty ? "—" : qPeak}</span>
          </div>
          <div>
            <span style={{ color: "#9aa0b4" }}>t </span>
            <span style={{ color: "#fff", fontWeight: 700 }}>{lastT === null ? "—" : formatClock(lastT)}</span>
          </div>
        </div>
      </div>

      <div style={{ width: "100%", height: 260, position: "relative" }}>
        {empty ? (
          <div
            style={{
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#9aa0b4",
              fontSize: 13,
            }}
          >
            {isRunning ? "Waiting for first frame…" : "Start a video to plot queue length"}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={samples} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="queueFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ffb454" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#ffb454" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis
                dataKey="t"
                type="number"
                domain={[0, xMax]}
                ticks={xTicks(xMax)}
                tickFormatter={formatClock}
                tick={{ fill: "#9aa0b4", fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
                axisLine={{ stroke: "rgba(255,255,255,0.12)" }}
                tickLine={false}
              />
              <YAxis
                dataKey="q"
                allowDecimals={false}
                width={36}
                domain={[0, yMax]}
                tick={{ fill: "#9aa0b4", fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ stroke: "rgba(255,180,84,0.35)" }}
                contentStyle={{
                  background: "#0b0e16",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 12,
                  fontSize: 12,
                  color: "#fff",
                }}
                labelFormatter={(value) => formatClock(Number(value))}
                formatter={(value) => [String(value), "queue"]}
              />
              <Area
                type="monotone"
                dataKey="q"
                name="queue"
                stroke="#ffb454"
                strokeWidth={2.25}
                fill="url(#queueFill)"
                isAnimationActive={false}
                dot={false}
                activeDot={{ r: 5, fill: "#ffb454", stroke: "#0b0e16", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

const SAMPLE_DT = 0.4;
const MAX_POINTS = 900;

function xTicks(max: number): number[] {
  const span = Math.max(max, 1);
  return [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(span * f * 10) / 10);
}

function resampleFullRange(samples: QueueSample[], maxPoints: number): QueueSample[] {
  if (samples.length <= maxPoints) return samples;
  const last = samples.length - 1;
  const out: QueueSample[] = new Array(maxPoints);
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.round((i * last) / (maxPoints - 1));
    out[i] = samples[idx];
  }
  return out;
}

export function pushQueueSample(prev: QueueSample[], t: number, q: number, maxPoints = MAX_POINTS): QueueSample[] {
  const time = Number.isFinite(t) && t >= 0 ? t : (prev[prev.length - 1]?.t ?? 0);

  if (prev.length === 0) {
    return time > 0 ? [{ t: 0, q }, { t: time, q }] : [{ t: 0, q }];
  }

  const last = prev[prev.length - 1];

  if (time - last.t < SAMPLE_DT) {
    if (last.q === q) return prev;
    const next = prev.slice();
    next[next.length - 1] = { t: last.t, q };
    return next;
  }

  return resampleFullRange([...prev, { t: time, q }], maxPoints);
}
