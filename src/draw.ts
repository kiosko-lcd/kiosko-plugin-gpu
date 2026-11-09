/**
 * Painting. The normal state is darkness: figures in dim ink, bars in faint
 * ink. Amber and red appear only past the thresholds, so a GPU that runs
 * hot is the one thing on the strip that claims you.
 */
import type { Graphics, Localized, RenderContext, ThemeColors } from '@kiosko/sdk';
import { gib, gpuTempLevel, gpuTitle, mainTemp, tempLevel, vramLevel, vramRatio, watts, type Level, type Thresholds } from './gpu.ts';
import type { GpuDevice, GpuSnapshot, HistoryPoint } from './types.ts';

type Ctx = Pick<RenderContext<unknown, unknown>, 'box' | 'theme' | 'kit' | 't' | 'now'>;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export const TEXT = {
  temp: { en: 'temp', es: 'temp' },
  util: { en: 'load', es: 'carga' },
  vram: { en: 'vram', es: 'vram' },
  power: { en: 'power', es: 'potencia' },
  fan: { en: 'fan', es: 'vent.' },
  free: { en: 'free', es: 'libre' },
  other: { en: 'other', es: 'otros' },
  noProcs: { en: 'no processes', es: 'sin procesos' },
  noData: { en: 'no GPU data (needs the gpu/devices source)', es: 'sin datos de GPU (necesita la fuente gpu/devices)' },
  unavailable: { en: 'no GPU: {reason}', es: 'sin GPU: {reason}' },
  minutes: { en: 'last {n} min', es: 'últimos {n} min' },
  collecting: { en: 'collecting…', es: 'recogiendo…' },
} satisfies Record<string, Localized>;

const fill = (text: string, vars: Record<string, string | number>) => text.replace(/\{(\w+)\}/g, (_, k: string) => String(vars[k] ?? ''));

export function levelColor(l: Level, c: ThemeColors, normal = c.inkDim): string {
  return l === 'alert' ? c.alert : l === 'warn' ? c.warn : normal;
}

/** Prints why there's nothing to draw, faint; returns false when there is something. */
function drawEmpty(g: Graphics, ctx: Ctx, s: GpuSnapshot | undefined): s is undefined {
  const { box, kit, theme, t } = ctx;
  if (s?.available && s.gpus.length) return false;
  const size = Math.max(10, Math.round(clamp(box.h / 18, 10, 18)));
  const msg = !s ? t(TEXT.noData) : fill(t(TEXT.unavailable), { reason: s.reason ?? '?' });
  const lines = kit.wrapLines(g, msg, box.w, 4, { size, family: theme.fonts.mono });
  lines.forEach((l, i) => kit.drawTracked(g, l, box.x, box.y + size * 1.2 * (i + 1), { size, family: theme.fonts.mono, color: theme.colors.inkFaint }));
  return true;
}

/** Faint host label above the GPUs; returns the height it took. */
function drawLabel(g: Graphics, ctx: Ctx, s: GpuSnapshot, scale: number): number {
  if (!s.label) return 0;
  const size = Math.max(10, Math.round(12 * scale));
  ctx.kit.drawClipped(g, s.label, ctx.box.x, ctx.box.y + size, ctx.box.w, { size, family: ctx.theme.fonts.sansBold, tracking: 2 * scale, smallCaps: true, color: ctx.theme.colors.inkFaint });
  return size + 6 * scale;
}

/** Splits the box into one slot per GPU: columns when wide, rows when tall. */
function slots(x: number, y: number, w: number, h: number, n: number, minColW: number): { x: number; y: number; w: number; h: number }[] {
  const c = w >= h * 1.2 ? clamp(Math.floor(w / minColW), 1, n) : 1;
  const rows = Math.ceil(n / c);
  const gap = 14;
  const sw = (w - gap * (c - 1)) / c;
  const sh = (h - gap * (rows - 1)) / rows;
  return Array.from({ length: n }, (_, i) => ({ x: x + (i % c) * (sw + gap), y: y + Math.floor(i / c) * (sh + gap), w: sw, h: sh }));
}

function bar(g: Graphics, c: ThemeColors, x: number, y: number, w: number, h: number, ratio: number, color: string): void {
  g.fillStyle = c.bg;
  g.fillRect(x, y, w, h);
  g.fillStyle = color;
  g.fillRect(x, y, Math.max(ratio > 0 ? 2 : 0, Math.round(w * clamp(ratio, 0, 1))), h);
  g.strokeStyle = c.surfaceEdge;
  g.lineWidth = 1;
  g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

/* ---------- gpu/cards ---------- */

export function drawCards(g: Graphics, ctx: Ctx, s: GpuSnapshot | undefined, th: Thresholds): void {
  if (drawEmpty(g, ctx, s)) return;
  const { box } = ctx;
  const top = drawLabel(g, ctx, s!, clamp(box.h / 300, 0.6, 1.4));
  for (const [i, r] of slots(box.x, box.y + top, box.w, box.h - top, s!.gpus.length, 150).entries()) drawCard(g, ctx, s!.gpus[i]!, r, th);
}

function drawCard(g: Graphics, ctx: Ctx, d: GpuDevice, r: { x: number; y: number; w: number; h: number }, th: Thresholds): void {
  const { kit, theme } = ctx;
  const c = theme.colors;
  const f = theme.fonts;
  const s = clamp(Math.min(r.w / 280, r.h / 330), 0.5, 1.6);
  const tempL = gpuTempLevel(d, th);
  const label = { size: Math.max(9, Math.round(12 * s)), family: f.sansBold, tracking: 2.2 * s, smallCaps: true, color: c.inkFaint };
  let y = r.y + Math.round(16 * s);
  kit.drawClipped(g, gpuTitle(d), r.x, y, r.w, { size: Math.max(10, Math.round(16 * s)), family: f.sansBold, tracking: 1.5 * s, color: tempL === 'normal' ? c.inkDim : levelColor(tempL, c) });

  // Big figures: temperature and load.
  const big = Math.round(46 * s);
  y += big * 0.85 + 10 * s;
  if (y + 14 * s > r.y + r.h) return;
  const temp = mainTemp(d);
  const half = r.w / 2;
  kit.drawClipped(g, temp === undefined ? '–' : `${Math.round(temp)}°`, r.x, y, half - 4, { size: big, family: f.monoBold, color: levelColor(tempL, c), tracking: -0.5 * s });
  kit.drawClipped(g, d.util === undefined ? '–' : `${Math.round(d.util)}%`, r.x + half, y, half, { size: big, family: f.monoBold, color: c.inkDim, tracking: -0.5 * s });
  const extra = [d.hotspot !== undefined ? `hot ${Math.round(d.hotspot)}°` : '', d.memTemp !== undefined ? `mem ${Math.round(d.memTemp)}°` : ''].filter(Boolean).join(' ');
  kit.drawClipped(g, extra ? `${ctx.t(TEXT.temp)} · ${extra}` : ctx.t(TEXT.temp), r.x, y + 16 * s, half - 4, label);
  kit.drawClipped(g, ctx.t(TEXT.util), r.x + half, y + 16 * s, half, label);
  y += 16 * s;

  // Bars: VRAM, then power.
  const barH = Math.max(6, Math.round(12 * s));
  const slot = 30 * s + barH;
  const rows: [string, string, number, string][] = [];
  const vr = vramRatio(d);
  if (vr !== undefined) rows.push([ctx.t(TEXT.vram), `${gib(d.memUsed!)}/${gib(d.memTotal!)}`, vr, levelColor(vramLevel(d, th), c, c.inkFaint)]);
  if (d.power !== undefined) rows.push([ctx.t(TEXT.power), d.powerLimit ? `${watts(d.power)}/${watts(d.powerLimit)}` : watts(d.power), d.powerLimit ? d.power / d.powerLimit : 0, c.inkFaint]);
  for (const [name, value, ratio, color] of rows) {
    if (y + slot > r.y + r.h) return;
    y += 24 * s;
    const vw = kit.drawTracked(g, value, r.x + r.w, y, { size: Math.max(10, Math.round(15 * s)), family: f.mono, color: color === c.inkFaint ? c.inkDim : color, align: 'right' });
    kit.drawClipped(g, name, r.x, y, Math.max(0, r.w - vw - 6), label);
    bar(g, c, r.x, y + 6 * s, r.w, barH, ratio, color);
    y += 6 * s + barH;
  }

  // Footer: fan, clocks, throttling — only if there's room.
  const foot = [d.fan !== undefined ? `${ctx.t(TEXT.fan)} ${Math.round(d.fan)}%` : '', d.smClock ? `${Math.round(d.smClock)}MHz` : '', ...(d.throttle ?? [])].filter(Boolean);
  const fs = Math.max(9, Math.round(13 * s));
  if (foot.length && y + fs * 1.8 <= r.y + r.h) {
    const throttled = (d.throttle?.length ?? 0) > 0 && d.throttle!.some((x) => /thermal|slowdown|brake/.test(x));
    kit.drawClipped(g, foot.join(' · '), r.x, y + fs * 1.6, r.w, { size: fs, family: f.mono, color: throttled ? c.warn : c.inkFaint });
  }
}

/* ---------- gpu/vram ---------- */

export function drawVram(g: Graphics, ctx: Ctx, s: GpuSnapshot | undefined, th: Thresholds, maxProcs: number): void {
  if (drawEmpty(g, ctx, s)) return;
  const { box, kit, theme } = ctx;
  const c = theme.colors;
  const f = theme.fonts;
  const gpus = s!.gpus.filter((d) => d.memTotal);
  const top = drawLabel(g, ctx, s!, clamp(box.h / 300, 0.6, 1.4));
  for (const [i, r] of slots(box.x, box.y + top, box.w, box.h - top, gpus.length, 420).entries()) {
    const d = gpus[i]!;
    const sc = clamp(Math.min(r.h / 130, r.w / 420), 0.5, 1.5);
    const level = vramLevel(d, th);
    const used = d.memUsed ?? 0;
    const total = d.memTotal!;
    const titleY = r.y + Math.round(16 * sc);
    const vw = kit.drawTracked(g, `${gib(used)} / ${gib(total)}`, r.x + r.w, titleY, { size: Math.max(10, Math.round(17 * sc)), family: f.monoBold, color: levelColor(level, c), align: 'right' });
    kit.drawClipped(g, gpuTitle(d), r.x, titleY, Math.max(0, r.w - vw - 10), { size: Math.max(10, Math.round(15 * sc)), family: f.sansBold, tracking: 1.5 * sc, color: c.inkDim });
    const by = titleY + 8 * sc;
    const bh = Math.max(10, Math.min(Math.round(36 * sc), r.y + r.h - by - (r.h > 90 ? 22 * sc : 0)));
    if (by + bh > r.y + r.h + 1) continue;
    g.fillStyle = c.bg;
    g.fillRect(r.x, by, r.w, bh);
    // Process segments, biggest first; the rest of what's used is "other".
    const procs = (d.processes ?? []).filter((p) => p.mib > 0).sort((a, b) => b.mib - a.mib);
    const shown = procs.slice(0, maxProcs);
    const hidden = procs.slice(maxProcs).reduce((a, p) => a + p.mib, 0);
    const other = Math.max(0, used - shown.reduce((a, p) => a + p.mib, 0));
    const segs: [string, number][] = shown.map((p) => [p.name, p.mib]);
    if (other >= total * 0.01 || (hidden && other)) segs.push([procs.length ? ctx.t(TEXT.other) : '', other]);
    const lit = level === 'normal' ? [c.inkDim, c.inkFaint] : [levelColor(level, c), levelColor(level, c)];
    let x = r.x;
    const ts = Math.max(9, Math.round(Math.min(15 * sc, bh * 0.5)));
    segs.forEach(([name, mib], k) => {
      const w = Math.round((mib / total) * r.w);
      if (w <= 0) return;
      g.globalAlpha = level === 'normal' ? 1 : k % 2 ? 0.55 : 0.85;
      g.fillStyle = lit[k % 2]!;
      g.fillRect(x, by, Math.max(1, w - 1), bh);
      g.globalAlpha = 1;
      const txt = name ? `${name} ${gib(mib)}` : gib(mib);
      if (w > ts * 2.5) kit.drawClipped(g, txt, x + 5 * sc, by + bh / 2 + ts * 0.35, w - 10 * sc, { size: ts, family: f.monoBold, color: c.bg });
      x += w;
    });
    g.strokeStyle = c.surfaceEdge;
    g.lineWidth = 1;
    g.strokeRect(r.x + 0.5, by + 0.5, r.w - 1, bh - 1);
    // Legend line under the bar: the free space, and the processes when the bar can't hold their names.
    const ly = by + bh + 16 * sc;
    if (ly <= r.y + r.h) {
      const legend = procs.length ? procs.slice(0, maxProcs).map((p) => `${p.name} ${gib(p.mib)}`).join(' · ') : (d.processes ? ctx.t(TEXT.noProcs) : '');
      const freeTxt = `${gib(Math.max(0, total - used))} ${ctx.t(TEXT.free)}`;
      const fw = kit.drawTracked(g, freeTxt, r.x + r.w, ly, { size: Math.max(9, Math.round(13 * sc)), family: f.mono, color: c.inkFaint, align: 'right' });
      if (legend) kit.drawClipped(g, legend, r.x, ly, Math.max(0, r.w - fw - 12), { size: Math.max(9, Math.round(13 * sc)), family: f.mono, color: c.inkFaint });
    }
  }
}

/* ---------- gpu/history ---------- */

export function drawHistory(g: Graphics, ctx: Ctx, s: GpuSnapshot | undefined, th: Thresholds, minutes: number): void {
  if (drawEmpty(g, ctx, s)) return;
  const { box, kit, theme, now } = ctx;
  const c = theme.colors;
  const f = theme.fonts;
  const top = drawLabel(g, ctx, s!, clamp(box.h / 300, 0.6, 1.4));
  const windowMs = minutes * 60_000;
  for (const [i, r] of slots(box.x, box.y + top, box.w, box.h - top, s!.gpus.length, 300).entries()) {
    const d = s!.gpus[i]!;
    const sc = clamp(Math.min(r.h / 220, r.w / 360), 0.5, 1.5);
    const level = gpuTempLevel(d, th);
    const titleY = r.y + Math.round(15 * sc);
    const nowTxt = `${d.temp !== undefined ? `${Math.round(d.temp)}°` : '–'}  ${d.util !== undefined ? `${Math.round(d.util)}%` : '–'}`;
    const vw = kit.drawTracked(g, nowTxt, r.x + r.w, titleY, { size: Math.max(10, Math.round(17 * sc)), family: f.monoBold, color: levelColor(level, c), align: 'right' });
    kit.drawClipped(g, gpuTitle(d), r.x, titleY, Math.max(0, r.w - vw - 10), { size: Math.max(10, Math.round(15 * sc)), family: f.sansBold, tracking: 1.5 * sc, color: c.inkDim });
    const footH = r.h > 110 ? 18 * sc : 0;
    const cy = titleY + 10 * sc;
    const ch = r.y + r.h - cy - footH;
    if (ch < 16) continue;
    g.fillStyle = c.bg;
    g.fillRect(r.x, cy, r.w, ch);
    const pts = (s!.history[d.uuid] ?? []).filter((p) => p[0] >= now - windowMs);
    const px = (t: number) => r.x + ((t - (now - windowMs)) / windowMs) * r.w;
    // Temperature scale 20..max(100, crit+5) °C; the critical line, faint and dashed.
    const tMin = 20;
    const tMax = Math.max(100, th.tempCrit + 5);
    const ty = (v: number) => cy + ch - ((clamp(v, tMin, tMax) - tMin) / (tMax - tMin)) * ch;
    g.strokeStyle = c.surfaceEdge;
    g.setLineDash([4, 4]);
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(r.x, Math.round(ty(th.tempCrit)) + 0.5);
    g.lineTo(r.x + r.w, Math.round(ty(th.tempCrit)) + 0.5);
    g.stroke();
    g.setLineDash([]);
    if (pts.length < 2) {
      kit.drawClipped(g, ctx.t(TEXT.collecting), r.x + 6, cy + ch / 2, r.w - 12, { size: Math.max(9, Math.round(13 * sc)), family: f.mono, color: c.inkFaint });
    } else {
      // Load as a faint filled area (0..100 %), temperature as a line on top.
      area(g, pts, px, (p) => (p[2] === null ? null : cy + ch - (clamp(p[2], 0, 100) / 100) * ch), cy + ch, c.inkFaint);
      line(g, pts, px, (p) => (p[1] === null ? null : ty(p[1])), (p) => levelColor(tempLevel(p[1] ?? undefined, th), c), Math.max(1.5, 2.5 * sc));
    }
    g.strokeStyle = c.surfaceEdge;
    g.strokeRect(r.x + 0.5, cy + 0.5, r.w - 1, ch - 1);
    if (footH) {
      const fy = cy + ch + 14 * sc;
      const lbl = { size: Math.max(9, Math.round(11 * sc)), family: f.sansBold, tracking: 2 * sc, smallCaps: true, color: c.inkFaint };
      kit.drawClipped(g, `${ctx.t(TEXT.temp)} — ${ctx.t(TEXT.util)} ▒`, r.x, fy, r.w / 2, lbl);
      kit.drawTracked(g, fill(ctx.t(TEXT.minutes), { n: minutes }), r.x + r.w, fy, { ...lbl, align: 'right' });
    }
  }
}

function area(g: Graphics, pts: HistoryPoint[], px: (t: number) => number, py: (p: HistoryPoint) => number | null, base: number, color: string): void {
  g.fillStyle = color;
  g.globalAlpha = 0.6;
  let open = false;
  let lastX = 0;
  g.beginPath();
  for (const p of pts) {
    const y = py(p);
    const x = px(p[0]);
    if (y === null) {
      if (open) g.lineTo(lastX, base);
      open = false;
      continue;
    }
    if (!open) {
      g.moveTo(x, base);
      open = true;
    }
    g.lineTo(x, y);
    lastX = x;
  }
  if (open) g.lineTo(lastX, base);
  g.closePath();
  g.fill();
  g.globalAlpha = 1;
}

function line(g: Graphics, pts: HistoryPoint[], px: (t: number) => number, py: (p: HistoryPoint) => number | null, color: (p: HistoryPoint) => string, width: number): void {
  g.lineWidth = width;
  g.lineJoin = 'round';
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1]!;
    const b = pts[i]!;
    const ya = py(a);
    const yb = py(b);
    if (ya === null || yb === null) continue;
    g.strokeStyle = color(b);
    g.beginPath();
    g.moveTo(px(a[0]), ya);
    g.lineTo(px(b[0]), yb);
    g.stroke();
  }
}
