/**
 * Pure decisions about GPU readings: names, levels, figures, the history
 * ring buffer, when to take the cell over and when to let the panel sleep.
 * Tested without drawing.
 */
import { matchFilter, type Filter } from '@kiosko/sdk';
import type { GpuDevice, GpuSnapshot, HistoryPoint } from './types.ts';

export interface Thresholds {
  /** Core temperature, °C: amber from `tempWarn`, red from `tempAlert`. */
  tempWarn: number;
  tempAlert: number;
  /** VRAM fill, % (0..100). */
  vramWarn: number;
  vramAlert: number;
}

export type Level = 'normal' | 'warn' | 'alert';

export const DEFAULT_THRESHOLDS: Thresholds = { tempWarn: 80, tempAlert: 90, vramWarn: 90, vramAlert: 97 };

/** `NVIDIA GeForce RTX 4090` → `RTX 4090`; `AMD Radeon RX 7900 XTX` → `RX 7900 XTX`. */
export function shortName(name: string): string {
  const s = name
    .replace(/\((?:R|TM)\)/gi, '')
    .replace(/\b(?:NVIDIA|GeForce|AMD|Radeon|Intel|Graphics)\b/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
  return s || name.trim() || 'GPU';
}

/** `GPU 0 · RTX 4090`. */
export function gpuTitle(g: Pick<GpuDevice, 'index' | 'name'>): string {
  return `GPU ${g.index} · ${shortName(g.name)}`;
}

/**
 * The temperature the thresholds apply to: the core's, else the hotspot's,
 * else the memory's. Not the hottest of them: GDDR6X at 92 °C and a hotspot
 * 15 °C over the core are normal life, and the same line for all would keep
 * a healthy card in red.
 */
export function mainTemp(g: GpuDevice): number | undefined {
  return g.temp ?? g.hotspot ?? g.memTemp;
}

export function tempLevel(c: number | undefined, t: Pick<Thresholds, 'tempWarn' | 'tempAlert'>): Level {
  return c === undefined ? 'normal' : c >= t.tempAlert ? 'alert' : c >= t.tempWarn ? 'warn' : 'normal';
}

export function vramRatio(g: GpuDevice): number | undefined {
  return g.memTotal && g.memUsed !== undefined ? g.memUsed / g.memTotal : undefined;
}

export function vramLevel(g: GpuDevice, t: Pick<Thresholds, 'vramWarn' | 'vramAlert'>): Level {
  const r = vramRatio(g);
  if (r === undefined) return 'normal';
  const pct = r * 100;
  return pct >= t.vramAlert ? 'alert' : pct >= t.vramWarn ? 'warn' : 'normal';
}

export function gpuTempLevel(g: GpuDevice, t: Thresholds): Level {
  return tempLevel(mainTemp(g), t);
}

export function worst(...l: Level[]): Level {
  return l.includes('alert') ? 'alert' : l.includes('warn') ? 'warn' : 'normal';
}

/** 18432 MiB → `18.0G`, 512 → `512M`. */
export function gib(mib: number): string {
  if (mib < 1024) return `${Math.round(mib)}M`;
  const g = mib / 1024;
  return `${g >= 100 ? Math.round(g) : g.toFixed(1)}G`;
}

export function watts(w: number): string {
  return `${Math.round(w)}W`;
}

/** The names a GPU answers to in a `gpus` filter: its index, uuid, full and short model names. */
export function gpuNames(g: Pick<GpuDevice, 'index' | 'uuid' | 'name'>): string[] {
  return [String(g.index), g.uuid, g.name, shortName(g.name), `GPU ${g.index}`];
}

/** The snapshot with only the GPUs the filter lets through (the same object when the filter is empty). */
export function filterGpus<S extends GpuSnapshot | undefined>(s: S, filter: Filter): S {
  if (!s || !(typeof filter === 'string' ? filter.trim() : filter.length)) return s;
  return { ...s, gpus: s.gpus.filter((g) => matchFilter(gpuNames(g), filter)) };
}

/** Any GPU past its red temperature: worth taking the cell over. */
export function anyAlert(s: GpuSnapshot | undefined, t: Thresholds): boolean {
  return !!s?.available && s.gpus.some((g) => gpuTempLevel(g, t) === 'alert');
}

/** Any GPU working (load or a process holding memory) or in the red. */
export function anyBusy(s: GpuSnapshot | undefined, t: Thresholds, utilFloor = 5): boolean {
  if (!s?.available) return false;
  return s.gpus.some((g) => (g.util ?? 0) >= utilFloor || (g.processes?.length ?? 0) > 0 || gpuTempLevel(g, t) === 'alert' || vramLevel(g, t) === 'alert');
}

/* ---------- History ring buffer ---------- */

/**
 * Adds a sample per GPU (not more than one every `minGapMs`) and drops what's
 * older than `windowMs`. Mutates and returns `h`: the source keeps one across
 * reads, so the history survives without a widget holding state.
 */
export function pushHistory(h: Record<string, HistoryPoint[]>, gpus: GpuDevice[], now: number, windowMs: number, minGapMs = 4_000): Record<string, HistoryPoint[]> {
  for (const g of gpus) {
    const list = (h[g.uuid] ??= []);
    const last = list[list.length - 1];
    if (!last || now - last[0] >= minGapMs) list.push([now, g.temp ?? null, g.util ?? null]);
    while (list.length && list[0]![0] < now - windowMs) list.shift();
  }
  for (const k of Object.keys(h)) {
    const list = h[k]!;
    while (list.length && list[0]![0] < now - windowMs) list.shift();
    if (!list.length) delete h[k];
  }
  return h;
}
