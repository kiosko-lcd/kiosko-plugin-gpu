/**
 * Pure parsers, one per tool output. Each takes the text the tool printed
 * and returns devices; none of them runs anything, so all are tested with
 * saved outputs.
 */
import type { GpuDevice, GpuProcess } from './types.ts';

/** A number, or undefined for `N/A`, `[N/A]`, `[Not Supported]`, empty… */
export function num(s: string | number | undefined | null): number | undefined {
  if (s === undefined || s === null) return undefined;
  if (typeof s === 'number') return Number.isFinite(s) ? s : undefined;
  const m = /-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/i.exec(s.trim());
  if (!m || /n\/a|not supported/i.test(s)) return undefined;
  const n = Number(m[0]);
  return Number.isFinite(n) ? n : undefined;
}

const set = <K extends keyof GpuDevice>(d: GpuDevice, k: K, v: GpuDevice[K] | undefined) => {
  if (v !== undefined) d[k] = v;
};

const basename = (p: string) => p.trim().split(/[\\/]/).pop() || p.trim();

/* ---------- NVIDIA: nvidia-smi --query-gpu=… --format=csv,noheader,nounits ---------- */

export const NVIDIA_FIELDS = [
  'index',
  'uuid',
  'name',
  'temperature.gpu',
  'temperature.memory',
  'utilization.gpu',
  'memory.used',
  'memory.total',
  'power.draw',
  'power.limit',
  'fan.speed',
  'clocks.sm',
  'clocks.mem',
  'clocks_throttle_reasons.active',
] as const;

export const NVIDIA_APP_FIELDS = ['gpu_uuid', 'pid', 'process_name', 'used_memory'] as const;

/**
 * Bits of `clocks_throttle_reasons.active` worth telling. Idle (0x1) and
 * application clocks (0x2) are normal life, so they are left out.
 */
const THROTTLE_BITS: [number, string][] = [
  [0x4, 'power cap'],
  [0x8, 'hw slowdown'],
  [0x20, 'thermal'],
  [0x40, 'hw thermal'],
  [0x80, 'power brake'],
];

export function throttleReasons(mask: string | undefined): string[] | undefined {
  if (!mask || !/^0x[0-9a-f]+$/i.test(mask.trim())) return undefined;
  const v = Number.parseInt(mask.trim().slice(2), 16);
  return THROTTLE_BITS.filter(([bit]) => (v & bit) !== 0).map(([, name]) => name);
}

export function parseNvidiaCsv(text: string): GpuDevice[] {
  const out: GpuDevice[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const f = line.split(',').map((x) => x.trim());
    if (f.length < NVIDIA_FIELDS.length) continue;
    const index = num(f[0]);
    if (index === undefined) continue;
    const d: GpuDevice = { index, uuid: f[1] || String(index), name: f[2] || 'GPU' };
    set(d, 'temp', num(f[3]));
    set(d, 'memTemp', num(f[4]));
    set(d, 'util', num(f[5]));
    set(d, 'memUsed', num(f[6]));
    set(d, 'memTotal', num(f[7]));
    set(d, 'power', num(f[8]));
    set(d, 'powerLimit', num(f[9]));
    set(d, 'fan', num(f[10]));
    set(d, 'smClock', num(f[11]));
    set(d, 'memClock', num(f[12]));
    set(d, 'throttle', throttleReasons(f[13]));
    out.push(d);
  }
  return out;
}

/** `--query-compute-apps=gpu_uuid,pid,process_name,used_memory`: processes by GPU uuid. */
export function parseNvidiaApps(text: string): Map<string, GpuProcess[]> {
  const out = new Map<string, GpuProcess[]>();
  for (const line of text.split(/\r?\n/)) {
    const f = line.split(',').map((x) => x.trim());
    if (f.length < 4) continue;
    // A process name may hold commas: it's everything between the pid and the last field.
    const uuid = f[0]!;
    const pid = num(f[1]);
    const mib = num(f[f.length - 1]);
    if (pid === undefined || mib === undefined || !uuid) continue;
    const list = out.get(uuid) ?? [];
    list.push({ pid, name: basename(f.slice(2, -1).join(',')), mib });
    out.set(uuid, list);
  }
  return out;
}

/* ---------- AMD: rocm-smi --json ---------- */

const MIB = 1024 * 1024;

/** `Navi 31 [Radeon RX 7900 XTX]` → `Radeon RX 7900 XTX`. */
function bracketed(s: string | undefined): string | undefined {
  if (!s) return undefined;
  const m = /\[([^\]]+)\]\s*$/.exec(s);
  return (m ? m[1]! : s).trim() || undefined;
}

export function parseRocmSmiJson(text: string): GpuDevice[] {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return [];
  }
  if (!json || typeof json !== 'object') return [];
  const out: GpuDevice[] = [];
  for (const [key, raw] of Object.entries(json as Record<string, unknown>)) {
    const m = /^card(\d+)$/.exec(key);
    if (!m || !raw || typeof raw !== 'object') continue;
    const r = raw as Record<string, string>;
    // Keys change between ROCm releases: match them by pattern, not verbatim.
    const pick = (re: RegExp) => {
      for (const [k, v] of Object.entries(r)) if (re.test(k)) return String(v);
      return undefined;
    };
    const index = Number(m[1]);
    const d: GpuDevice = {
      index,
      uuid: pick(/^unique id/i) ?? `card${index}`,
      name: bracketed(pick(/^card series/i)) ?? bracketed(pick(/^card model/i)) ?? 'AMD GPU',
    };
    set(d, 'temp', num(pick(/temperature.*edge/i)));
    set(d, 'hotspot', num(pick(/temperature.*junction/i)));
    set(d, 'memTemp', num(pick(/temperature.*memory/i)));
    set(d, 'util', num(pick(/^gpu use/i)));
    const total = num(pick(/vram total memory/i));
    const used = num(pick(/vram total used memory/i));
    if (total !== undefined) d.memTotal = Math.round(total / MIB);
    if (used !== undefined) d.memUsed = Math.round(used / MIB);
    set(d, 'power', num(pick(/(average|current socket).*power/i)));
    set(d, 'powerLimit', num(pick(/max graphics package power/i)));
    set(d, 'fan', num(pick(/^fan speed \(%\)/i)));
    set(d, 'smClock', num(pick(/^sclk/i)));
    set(d, 'memClock', num(pick(/^mclk/i)));
    out.push(d);
  }
  return out.sort((a, b) => a.index - b.index);
}

/* ---------- AMD: amd-smi metric --json (newer ROCm) ---------- */

/** `{ value, unit }` or a bare number, in MiB when the unit says bytes/MB/GB. */
function metric(v: unknown, toMiB = false): number | undefined {
  if (v && typeof v === 'object' && 'value' in v) {
    const n = num((v as { value: string | number }).value);
    if (n === undefined || !toMiB) return n;
    const unit = String((v as { unit?: string }).unit ?? 'MB').toUpperCase();
    return unit === 'B' ? n / MIB : unit === 'KB' ? n / 1024 : unit === 'GB' ? n * 1024 : n;
  }
  return typeof v === 'number' || typeof v === 'string' ? num(v) : undefined;
}

export function parseAmdSmiJson(text: string): GpuDevice[] {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return [];
  }
  // Older releases print a list, newer ones wrap it in { gpu_data: [...] }.
  const list = Array.isArray(json) ? json : json && typeof json === 'object' && Array.isArray((json as { gpu_data?: unknown }).gpu_data) ? (json as { gpu_data: unknown[] }).gpu_data : [];
  const out: GpuDevice[] = [];
  for (const raw of list) {
    if (!raw || typeof raw !== 'object') continue;
    const r = raw as Record<string, Record<string, unknown> | undefined> & { gpu?: number };
    const index = num(r.gpu as unknown as string) ?? out.length;
    const d: GpuDevice = { index, uuid: `card${index}`, name: 'AMD GPU' };
    set(d, 'util', metric(r.usage?.gfx_activity));
    set(d, 'power', metric(r.power?.socket_power));
    set(d, 'temp', metric(r.temperature?.edge));
    set(d, 'hotspot', metric(r.temperature?.hotspot));
    set(d, 'memTemp', metric(r.temperature?.mem));
    const total = metric(r.mem_usage?.total_vram, true);
    const used = metric(r.mem_usage?.used_vram, true);
    if (total !== undefined) d.memTotal = Math.round(total);
    if (used !== undefined) d.memUsed = Math.round(used);
    set(d, 'fan', metric(r.fan?.usage));
    set(d, 'smClock', metric((r.clock?.gfx_0 as Record<string, unknown> | undefined)?.clk));
    out.push(d);
  }
  return out;
}

/* ---------- Prometheus: DCGM exporter or nvidia_gpu_exporter ---------- */

export interface PromSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

/** The text exposition format, enough of it: comments skipped, labels unescaped, timestamps ignored. */
export function parsePromText(text: string): PromSample[] {
  const out: PromSample[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const m = /^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)/.exec(line);
    if (!m) continue;
    const value = Number(m[3]);
    if (!Number.isFinite(value)) continue;
    const labels: Record<string, string> = {};
    for (const l of (m[2] ?? '').matchAll(/([a-zA-Z_]\w*)="((?:[^"\\]|\\.)*)"/g)) labels[l[1]!] = l[2]!.replace(/\\(.)/g, (_, c: string) => (c === 'n' ? '\n' : c));
    out.push({ name: m[1]!, labels, value });
  }
  return out;
}

/** Which device field each metric fills, and how to scale it. */
const DCGM: Record<string, [keyof GpuDevice, number]> = {
  DCGM_FI_DEV_GPU_TEMP: ['temp', 1],
  DCGM_FI_DEV_MEMORY_TEMP: ['memTemp', 1],
  DCGM_FI_DEV_GPU_UTIL: ['util', 1],
  DCGM_FI_DEV_FB_USED: ['memUsed', 1],
  DCGM_FI_DEV_POWER_USAGE: ['power', 1],
  DCGM_FI_DEV_POWER_MGMT_LIMIT: ['powerLimit', 1],
  DCGM_FI_DEV_FAN_SPEED: ['fan', 1],
  DCGM_FI_DEV_SM_CLOCK: ['smClock', 1],
  DCGM_FI_DEV_MEM_CLOCK: ['memClock', 1],
};
const NV_EXPORTER: Record<string, [keyof GpuDevice, number]> = {
  nvidia_smi_temperature_gpu: ['temp', 1],
  nvidia_smi_temperature_memory: ['memTemp', 1],
  nvidia_smi_utilization_gpu_ratio: ['util', 100],
  nvidia_smi_memory_used_bytes: ['memUsed', 1 / MIB],
  nvidia_smi_memory_total_bytes: ['memTotal', 1 / MIB],
  nvidia_smi_power_draw_watts: ['power', 1],
  nvidia_smi_power_limit_watts: ['powerLimit', 1],
  nvidia_smi_fan_speed_ratio: ['fan', 100],
  nvidia_smi_clocks_current_sm_clock_hz: ['smClock', 1e-6],
  nvidia_smi_clocks_current_memory_clock_hz: ['memClock', 1e-6],
};

export function parsePrometheus(text: string): GpuDevice[] {
  const byId = new Map<string, GpuDevice>();
  const free = new Map<string, number>();
  const get = (s: PromSample): GpuDevice | undefined => {
    const uuid = s.labels.UUID ?? s.labels.uuid;
    const gpu = s.labels.gpu ?? s.labels.minor_number;
    const id = uuid ?? (gpu !== undefined ? `gpu${gpu}` : undefined);
    if (!id) return undefined;
    let d = byId.get(id);
    if (!d) {
      d = { index: num(gpu) ?? byId.size, uuid: id, name: 'GPU' };
      byId.set(id, d);
    }
    const name = s.labels.modelName ?? s.labels.name;
    if (name) d.name = name;
    return d;
  };
  for (const s of parsePromText(text)) {
    const map = DCGM[s.name] ?? NV_EXPORTER[s.name];
    if (map) {
      const d = get(s);
      // DCGM reports 0 °C for a sensor the card doesn't have (GeForce memory temperature).
      if (d && !(s.value === 0 && /temp/i.test(map[0]))) (d as unknown as Record<string, number>)[map[0]] = Math.round(s.value * map[1] * 10) / 10;
    } else if (s.name === 'DCGM_FI_DEV_FB_FREE') {
      const d = get(s);
      if (d) free.set(d.uuid, s.value);
    } else if (s.name === 'nvidia_smi_index') {
      const d = get(s);
      if (d) d.index = s.value;
    } else if (s.name === 'nvidia_smi_gpu_info') {
      get(s);
    }
  }
  for (const d of byId.values()) {
    const f = free.get(d.uuid);
    if (d.memTotal === undefined && f !== undefined && d.memUsed !== undefined) d.memTotal = d.memUsed + f;
  }
  return [...byId.values()].sort((a, b) => a.index - b.index);
}
