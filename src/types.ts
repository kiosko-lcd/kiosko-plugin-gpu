/**
 * What the `gpu/devices` source hands over. Every reading but the identity
 * is optional: each back end knows a different subset (DCGM has no fan,
 * sysfs has no process list), and what can't be read is simply not painted.
 */

export interface GpuProcess {
  pid: number;
  /** Executable name, without its path (`ollama`, `python3`). */
  name: string;
  /** VRAM it holds, MiB. */
  mib: number;
}

export interface GpuDevice {
  index: number;
  /** Stable id (`GPU-5f1c…` on NVIDIA); the index as text when there is none. */
  uuid: string;
  /** As the driver reports it (`NVIDIA GeForce RTX 4090`). */
  name: string;
  /** Core / edge temperature, °C. */
  temp?: number;
  /** Memory temperature, °C (HBM, GDDR6X on some cards). */
  memTemp?: number;
  /** Hotspot / junction temperature, °C. */
  hotspot?: number;
  /** Utilisation, 0..100. */
  util?: number;
  /** VRAM, MiB. */
  memUsed?: number;
  memTotal?: number;
  /** Power, W. */
  power?: number;
  powerLimit?: number;
  /** Fan, 0..100 %. */
  fan?: number;
  /** Clocks, MHz. */
  smClock?: number;
  memClock?: number;
  /** Active throttle reasons that matter (`power cap`, `thermal`…); idle is not one. */
  throttle?: string[];
  processes?: GpuProcess[];
}

/** One history sample: [instant ms, temperature °C or null, utilisation % or null]. */
export type HistoryPoint = [number, number | null, number | null];

export type Backend = 'auto' | 'nvidia' | 'amd' | 'intel' | 'prometheus';

export interface GpuSnapshot {
  /** False when there is no GPU or no tool to read it: `reason` says why. */
  available: boolean;
  reason?: string;
  /** Back end that answered (`nvidia`, `amd`, `amd-sysfs`, `prometheus`…). */
  backend?: string;
  /** Host label to show, when set or when reading over ssh. */
  label?: string;
  gpus: GpuDevice[];
  /** Recent samples per GPU uuid, oldest first. */
  history: Record<string, HistoryPoint[]>;
  at: number;
}

export const DEVICES_SOURCE = 'gpu/devices';

export function snapshotOf(sources: Record<string, unknown>): GpuSnapshot | undefined {
  const v = sources[DEVICES_SOURCE];
  return v && typeof v === 'object' && Array.isArray((v as GpuSnapshot).gpus) ? (v as GpuSnapshot) : undefined;
}
