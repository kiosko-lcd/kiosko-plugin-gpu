/**
 * Where the readings come from. Every outside effect goes through `Deps`
 * (a command runner, `fetch`, the sysfs root), so the whole choice of back
 * end runs in tests against fake runners and a fake /sys.
 *
 * Commands are always argv arrays, never a shell line. Over ssh the remote
 * shell does join them, so only fixed strings of this file travel there;
 * the one user string, the host, is validated and goes after `--`.
 */
import { execFile } from 'node:child_process';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { NVIDIA_APP_FIELDS, NVIDIA_FIELDS, parseAmdSmiJson, parseNvidiaApps, parseNvidiaCsv, parsePrometheus, parseRocmSmiJson } from './parse.ts';
import type { Backend, GpuDevice, GpuSnapshot } from './types.ts';

export type Runner = (argv: string[], signal?: AbortSignal) => Promise<string>;

export interface Deps {
  run: Runner;
  fetch: typeof fetch;
  /** `/sys` normally; a temporary tree in tests. */
  sysRoot: string;
}

export interface ReadOptions {
  backend: Backend;
  /** `user@host` to run the CLI back ends over ssh; empty = this machine. */
  ssh: string;
  /** Prometheus metrics URL (DCGM exporter, nvidia_gpu_exporter). */
  url: string;
  /** Bearer token for the metrics URL, if it asks for one. */
  token: string;
  /** Name for the host; defaults to the ssh host. */
  label: string;
  processes: boolean;
}

/** An error saying the tool isn't there, as opposed to the tool failing. */
export class MissingTool extends Error {}

export const defaultRunner: Runner = (argv, signal) =>
  new Promise((resolve, reject) => {
    execFile(argv[0]!, argv.slice(1), { timeout: 15_000, maxBuffer: 4_000_000, signal, windowsHide: true }, (err, stdout, stderr) => {
      if (!err) return resolve(stdout);
      const e = err as unknown as { code?: string | number };
      if (e.code === 'ENOENT') return reject(new MissingTool(`${argv[0]} not found`));
      // ssh answers 127 when the remote command doesn't exist.
      if (argv[0] === 'ssh' && e.code === 127) return reject(new MissingTool(`${argv[argv.indexOf('--') + 2]} not found on the remote host`));
      reject(new Error(String(stderr || err.message).trim().split('\n')[0]));
    });
  });

export const DEFAULT_DEPS: Deps = { run: defaultRunner, fetch: (...a) => fetch(...a), sysRoot: '/sys' };

/** `user@host` or `host`, letters, digits, dot, dash, underscore; never an ssh option. */
export function validSshHost(host: string): boolean {
  return /^(?:[A-Za-z0-9_][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*$/.test(host);
}

/** The argv that runs `cmd` here, or on `ssh` when set. */
export function commandFor(cmd: string[], ssh: string): string[] {
  if (!ssh) return cmd;
  if (!validSshHost(ssh)) throw new Error(`invalid ssh host: ${ssh}`);
  return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '--', ssh, ...cmd];
}

export const NVIDIA_CMD = ['nvidia-smi', `--query-gpu=${NVIDIA_FIELDS.join(',')}`, '--format=csv,noheader,nounits'];
export const NVIDIA_APPS_CMD = ['nvidia-smi', `--query-compute-apps=${NVIDIA_APP_FIELDS.join(',')}`, '--format=csv,noheader,nounits'];
export const ROCM_CMD = ['rocm-smi', '--showproductname', '--showtemp', '--showuse', '--showmeminfo', 'vram', '--showpower', '--showmaxpower', '--showfan', '--showuniqueid', '--showclocks', '--json'];
export const AMDSMI_CMD = ['amd-smi', 'metric', '--json'];

async function readNvidia(o: ReadOptions, d: Deps, signal?: AbortSignal): Promise<GpuDevice[]> {
  const gpus = parseNvidiaCsv(await d.run(commandFor(NVIDIA_CMD, o.ssh), signal));
  if (o.processes && gpus.length) {
    try {
      const apps = parseNvidiaApps(await d.run(commandFor(NVIDIA_APPS_CMD, o.ssh), signal));
      for (const g of gpus) g.processes = (apps.get(g.uuid) ?? []).sort((a, b) => b.mib - a.mib);
    } catch {
      // Processes are a bonus: the GPUs without them are still worth showing.
    }
  }
  return gpus;
}

async function readAmdCli(o: ReadOptions, d: Deps, signal?: AbortSignal): Promise<GpuDevice[]> {
  try {
    const gpus = parseRocmSmiJson(await d.run(commandFor(ROCM_CMD, o.ssh), signal));
    if (gpus.length) return gpus;
  } catch (e) {
    if (!(e instanceof MissingTool)) throw e;
  }
  return parseAmdSmiJson(await d.run(commandFor(AMDSMI_CMD, o.ssh), signal));
}

/* ---------- sysfs (AMD and Intel, local only) ---------- */

const VENDORS: Record<string, 'amd' | 'intel' | 'nvidia'> = { '0x1002': 'amd', '0x8086': 'intel', '0x10de': 'nvidia' };

async function text(p: string): Promise<string | undefined> {
  try {
    return (await readFile(p, 'utf8')).trim();
  } catch {
    return undefined;
  }
}
async function number(p: string): Promise<number | undefined> {
  const t = await text(p);
  const n = t === undefined ? NaN : Number(t);
  return Number.isFinite(n) ? n : undefined;
}

/** Reads the DRM cards of a vendor from sysfs: hwmon temperatures, power and fan, VRAM on amdgpu. */
export async function readSysfs(root: string, vendor: 'amd' | 'intel'): Promise<GpuDevice[]> {
  const drm = join(root, 'class', 'drm');
  let cards: string[];
  try {
    cards = (await readdir(drm)).filter((c) => /^card\d+$/.test(c)).sort((a, b) => Number(a.slice(4)) - Number(b.slice(4)));
  } catch {
    return [];
  }
  const out: GpuDevice[] = [];
  for (const card of cards) {
    const dev = join(drm, card, 'device');
    if (VENDORS[(await text(join(dev, 'vendor'))) ?? ''] !== vendor) continue;
    const index = out.length;
    const g: GpuDevice = {
      index,
      uuid: (await text(join(dev, 'unique_id'))) ?? card,
      name: (await text(join(dev, 'product_name'))) || (vendor === 'amd' ? 'AMD GPU' : 'Intel GPU'),
    };
    const busy = await number(join(dev, 'gpu_busy_percent'));
    if (busy !== undefined) g.util = busy;
    const total = await number(join(dev, 'mem_info_vram_total'));
    const used = await number(join(dev, 'mem_info_vram_used'));
    if (total) g.memTotal = Math.round(total / 1024 ** 2);
    if (used !== undefined && total) g.memUsed = Math.round(used / 1024 ** 2);
    let hwmons: string[] = [];
    try {
      hwmons = (await readdir(join(dev, 'hwmon'))).sort();
    } catch {
      /* no hwmon: integrated or not exposed */
    }
    const hw = hwmons[0] ? join(dev, 'hwmon', hwmons[0]) : undefined;
    if (hw) {
      for (let i = 1; i <= 4; i++) {
        const v = await number(join(hw, `temp${i}_input`));
        if (v === undefined) continue;
        const label = ((await text(join(hw, `temp${i}_label`))) ?? (i === 1 ? 'edge' : '')).toLowerCase();
        const c = Math.round(v / 100) / 10;
        if (label === 'junction') g.hotspot = c;
        else if (label === 'mem' || label === 'vram') g.memTemp = c;
        else if (g.temp === undefined) g.temp = c;
      }
      const p = (await number(join(hw, 'power1_average'))) ?? (await number(join(hw, 'power1_input')));
      if (p !== undefined) g.power = Math.round(p / 1e5) / 10;
      const cap = (await number(join(hw, 'power1_cap'))) ?? (await number(join(hw, 'power1_max')));
      if (cap) g.powerLimit = Math.round(cap / 1e6);
      const pwm = await number(join(hw, 'pwm1'));
      if (pwm !== undefined) g.fan = Math.round((pwm / 255) * 100);
    }
    out.push(g);
  }
  return out;
}

/* ---------- Prometheus ---------- */

async function readPrometheus(o: ReadOptions, d: Deps, signal?: AbortSignal): Promise<GpuDevice[]> {
  if (!o.url) throw new Error('no metrics URL set');
  const res = await d.fetch(o.url, { headers: o.token ? { authorization: `Bearer ${o.token}` } : {}, signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10_000)]) : AbortSignal.timeout(10_000) });
  if (!res.ok) throw new Error(`metrics answered ${res.status}`);
  return parsePrometheus(await res.text());
}

/* ---------- Choosing ---------- */

export function hostLabel(o: Pick<ReadOptions, 'label' | 'ssh'>): string | undefined {
  return o.label.trim() || (o.ssh ? o.ssh.split('@').pop() : undefined) || undefined;
}

type Attempt = [string, () => Promise<GpuDevice[]>];

/**
 * Reads the GPUs with the chosen back end, or with the first that answers in
 * `auto`. Never throws: no GPU, no tool or a failing tool is `available: false`
 * with the reason, which the widgets print in faint ink.
 */
export async function readGpus(o: ReadOptions, d: Deps = DEFAULT_DEPS, signal?: AbortSignal, now = Date.now()): Promise<GpuSnapshot> {
  const label = hostLabel(o);
  const base = { gpus: [], history: {}, at: now, ...(label ? { label } : {}) };
  if (o.ssh && !validSshHost(o.ssh)) return { ...base, available: false, reason: `invalid ssh host: ${o.ssh}` };
  const nvidia: Attempt = ['nvidia', () => readNvidia(o, d, signal)];
  const amd: Attempt = ['amd', () => readAmdCli(o, d, signal)];
  const amdSys: Attempt = ['amd-sysfs', () => readSysfs(d.sysRoot, 'amd')];
  const intel: Attempt = ['intel', () => readSysfs(d.sysRoot, 'intel')];
  const prom: Attempt = ['prometheus', () => readPrometheus(o, d, signal)];
  // sysfs is this machine's: over ssh it would describe the wrong computer.
  const local = (a: Attempt[]) => (o.ssh ? [] : a);
  const plan: Attempt[] =
    o.backend === 'nvidia' ? [nvidia]
    : o.backend === 'amd' ? [amd, ...local([amdSys])]
    : o.backend === 'intel' ? local([intel])
    : o.backend === 'prometheus' ? [prom]
    : [...(o.url ? [prom] : []), nvidia, amd, ...local([amdSys, intel])];
  if (!plan.length) return { ...base, available: false, reason: 'the intel back end reads sysfs and only works locally' };
  const reasons: string[] = [];
  for (const [name, attempt] of plan) {
    try {
      const gpus = await attempt();
      if (gpus.length) return { ...base, available: true, backend: name, gpus };
      reasons.push(`${name}: no GPU`);
    } catch (e) {
      if (signal?.aborted) break;
      reasons.push(`${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return { ...base, available: false, reason: o.backend === 'auto' && reasons.every((r) => /not found|no GPU/.test(r)) ? 'no GPU found' : reasons.join('; ') };
}
