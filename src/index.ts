/**
 * GPUs of a local AI box: a shared `gpu/devices` source (nvidia-smi,
 * rocm-smi / amd-smi, sysfs, a Prometheus exporter, or any of the CLIs over
 * ssh) and three widgets that read it: per-GPU cards, VRAM by process and
 * the temperature / load history.
 */
import { defineSource, definePlugin, defineWidget, t, type Filter } from '@kiosko/sdk';
import { DEFAULT_DEPS, readGpus, type Deps } from './backends.ts';
import { drawCards, drawHistory, drawVram } from './draw.ts';
import { anyAlert, anyBusy, filterGpus, pushHistory, type Thresholds } from './gpu.ts';
import { DEVICES_SOURCE, snapshotOf, type GpuDevice, type GpuSnapshot, type HistoryPoint } from './types.ts';

export type { GpuDevice, GpuProcess, GpuSnapshot, HistoryPoint } from './types.ts';
export { readGpus } from './backends.ts';

/* ---------- Demo: an RTX 3090 serving a 70B model through Ollama and an A6000 running vLLM ---------- */

export const DEMO_GPUS: GpuDevice[] = [
  {
    index: 0,
    uuid: 'GPU-3a9e2c41-7d0b-4f6e-9a51-0c2d8e7f1b36',
    name: 'NVIDIA GeForce RTX 3090',
    temp: 74,
    memTemp: 92,
    util: 96,
    memUsed: 22_874,
    memTotal: 24_576,
    power: 338,
    powerLimit: 350,
    fan: 78,
    smClock: 1785,
    memClock: 9751,
    throttle: ['power cap'],
    processes: [
      { pid: 2214, name: 'ollama', mib: 21_012 },
      { pid: 3307, name: 'python3', mib: 1_590 },
    ],
  },
  {
    index: 1,
    uuid: 'GPU-c81f5b07-22e4-4a93-b6d8-5e40f19a7c02',
    name: 'NVIDIA RTX A6000',
    temp: 63,
    util: 41,
    memUsed: 44_120,
    memTotal: 49_140,
    power: 187,
    powerLimit: 300,
    fan: 46,
    smClock: 1410,
    memClock: 8001,
    throttle: [],
    processes: [{ pid: 4120, name: 'vllm', mib: 43_890 }],
  },
];

/** Thirty minutes of a plausible afternoon: a long generation on GPU 0, a batch job ramping up on GPU 1. */
export function demoHistory(now: number, minutes = 30): Record<string, HistoryPoint[]> {
  const n = Math.round((minutes * 60) / 10);
  const h: Record<string, HistoryPoint[]> = {};
  for (const g of DEMO_GPUS) {
    h[g.uuid] = Array.from({ length: n + 1 }, (_, i): HistoryPoint => {
      const k = i / n;
      const at = now - (n - i) * 10_000;
      if (g.index === 0) {
        const busy = k > 0.25 ? 1 : 0.1;
        const util = Math.round(clamp01(busy * (0.92 + 0.06 * Math.sin(i / 3))) * 100);
        return [at, Math.round(48 + 26 * busy * (1 - Math.exp(-(k - 0.25) * 12 * (k > 0.25 ? 1 : 0))) + Math.sin(i / 5)), util];
      }
      const util = Math.round(clamp01(0.05 + 0.4 * k + 0.05 * Math.sin(i / 6)) * 100);
      return [at, Math.round(38 + 25 * k + Math.sin(i / 4)), util];
    });
  }
  return h;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export function demoSnapshot(now: number): GpuSnapshot {
  return { available: true, backend: 'nvidia', label: 'gpubox', gpus: DEMO_GPUS, history: demoHistory(now), at: now };
}

/* ---------- Source ---------- */

const sourceOptions = t.object({
  backend: t.enum(['auto', 'nvidia', 'amd', 'intel', 'prometheus'] as const, {
    label: { en: 'Back end', es: 'Origen' },
    help: { en: 'auto tries the metrics URL (if set), nvidia-smi, rocm-smi / amd-smi and sysfs, in that order.', es: 'auto prueba la URL de métricas (si hay), nvidia-smi, rocm-smi / amd-smi y sysfs, por ese orden.' },
    labels: { auto: { en: 'Automatic', es: 'Automático' }, nvidia: 'NVIDIA (nvidia-smi)', amd: 'AMD (rocm-smi, amd-smi, sysfs)', intel: { en: 'Intel (sysfs, best effort)', es: 'Intel (sysfs, lo que se pueda)' }, prometheus: 'Prometheus (DCGM / nvidia_gpu_exporter)' },
  }),
  ssh: t.string({
    placeholder: 'user@gpubox',
    risk: 'exec',
    label: { en: 'Read over ssh', es: 'Leer por ssh' },
    help: { en: 'Runs nvidia-smi / rocm-smi on that host with ssh -o BatchMode=yes: needs key login. Empty = this machine.', es: 'Ejecuta nvidia-smi / rocm-smi en ese host con ssh -o BatchMode=yes: necesita acceso por clave. Vacío = esta máquina.' },
  }),
  url: t.string({
    format: 'url',
    secret: true,
    risk: 'network',
    placeholder: 'http://gpubox:9400/metrics',
    label: { en: 'Metrics URL', es: 'URL de métricas' },
    help: { en: 'DCGM exporter (port 9400) or nvidia_gpu_exporter (port 9835).', es: 'DCGM exporter (puerto 9400) o nvidia_gpu_exporter (puerto 9835).' },
  }),
  token: t.string({ secret: true, advanced: true, label: { en: 'Metrics bearer token', es: 'Token bearer de las métricas' }, help: { en: 'Only if the metrics URL is behind auth.', es: 'Solo si la URL de métricas pide autenticación.' } }),
  label: t.string({ placeholder: 'gpubox', label: { en: 'Host label', es: 'Nombre del host' }, help: { en: 'Shown above the GPUs. Defaults to the ssh host.', es: 'Se muestra encima de las GPU. Por defecto, el host de ssh.' } }),
  processes: t.boolean({ default: true, label: { en: 'Per-process VRAM', es: 'VRAM por proceso' }, help: { en: 'One more nvidia-smi call per reading.', es: 'Una llamada más a nvidia-smi por lectura.' } }),
  historyMs: t.duration({ default: 30 * 60_000, min: 60_000, max: 4 * 3_600_000, label: { en: 'History kept', es: 'Historial guardado' }, help: { en: 'How far back GPU history can look.', es: 'Hasta dónde puede mirar atrás el historial de GPU.' } }),
});

/** The history, kept across reads by the one source instance. */
const history: Record<string, HistoryPoint[]> = {};

/** Overridable for tests: the runner, fetch and sysfs root. */
export const deps: Deps = { ...DEFAULT_DEPS };

export const devices = defineSource({
  id: 'devices',
  name: { en: 'GPUs', es: 'GPU' },
  description: {
    en: 'Temperature, load, VRAM (and who holds it), power, fan and clocks per GPU, from this machine, another over ssh, or a Prometheus exporter.',
    es: 'Temperatura, carga, VRAM (y quién la ocupa), potencia, ventilador y relojes por GPU, de esta máquina, de otra por ssh o de un exporter de Prometheus.',
  },
  options: sourceOptions,
  intervalMs: 5_000,
  // One nvidia-smi call (two with per-process VRAM) is cheap; faster than this the history only gets noisier.
  minIntervalMs: 1_000,
  // Each back end gives up on its own after 10-15 s; `auto` may try several.
  timeoutMs: 20_000,
  read: ({ options, signal, now }) => readDevices(options, signal, now),
  demo: ({ now }) => demoSnapshot(now),
});

/** One reading, with the history it adds to. */
export async function readDevices(o: typeof sourceOptions.default, signal: AbortSignal | undefined, now: number): Promise<GpuSnapshot> {
  const snap = await readGpus(o, deps, signal, now);
  snap.history = snap.available ? structuredClone(pushHistory(history, snap.gpus, now, o.historyMs)) : {};
  return snap;
}

/* ---------- Widgets ---------- */

const options = {
  tempWarn: t.integer({ default: 80, min: 30, max: 120, unit: '°C', label: { en: 'Amber temperature', es: 'Temperatura ámbar' } }),
  tempAlert: t.integer({ default: 90, min: 30, max: 120, unit: '°C', label: { en: 'Red temperature', es: 'Temperatura roja' }, help: { en: 'Of the core (the hotspot or memory sensor when there is no core one). The widget keeps the tile while a GPU is past it.', es: 'Del núcleo (o del sensor de hotspot o de memoria si no hay del núcleo). El widget se queda la tesela mientras una GPU la pase.' } }),
  vramWarn: t.integer({ default: 90, min: 0, max: 100, unit: '%', slider: true, label: { en: 'Amber VRAM from', es: 'VRAM ámbar desde' } }),
  vramAlert: t.integer({ default: 97, min: 0, max: 100, unit: '%', slider: true, label: { en: 'Red VRAM from', es: 'VRAM roja desde' } }),
  gpus: t.filter({
    label: { en: 'GPUs', es: 'GPU' },
    help: { en: 'Empty: all. Indexes or model names, comma-separated: * matches anything, a leading ! leaves out (0, 1, *4090*, !*A6000*).', es: 'Vacío: todas. Índices o modelos separados por comas: * vale por cualquier cosa y un ! delante excluye (0, 1, *4090*, !*A6000*).' },
  }),
};

type Opts = Thresholds & { gpus: Filter };
type State = { options: Opts; sources: Record<string, unknown> };
/** The widget's GPUs: the source's, through its `gpus` filter. */
const gpusOf = ({ options, sources }: State) => filterGpus(snapshotOf(sources), options.gpus);
// Keeps the tile while one of its GPUs is past the red temperature (the instance's "keep the tile" switch can turn it off).
const urgent = (st: State) => anyAlert(gpusOf(st), st.options);
// Dark when the GPUs rest: no load, no process holding memory, nothing in the red.
const idle = (st: State) => !anyBusy(gpusOf(st), st.options);
const demo = ({ now }: { now: number }) => ({ sources: { [DEVICES_SOURCE]: demoSnapshot(now) } });

export const cards = defineWidget({
  id: 'cards',
  name: { en: 'GPU cards', es: 'Tarjetas de GPU' },
  description: {
    en: 'One column per GPU: temperature, load, VRAM and power bars, fan and throttling. Dim until a threshold is crossed.',
    es: 'Una columna por GPU: temperatura, carga, barras de VRAM y potencia, ventilador y limitación. Apagada hasta cruzar un umbral.',
  },
  icon: 'cpu',
  category: 'system',
  tags: ['gpu', 'nvidia', 'amd', 'vram', 'temperature', 'ai'],
  size: { min: { w: 150, h: 120 }, preferred: { w: 640, h: 462 } },
  options: t.object(options),
  sources: [DEVICES_SOURCE],
  render(g, ctx) {
    drawCards(g, ctx, gpusOf(ctx), ctx.options);
  },
  urgent,
  idle,
  demo,
});

export const vram = defineWidget({
  id: 'vram',
  name: { en: 'GPU memory', es: 'Memoria de GPU' },
  description: {
    en: 'VRAM per GPU as a bar stacked by process (ollama, vllm, python…), with what is left free.',
    es: 'VRAM por GPU como una barra apilada por proceso (ollama, vllm, python…), con lo que queda libre.',
  },
  icon: 'cpu',
  category: 'system',
  tags: ['gpu', 'vram', 'models', 'ai'],
  size: { min: { w: 200, h: 60 }, preferred: { w: 960, h: 230 } },
  options: t.object({ ...options, maxProcesses: t.integer({ default: 4, min: 1, max: 12, label: { en: 'Processes per bar', es: 'Procesos por barra' }, help: { en: 'The rest is grouped as "other".', es: 'El resto se agrupa como «otros».' } }) }),
  sources: [DEVICES_SOURCE],
  render(g, ctx) {
    drawVram(g, ctx, gpusOf(ctx), ctx.options, ctx.options.maxProcesses);
  },
  urgent,
  idle,
  demo,
});

export const historyWidget = defineWidget({
  id: 'history',
  name: { en: 'GPU history', es: 'Historial de GPU' },
  description: {
    en: 'Temperature line over load area for the last minutes, per GPU, with the red temperature dashed.',
    es: 'Línea de temperatura sobre área de carga de los últimos minutos, por GPU, con la temperatura roja discontinua.',
  },
  icon: 'chart',
  category: 'system',
  tags: ['gpu', 'temperature', 'history', 'chart'],
  size: { min: { w: 200, h: 100 }, preferred: { w: 960, h: 230 } },
  options: t.object({ ...options, historyMs: t.duration({ default: 30 * 60_000, min: 60_000, max: 4 * 3_600_000, label: { en: 'Time shown', es: 'Tiempo mostrado' }, help: { en: 'Up to the history kept by the source.', es: 'Hasta el historial que guarda la fuente.' } }) }),
  sources: [DEVICES_SOURCE],
  render(g, ctx) {
    drawHistory(g, ctx, gpusOf(ctx), ctx.options, ctx.options.historyMs);
  },
  urgent,
  idle,
  demo,
});

export default definePlugin({
  id: 'gpu',
  name: { en: 'GPU', es: 'GPU' },
  version: '0.2.0',
  description: {
    en: 'The GPUs of your AI box: temperature, load, VRAM by process and history. Local, over ssh or from a Prometheus exporter.',
    es: 'Las GPU de tu máquina de IA: temperatura, carga, VRAM por proceso e historial. En local, por ssh o desde un exporter de Prometheus.',
  },
  author: 'Kiosko',
  license: 'MIT',
  icon: 'cpu',
  widgets: [cards, vram, historyWidget],
  sources: [devices],
});
