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
    label: { en: 'Back end', es: 'Origen', de: 'Backend', 'zh-Hant': '後端', 'zh-Hans': '后端' },
    help: { en: 'auto tries the metrics URL (if set), nvidia-smi, rocm-smi / amd-smi and sysfs, in that order.', es: 'auto prueba la URL de métricas (si hay), nvidia-smi, rocm-smi / amd-smi y sysfs, por ese orden.', de: 'Automatisch probiert die Metrik-URL (falls eingetragen), nvidia-smi, rocm-smi / amd-smi und sysfs, in dieser Reihenfolge.', 'zh-Hant': 'auto 會依序嘗試指標 URL（若有設定）、nvidia-smi、rocm-smi / amd-smi 及 sysfs。', 'zh-Hans': 'auto 会依次尝试指标 URL（如已设置）、nvidia-smi、rocm-smi / amd-smi 和 sysfs。' },
    labels: { auto: { en: 'Automatic', es: 'Automático', de: 'Automatisch', 'zh-Hant': '自動', 'zh-Hans': '自动' }, nvidia: 'NVIDIA (nvidia-smi)', amd: 'AMD (rocm-smi, amd-smi, sysfs)', intel: { en: 'Intel (sysfs, best effort)', es: 'Intel (sysfs, lo que se pueda)', de: 'Intel (sysfs, ohne Gewähr)', 'zh-Hant': 'Intel（sysfs，盡力而為）', 'zh-Hans': 'Intel（sysfs，尽力而为）' }, prometheus: 'Prometheus (DCGM / nvidia_gpu_exporter)' },
  }),
  ssh: t.string({
    placeholder: 'user@gpubox',
    risk: 'exec',
    label: { en: 'Read over ssh', es: 'Leer por ssh', de: 'Über ssh lesen', 'zh-Hant': '透過 ssh 讀取', 'zh-Hans': '通过 ssh 读取' },
    help: { en: 'Runs nvidia-smi / rocm-smi on that host with ssh -o BatchMode=yes: needs key login. Empty = this machine.', es: 'Ejecuta nvidia-smi / rocm-smi en ese host con ssh -o BatchMode=yes: necesita acceso por clave. Vacío = esta máquina.', de: 'Führt nvidia-smi / rocm-smi auf diesem Host mit ssh -o BatchMode=yes aus: braucht Anmeldung per Schlüssel. Leer = dieser Rechner.', 'zh-Hant': '以 ssh -o BatchMode=yes 在該主機上執行 nvidia-smi / rocm-smi：需要金鑰登入。留空 = 本機。', 'zh-Hans': '用 ssh -o BatchMode=yes 在该主机上运行 nvidia-smi / rocm-smi：需要密钥登录。留空 = 本机。' },
  }),
  url: t.string({
    format: 'url',
    secret: true,
    risk: 'network',
    placeholder: 'http://gpubox:9400/metrics',
    label: { en: 'Metrics URL', es: 'URL de métricas', de: 'Metrik-URL', 'zh-Hant': '指標 URL', 'zh-Hans': '指标 URL' },
    help: { en: 'DCGM exporter (port 9400) or nvidia_gpu_exporter (port 9835).', es: 'DCGM exporter (puerto 9400) o nvidia_gpu_exporter (puerto 9835).', de: 'DCGM-Exporter (Port 9400) oder nvidia_gpu_exporter (Port 9835).', 'zh-Hant': 'DCGM exporter（連接埠 9400）或 nvidia_gpu_exporter（連接埠 9835）。', 'zh-Hans': 'DCGM exporter（端口 9400）或 nvidia_gpu_exporter（端口 9835）。' },
  }),
  token: t.string({ secret: true, advanced: true, label: { en: 'Metrics bearer token', es: 'Token bearer de las métricas', de: 'Bearer-Token für Metriken', 'zh-Hant': '指標 Bearer 權杖', 'zh-Hans': '指标 Bearer 令牌' }, help: { en: 'Only if the metrics URL is behind auth.', es: 'Solo si la URL de métricas pide autenticación.', de: 'Nur wenn die Metrik-URL eine Anmeldung verlangt.', 'zh-Hant': '僅在指標 URL 需要驗證時填寫。', 'zh-Hans': '仅当指标 URL 需要认证时填写。' } }),
  label: t.string({ placeholder: 'gpubox', label: { en: 'Host label', es: 'Nombre del host', de: 'Host-Bezeichnung', 'zh-Hant': '主機標籤', 'zh-Hans': '主机标签' }, help: { en: 'Shown above the GPUs. Defaults to the ssh host.', es: 'Se muestra encima de las GPU. Por defecto, el host de ssh.', de: 'Wird über den GPUs angezeigt. Standard ist der ssh-Host.', 'zh-Hant': '顯示在 GPU 上方。預設為 ssh 主機。', 'zh-Hans': '显示在 GPU 上方。默认为 ssh 主机。' } }),
  processes: t.boolean({ default: true, label: { en: 'Per-process VRAM', es: 'VRAM por proceso', de: 'VRAM pro Prozess', 'zh-Hant': '各處理程序的顯示記憶體', 'zh-Hans': '各进程的显存' }, help: { en: 'One more nvidia-smi call per reading.', es: 'Una llamada más a nvidia-smi por lectura.', de: 'Ein zusätzlicher nvidia-smi-Aufruf pro Abfrage.', 'zh-Hant': '每次讀取多呼叫一次 nvidia-smi。', 'zh-Hans': '每次读取多调用一次 nvidia-smi。' } }),
  historyMs: t.duration({ default: 30 * 60_000, min: 60_000, max: 4 * 3_600_000, label: { en: 'History kept', es: 'Historial guardado', de: 'Aufbewahrter Verlauf', 'zh-Hant': '歷史保留時長', 'zh-Hans': '历史保留时长' }, help: { en: 'How far back GPU history can look.', es: 'Hasta dónde puede mirar atrás el historial de GPU.', de: 'Wie weit der GPU-Verlauf zurückreichen kann.', 'zh-Hant': 'GPU 歷史最多可回溯多久。', 'zh-Hans': 'GPU 历史最多可回溯多久。' } }),
});

/** The history, kept across reads by the one source instance. */
const history: Record<string, HistoryPoint[]> = {};

/** Overridable for tests: the runner, fetch and sysfs root. */
export const deps: Deps = { ...DEFAULT_DEPS };

export const devices = defineSource({
  id: 'devices',
  name: { en: 'GPUs', es: 'GPU', de: 'GPUs', 'zh-Hant': 'GPU', 'zh-Hans': 'GPU' },
  description: {
    en: 'Temperature, load, VRAM (and who holds it), power, fan and clocks per GPU, from this machine, another over ssh, or a Prometheus exporter.',
    es: 'Temperatura, carga, VRAM (y quién la ocupa), potencia, ventilador y relojes por GPU, de esta máquina, de otra por ssh o de un exporter de Prometheus.',
    de: 'Temperatur, Last, VRAM (und wer ihn belegt), Leistung, Lüfter und Takt pro GPU, von diesem Rechner, einem anderen über ssh oder einem Prometheus-Exporter.',
    'zh-Hant': '每張 GPU 的溫度、負載、顯示記憶體（及其佔用者）、功率、風扇與時脈，來源可為本機、透過 ssh 連線的另一台機器或 Prometheus exporter。',
    'zh-Hans': '每块 GPU 的温度、负载、显存（及其占用者）、功率、风扇和频率，可来自本机、通过 ssh 连接的另一台机器或 Prometheus exporter。',
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
  tempWarn: t.integer({ default: 80, min: 30, max: 120, unit: '°C', label: { en: 'Amber temperature', es: 'Temperatura ámbar', de: 'Temperatur gelb ab', 'zh-Hant': '溫度琥珀色門檻', 'zh-Hans': '温度琥珀色阈值' } }),
  tempAlert: t.integer({ default: 90, min: 30, max: 120, unit: '°C', label: { en: 'Red temperature', es: 'Temperatura roja', de: 'Temperatur rot ab', 'zh-Hant': '溫度紅色門檻', 'zh-Hans': '温度红色阈值' }, help: { en: 'Of the core (the hotspot or memory sensor when there is no core one). The widget keeps the tile while a GPU is past it.', es: 'Del núcleo (o del sensor de hotspot o de memoria si no hay del núcleo). El widget se queda la tesela mientras una GPU la pase.', de: 'Des Kerns (Hotspot- oder Speichersensor, wenn es keinen Kernsensor gibt). Das Widget behält die Kachel, solange eine GPU darüber liegt.', 'zh-Hant': '指核心溫度（沒有核心感測器時改用熱點或記憶體感測器）。只要有 GPU 超過此值，小工具就會持續佔用方格。', 'zh-Hans': '指核心温度（没有核心传感器时改用热点或显存传感器）。只要有 GPU 超过此值，小组件就会持续占用格子。' } }),
  vramWarn: t.integer({ default: 90, min: 0, max: 100, unit: '%', slider: true, label: { en: 'Amber VRAM from', es: 'VRAM ámbar desde', de: 'VRAM gelb ab', 'zh-Hant': '顯示記憶體琥珀色門檻', 'zh-Hans': '显存琥珀色阈值' } }),
  vramAlert: t.integer({ default: 97, min: 0, max: 100, unit: '%', slider: true, label: { en: 'Red VRAM from', es: 'VRAM roja desde', de: 'VRAM rot ab', 'zh-Hant': '顯示記憶體紅色門檻', 'zh-Hans': '显存红色阈值' } }),
  gpus: t.filter({
    label: { en: 'GPUs', es: 'GPU', de: 'GPUs', 'zh-Hant': 'GPU', 'zh-Hans': 'GPU' },
    help: { en: 'Empty: all. Indexes or model names, comma-separated: * matches anything, a leading ! leaves out (0, 1, *4090*, !*A6000*).', es: 'Vacío: todas. Índices o modelos separados por comas: * vale por cualquier cosa y un ! delante excluye (0, 1, *4090*, !*A6000*).', de: 'Leer: alle. Indizes oder Modellnamen, durch Kommas getrennt: * steht für beliebigen Text, ein vorangestelltes ! schließt aus (0, 1, *4090*, !*A6000*).', 'zh-Hant': '留空：全部。以逗號分隔的索引或型號：* 符合任何內容，開頭加 ! 表示排除（0, 1, *4090*, !*A6000*）。', 'zh-Hans': '留空：全部。以逗号分隔的索引或型号：* 匹配任意内容，开头加 ! 表示排除（0, 1, *4090*, !*A6000*）。' },
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
  name: { en: 'GPU cards', es: 'Tarjetas de GPU', de: 'GPU-Karten', 'zh-Hant': 'GPU 卡片', 'zh-Hans': 'GPU 卡片' },
  description: {
    en: 'One column per GPU: temperature, load, VRAM and power bars, fan and throttling. Dim until a threshold is crossed.',
    es: 'Una columna por GPU: temperatura, carga, barras de VRAM y potencia, ventilador y limitación. Apagada hasta cruzar un umbral.',
    de: 'Eine Spalte pro GPU: Balken für Temperatur, Last, VRAM und Leistung, dazu Lüfter und Drosselung. Gedimmt, bis ein Schwellenwert überschritten ist.',
    'zh-Hant': '每張 GPU 一欄：溫度、負載、顯示記憶體與功率長條、風扇和降頻狀態。未超過門檻前保持暗色。',
    'zh-Hans': '每块 GPU 一列：温度、负载、显存和功率条、风扇及降频状态。未超过阈值前保持暗色。',
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
  name: { en: 'GPU memory', es: 'Memoria de GPU', de: 'GPU-Speicher', 'zh-Hant': 'GPU 顯示記憶體', 'zh-Hans': 'GPU 显存' },
  description: {
    en: 'VRAM per GPU as a bar stacked by process (ollama, vllm, python…), with what is left free.',
    es: 'VRAM por GPU como una barra apilada por proceso (ollama, vllm, python…), con lo que queda libre.',
    de: 'VRAM pro GPU als nach Prozessen gestapelter Balken (ollama, vllm, python…), mit dem, was frei bleibt.',
    'zh-Hant': '每張 GPU 的顯示記憶體以依處理程序堆疊的長條呈現（ollama、vllm、python…），並標出剩餘空間。',
    'zh-Hans': '每块 GPU 的显存以按进程堆叠的条形显示（ollama、vllm、python…），并标出剩余空间。',
  },
  icon: 'cpu',
  category: 'system',
  tags: ['gpu', 'vram', 'models', 'ai'],
  size: { min: { w: 200, h: 60 }, preferred: { w: 960, h: 230 } },
  options: t.object({ ...options, maxProcesses: t.integer({ default: 4, min: 1, max: 12, label: { en: 'Processes per bar', es: 'Procesos por barra', de: 'Prozesse pro Balken', 'zh-Hant': '每條長條的處理程序數', 'zh-Hans': '每条的进程数' }, help: { en: 'The rest is grouped as "other".', es: 'El resto se agrupa como «otros».', de: 'Der Rest wird als „andere“ zusammengefasst.', 'zh-Hant': '其餘歸為「其他」。', 'zh-Hans': '其余归为“其他”。' } }) }),
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
  name: { en: 'GPU history', es: 'Historial de GPU', de: 'GPU-Verlauf', 'zh-Hant': 'GPU 歷史', 'zh-Hans': 'GPU 历史' },
  description: {
    en: 'Temperature line over load area for the last minutes, per GPU, with the red temperature dashed.',
    es: 'Línea de temperatura sobre área de carga de los últimos minutos, por GPU, con la temperatura roja discontinua.',
    de: 'Temperaturlinie über der Lastfläche der letzten Minuten, pro GPU, mit der roten Temperaturschwelle gestrichelt.',
    'zh-Hant': '每張 GPU 最近幾分鐘的溫度折線疊在負載面積圖上，紅色溫度門檻以虛線標示。',
    'zh-Hans': '每块 GPU 最近几分钟的温度折线叠加在负载面积图上，红色温度阈值以虚线标示。',
  },
  icon: 'chart',
  category: 'system',
  tags: ['gpu', 'temperature', 'history', 'chart'],
  size: { min: { w: 200, h: 100 }, preferred: { w: 960, h: 230 } },
  options: t.object({ ...options, historyMs: t.duration({ default: 30 * 60_000, min: 60_000, max: 4 * 3_600_000, label: { en: 'Time shown', es: 'Tiempo mostrado', de: 'Angezeigter Zeitraum', 'zh-Hant': '顯示時長', 'zh-Hans': '显示时长' }, help: { en: 'Up to the history kept by the source.', es: 'Hasta el historial que guarda la fuente.', de: 'Bis zum Verlauf, den die Datenquelle aufbewahrt.', 'zh-Hant': '最多為資料來源保留的歷史長度。', 'zh-Hans': '最多为数据源保留的历史长度。' } }) }),
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
  name: { en: 'GPU', es: 'GPU', de: 'GPU', 'zh-Hant': 'GPU', 'zh-Hans': 'GPU' },
  version: '0.2.1',
  languages: ['en', 'es', 'de', 'zh-Hant', 'zh-Hans'],
  sdk: '^0.2.0',
  permissions: { exec: ['nvidia-smi', 'rocm-smi', 'amd-smi', 'ssh'], files: ['/sys/class/drm'], network: ['user'] },
  description: {
    en: 'The GPUs of your AI box: temperature, load, VRAM by process and history. Local, over ssh or from a Prometheus exporter.',
    es: 'Las GPU de tu máquina de IA: temperatura, carga, VRAM por proceso e historial. En local, por ssh o desde un exporter de Prometheus.',
    de: 'Die GPUs deines KI-Rechners: Temperatur, Last, VRAM nach Prozess und Verlauf. Lokal, über ssh oder von einem Prometheus-Exporter.',
    'zh-Hant': '你 AI 主機上的 GPU：溫度、負載、各處理程序的顯示記憶體與歷史。可在本機讀取、透過 ssh 或從 Prometheus exporter 取得。',
    'zh-Hans': '你 AI 主机上的 GPU：温度、负载、各进程的显存和历史。可在本机读取、通过 ssh 或从 Prometheus exporter 获取。',
  },
  author: 'Kiosko',
  license: 'MIT',
  icon: 'cpu',
  widgets: [cards, vram, historyWidget],
  sources: [devices],
});
