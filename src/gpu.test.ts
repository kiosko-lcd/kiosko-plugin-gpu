import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as kit from '@loksly/lcd-canvas-kit';
import { THEMES, localize, type RenderContext } from '@kiosko/sdk';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { anyBusy, anyCritical, DEFAULT_THRESHOLDS as T, gib, gpuTitle, gpuTempLevel, pushHistory, shortName, vramLevel } from './gpu.ts';
import plugin, { cards, DEMO_GPUS, demoSnapshot, deps, historyWidget, readWithCache, vram } from './index.ts';
import { DEVICES_SOURCE, type GpuSnapshot, type HistoryPoint } from './types.ts';

test('names: GPU index · short model', () => {
  assert.equal(gpuTitle({ index: 0, name: 'NVIDIA GeForce RTX 4090' }), 'GPU 0 · RTX 4090');
  assert.equal(shortName('NVIDIA RTX A6000'), 'RTX A6000');
  assert.equal(shortName('Radeon RX 7900 XTX'), 'RX 7900 XTX');
  assert.equal(shortName('Intel(R) Arc(TM) A770 Graphics'), 'Arc A770');
  assert.equal(shortName('NVIDIA'), 'NVIDIA');
});

test('levels: the core temperature counts; VRAM by ratio', () => {
  assert.equal(gpuTempLevel({ index: 0, uuid: 'a', name: 'x', temp: 70 }, T), 'normal');
  assert.equal(gpuTempLevel({ index: 0, uuid: 'a', name: 'x', temp: 70, memTemp: 92 }, T), 'normal', 'GDDR6X at 92 °C is normal life');
  assert.equal(gpuTempLevel({ index: 0, uuid: 'a', name: 'x', hotspot: 93 }, T), 'alert', 'no core sensor: the hotspot counts');
  assert.equal(gpuTempLevel({ index: 0, uuid: 'a', name: 'x', temp: 83 }, T), 'warn');
  assert.equal(vramLevel(DEMO_GPUS[0]!, T), 'warn');
  assert.equal(vramLevel({ index: 0, uuid: 'a', name: 'x' }, T), 'normal');
  assert.equal(gib(22874), '22.3G');
  assert.equal(gib(512), '512M');
});

test('urgent only past critical; idle when the GPUs rest', () => {
  const calm: GpuSnapshot = { available: true, gpus: [{ index: 0, uuid: 'a', name: 'x', temp: 40, util: 0, processes: [] }], history: {}, at: 0 };
  assert.equal(anyCritical(calm, T), false);
  assert.equal(anyBusy(calm, T), false);
  assert.equal(anyBusy({ ...calm, gpus: [{ ...calm.gpus[0]!, processes: [{ pid: 1, name: 'ollama', mib: 9000 }] }] }, T), true, 'a model loaded is worth seeing');
  const hot = { ...calm, gpus: [{ ...calm.gpus[0]!, temp: 93 }] };
  assert.equal(anyCritical(hot, T), true);
  const st = (s: GpuSnapshot, takeOver = true) => ({ options: { ...T, takeOver, maxProcesses: 4, minutes: 30 }, data: undefined, sources: { [DEVICES_SOURCE]: s }, now: 0, working: true });
  assert.equal(cards.urgent!(st(hot)), true);
  assert.equal(cards.urgent!(st(hot, false)), false);
  assert.equal(cards.urgent!(st(calm)), false);
  assert.equal(cards.idle!(st(calm)), true);
  assert.equal(vram.idle!(st(demoSnapshot(0))), false);
  assert.equal(cards.urgent!(st({ available: false, reason: 'x', gpus: [], history: {}, at: 0 })), false);
});

test('history ring buffer: spaced samples, trimmed window, vanished GPUs dropped', () => {
  const h: Record<string, HistoryPoint[]> = {};
  const g = [{ index: 0, uuid: 'a', name: 'x', temp: 50, util: 10 }];
  pushHistory(h, g, 0, 60_000);
  pushHistory(h, g, 2_000, 60_000);
  assert.equal(h.a!.length, 1, 'closer than 4 s is skipped');
  for (let t = 5_000; t <= 120_000; t += 5_000) pushHistory(h, g, t, 60_000);
  assert.equal(h.a![0]![0], 60_000);
  assert.deepEqual(h.a!.at(-1), [120_000, 50, 10]);
  pushHistory(h, [], 300_000, 60_000);
  assert.equal(h.a, undefined);
});

test('source: reads honour the interval option and carry the history', async () => {
  const csv = '0, GPU-x, NVIDIA GeForce RTX 4090, 60, N/A, 50, 1000, 24564, 200, 450, 40, 2000, 10000, 0x1\n';
  let runs = 0;
  const saved = { ...deps };
  Object.assign(deps, { run: async (argv: string[]) => (runs++, argv.some((a) => a.includes('compute-apps')) ? '' : csv), sysRoot: '/nonexistent' });
  try {
    const o = { backend: 'nvidia' as const, ssh: '', url: '', token: '', label: '', processes: true, historyMinutes: 30, intervalMs: 5_000 };
    const a = await readWithCache(o, undefined, 1_000_000);
    const b = await readWithCache(o, undefined, 1_002_000);
    assert.equal(a, b, 'within the interval the snapshot is reused');
    assert.equal(runs, 2);
    const c = await readWithCache(o, undefined, 1_005_000);
    assert.equal(c.history['GPU-x']!.length, 2);
  } finally {
    Object.assign(deps, saved);
  }
});

test('plugin shape: one source, three widgets, all with demo; secrets masked', () => {
  assert.equal(plugin.id, 'gpu');
  assert.deepEqual(plugin.widgets!.map((w) => w.id), ['cards', 'vram', 'history']);
  assert.deepEqual(plugin.sources!.map((s) => s.id), ['devices']);
  for (const w of plugin.widgets!) assert.ok(w.demo, w.id);
  const props = (plugin.sources![0]!.options as { props: Record<string, { secret?: boolean }> }).props;
  assert.equal(props.token!.secret, true);
  assert.equal(props.url!.secret, true);
  for (const w of plugin.widgets!) assert.equal(typeof localize(w.name, 'es'), 'string');
});

/* ---------- Every widget paints, on every panel size, with data and without ---------- */

const SIZES: [number, number][] = [
  [1920, 462],
  [960, 230],
  [480, 480],
  [320, 240],
  [150, 120],
];

test('render smoke: demo, unavailable and missing source, on strip and small panels', () => {
  const theme = THEMES[0]!;
  const now = 1_762_700_000_000;
  const snaps: Record<string, unknown>[] = [
    { [DEVICES_SOURCE]: demoSnapshot(now) },
    { [DEVICES_SOURCE]: { available: false, reason: 'no GPU found', gpus: [], history: {}, at: now } },
    {},
    { [DEVICES_SOURCE]: { ...demoSnapshot(now), history: {}, label: undefined } },
  ];
  for (const w of [cards, vram, historyWidget]) {
    for (const [W, H] of SIZES) {
      for (const sources of snaps) {
        const canvas = createCanvas(W, H);
        const g = canvas.getContext('2d');
        const ctx = {
          now,
          locale: 'es',
          t: (x: Parameters<typeof localize>[0]) => localize(x, 'es'),
          log: () => {},
          dataDir: '/tmp',
          options: { ...T, takeOver: true, maxProcesses: 4, minutes: 30 },
          data: undefined,
          sources,
          working: true,
          box: { x: 10, y: 10, w: W - 20, h: H - 20 },
          theme,
          fonts: theme.fonts,
          kit,
          canvas: { createCanvas, loadImage },
          preview: false,
        } as unknown as RenderContext<never, never>;
        w.render(g, ctx);
      }
    }
  }
});
