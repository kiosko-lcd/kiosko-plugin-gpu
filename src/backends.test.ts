import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { commandFor, MissingTool, readGpus, readSysfs, validSshHost, type Deps, type ReadOptions } from './backends.ts';

const fx = (name: string) => readFileSync(join(import.meta.dirname, 'fixtures', name), 'utf8');
const O: ReadOptions = { backend: 'auto', ssh: '', url: '', token: '', label: '', processes: true };
const noFetch: typeof fetch = async () => {
  throw new Error('no network in tests');
};

/** A fake runner answering by tool name, recording every argv. */
function fake(answers: Record<string, string | Error>): Deps & { calls: string[][] } {
  const calls: string[][] = [];
  return {
    calls,
    fetch: noFetch,
    sysRoot: '/nonexistent',
    run: async (argv) => {
      calls.push(argv);
      const cmd = argv[0] === 'ssh' ? argv[argv.indexOf('--') + 2]! : argv[0]!;
      const key = cmd === 'nvidia-smi' && argv.some((a) => a.startsWith('--query-compute-apps')) ? 'nvidia-apps' : cmd;
      const a = answers[key];
      if (a === undefined) throw new MissingTool(`${cmd} not found`);
      if (a instanceof Error) throw a;
      return a;
    },
  };
}

test('auto: nvidia-smi answers, processes attached by uuid, biggest first', async () => {
  const d = fake({ 'nvidia-smi': fx('nvidia-smi-2x4090.csv'), 'nvidia-apps': fx('nvidia-smi-apps.csv') });
  const s = await readGpus(O, d, undefined, 1000);
  assert.equal(s.available, true);
  assert.equal(s.backend, 'nvidia');
  assert.deepEqual(s.gpus[0]!.processes!.map((p) => p.name), ['ollama', 'python3']);
  assert.deepEqual(s.gpus[1]!.processes, []);
  assert.equal(s.label, undefined);
});

test('auto: no nvidia-smi falls through to rocm-smi', async () => {
  const s = await readGpus(O, fake({ 'rocm-smi': fx('rocm-smi.json') }));
  assert.equal(s.backend, 'amd');
  assert.equal(s.gpus.length, 2);
});

test('amd: rocm-smi missing, amd-smi answers', async () => {
  const s = await readGpus({ ...O, backend: 'amd' }, fake({ 'amd-smi': fx('amd-smi-metric.json') }));
  assert.equal(s.backend, 'amd');
  assert.equal(s.gpus[0]!.memTotal, 196592);
});

test('process list failing does not lose the GPUs', async () => {
  const s = await readGpus(O, fake({ 'nvidia-smi': fx('nvidia-smi-2x4090.csv'), 'nvidia-apps': new Error('boom') }));
  assert.equal(s.available, true);
  assert.equal(s.gpus[0]!.processes, undefined);
});

test('nothing at all: available false, "no GPU found", never throws', async () => {
  const s = await readGpus(O, fake({}));
  assert.deepEqual([s.available, s.reason, s.gpus.length], [false, 'no GPU found', 0]);
});

test('explicit back end failing gives its error as the reason', async () => {
  const s = await readGpus({ ...O, backend: 'nvidia' }, fake({ 'nvidia-smi': new Error('NVIDIA-SMI has failed because it couldn\'t communicate with the NVIDIA driver.') }));
  assert.equal(s.available, false);
  assert.match(s.reason!, /^nvidia: NVIDIA-SMI has failed/);
});

test('ssh: argv array, BatchMode, host after --, label from the host, no sysfs', async () => {
  const d = fake({ 'nvidia-smi': fx('nvidia-smi-a100.csv') });
  const s = await readGpus({ ...O, ssh: 'ops@gpubox' }, d);
  assert.equal(s.label, 'gpubox');
  assert.deepEqual(d.calls[0]!.slice(0, 7), ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '--', 'ops@gpubox']);
  assert.equal(d.calls[0]![7], 'nvidia-smi');
  const none = await readGpus({ ...O, ssh: 'gpubox', backend: 'intel' }, fake({}));
  assert.match(none.reason!, /only works locally/);
});

test('ssh host validation refuses options and shell', async () => {
  for (const bad of ['-oProxyCommand=sh', 'host;rm -rf /', 'a b', '$(id)', 'user@-x', '']) assert.equal(validSshHost(bad), false, bad);
  for (const ok of ['gpubox', 'ops@gpubox.lan', 'u_1@10.0.0.7']) assert.equal(validSshHost(ok), true, ok);
  assert.throws(() => commandFor(['nvidia-smi'], '-oProxyCommand=x'));
  const d = fake({ 'nvidia-smi': fx('nvidia-smi-a100.csv') });
  const s = await readGpus({ ...O, ssh: 'x;id' }, d);
  assert.equal(s.available, false);
  assert.equal(d.calls.length, 0, 'nothing ran');
});

test('prometheus: fetches the URL with the token, parses DCGM', async () => {
  let seen: [string, RequestInit | undefined] | undefined;
  const d: Deps = {
    ...fake({}),
    fetch: (async (url: string, init?: RequestInit) => {
      seen = [url, init];
      return new Response(fx('dcgm-exporter.txt'));
    }) as typeof fetch,
  };
  const s = await readGpus({ ...O, backend: 'prometheus', url: 'http://gpubox:9400/metrics', token: 'abc', label: 'rack' }, d);
  assert.equal(s.backend, 'prometheus');
  assert.equal(s.label, 'rack');
  assert.equal(s.gpus.length, 2);
  assert.equal(seen![0], 'http://gpubox:9400/metrics');
  assert.deepEqual(seen![1]!.headers, { authorization: 'Bearer abc' });
  const bad = await readGpus({ ...O, backend: 'prometheus' }, d);
  assert.match(bad.reason!, /no metrics URL/);
});

test('sysfs: amdgpu card with hwmon and VRAM, Intel card next to it', async () => {
  const root = mkdtempSync(join(tmpdir(), 'gpu-sysfs-'));
  const put = (p: string, v: string) => {
    mkdirSync(dirname(join(root, p)), { recursive: true });
    writeFileSync(join(root, p), `${v}\n`);
  };
  try {
    const amd = 'class/drm/card1/device';
    put(`${amd}/vendor`, '0x1002');
    put(`${amd}/gpu_busy_percent`, '73');
    put(`${amd}/mem_info_vram_total`, String(16 * 1024 ** 3));
    put(`${amd}/mem_info_vram_used`, String(9 * 1024 ** 3));
    put(`${amd}/hwmon/hwmon3/temp1_input`, '61000');
    put(`${amd}/hwmon/hwmon3/temp1_label`, 'edge');
    put(`${amd}/hwmon/hwmon3/temp2_input`, '78000');
    put(`${amd}/hwmon/hwmon3/temp2_label`, 'junction');
    put(`${amd}/hwmon/hwmon3/temp3_input`, '84000');
    put(`${amd}/hwmon/hwmon3/temp3_label`, 'mem');
    put(`${amd}/hwmon/hwmon3/power1_average`, '212000000');
    put(`${amd}/hwmon/hwmon3/power1_cap`, '263000000');
    put(`${amd}/hwmon/hwmon3/pwm1`, '128');
    put('class/drm/card0/device/vendor', '0x8086');
    put('class/drm/card0/device/hwmon/hwmon5/temp1_input', '47000');
    put('class/drm/card0-DP-1/status', 'connected');
    const [a] = await readSysfs(root, 'amd');
    assert.deepEqual(a, { index: 0, uuid: 'card1', name: 'AMD GPU', util: 73, memTotal: 16384, memUsed: 9216, temp: 61, hotspot: 78, memTemp: 84, power: 212, powerLimit: 263, fan: 50 });
    const [i] = await readSysfs(root, 'intel');
    assert.deepEqual(i, { index: 0, uuid: 'card0', name: 'Intel GPU', temp: 47 });
    const s = await readGpus(O, { ...fake({}), sysRoot: root });
    assert.equal(s.backend, 'amd-sysfs');
    assert.deepEqual(await readSysfs(join(root, 'nope'), 'amd'), []);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
