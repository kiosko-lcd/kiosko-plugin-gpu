import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { num, parseAmdSmiJson, parseNvidiaApps, parseNvidiaCsv, parsePrometheus, parsePromText, parseRocmSmiJson, throttleReasons } from './parse.ts';

const fx = (name: string) => readFileSync(join(import.meta.dirname, 'fixtures', name), 'utf8');

test('num: N/A and friends are undefined, units are ignored', () => {
  assert.equal(num('N/A'), undefined);
  assert.equal(num('[N/A]'), undefined);
  assert.equal(num('[Not Supported]'), undefined);
  assert.equal(num(''), undefined);
  assert.equal(num('412.37'), 412.37);
  assert.equal(num('(2482Mhz)'), 2482);
  assert.equal(num('2.3924310016e+10'), 23924310016);
});

test('nvidia-smi CSV: two RTX 4090s, one busy and one idle', () => {
  const [a, b] = parseNvidiaCsv(fx('nvidia-smi-2x4090.csv'));
  assert.deepEqual(a, {
    index: 0,
    uuid: 'GPU-5f1c3b8e-2a47-9d61-0e3f-7b4c9a2d18e5',
    name: 'NVIDIA GeForce RTX 4090',
    temp: 71,
    util: 98,
    memUsed: 22816,
    memTotal: 24564,
    power: 412.37,
    powerLimit: 450,
    fan: 64,
    smClock: 2730,
    memClock: 10501,
    throttle: ['power cap'],
  });
  assert.equal((a as { memTemp?: number }).memTemp, undefined, 'GeForce has no memory temperature');
  assert.deepEqual(b!.throttle, [], 'idle is not a throttle reason');
  assert.equal(b!.util, 0);
});

test('nvidia-smi CSV: A100 with memory temperature, no fan, thermal throttling', () => {
  const [a] = parseNvidiaCsv(fx('nvidia-smi-a100.csv'));
  assert.equal(a!.memTemp, 66);
  assert.equal(a!.fan, undefined);
  assert.deepEqual(a!.throttle, ['thermal', 'hw thermal']);
  assert.equal(parseNvidiaCsv('garbage\n\n').length, 0);
});

test('throttle reasons', () => {
  assert.deepEqual(throttleReasons('0x0000000000000088'), ['hw slowdown', 'power brake']);
  assert.equal(throttleReasons('[N/A]'), undefined);
});

test('nvidia-smi compute apps: by uuid, path stripped, N/A memory skipped', () => {
  const apps = parseNvidiaApps(fx('nvidia-smi-apps.csv'));
  assert.deepEqual(apps.get('GPU-5f1c3b8e-2a47-9d61-0e3f-7b4c9a2d18e5'), [
    { pid: 2214, name: 'ollama', mib: 21504 },
    { pid: 3311, name: 'python3', mib: 1206 },
  ]);
  assert.equal(apps.get('GPU-a03e7d52-91bc-4f08-c6d2-3e8f05b7a914'), undefined);
  assert.equal(parseNvidiaApps('').size, 0, 'no compute apps prints nothing');
});

test('rocm-smi JSON: 7900 XTX busy and W7900 idle', () => {
  const [a, b] = parseRocmSmiJson(fx('rocm-smi.json'));
  assert.deepEqual(a, {
    index: 0,
    uuid: '0x3f7a2b19c4d8e061',
    name: 'Radeon RX 7900 XTX',
    temp: 54,
    hotspot: 71,
    memTemp: 66,
    util: 87,
    memTotal: 24560,
    memUsed: 20480,
    power: 297,
    powerLimit: 327,
    fan: 28,
    smClock: 2482,
    memClock: 1249,
  });
  assert.equal(b!.name, 'Radeon Pro W7900');
  assert.equal(b!.power, 11, 'older releases say Average Graphics Package Power');
  assert.equal(b!.memTotal, 49136);
  assert.deepEqual(parseRocmSmiJson('not json'), []);
});

test('amd-smi metric JSON: MI300X, N/A fields left out', () => {
  const [a] = parseAmdSmiJson(fx('amd-smi-metric.json'));
  assert.equal(a!.util, 64);
  assert.equal(a!.power, 512);
  assert.equal(a!.temp, undefined);
  assert.equal(a!.hotspot, 61);
  assert.equal(a!.memTotal, 196592);
  assert.equal(a!.memUsed, 143360);
  assert.equal(a!.fan, undefined);
  assert.equal(a!.smClock, 1705);
  assert.equal(parseAmdSmiJson(JSON.stringify({ gpu_data: [{ gpu: 1, usage: { gfx_activity: { value: 3 } } }] }))[0]!.index, 1);
});

test('Prometheus text: labels, escapes, comments', () => {
  const [s] = parsePromText('# HELP x\nfoo{a="b\\"c",d="e"} 1.5 1700000000\n');
  assert.deepEqual(s, { name: 'foo', labels: { a: 'b"c', d: 'e' }, value: 1.5 });
});

test('DCGM exporter: two GPUs, total from used + free, 0 °C memory sensor dropped', () => {
  const [a, b] = parsePrometheus(fx('dcgm-exporter.txt'));
  assert.deepEqual(a, {
    index: 0,
    uuid: 'GPU-3a9e2c41-7d0b-4f6e-9a51-0c2d8e7f1b36',
    name: 'NVIDIA GeForce RTX 3090',
    smClock: 1785,
    memClock: 9751,
    temp: 74,
    power: 338.1,
    util: 96,
    memUsed: 22874,
    memTotal: 24198,
  });
  assert.equal(b!.name, 'NVIDIA RTX A6000');
  assert.equal(b!.memTotal, 48622);
});

test('nvidia_gpu_exporter: ratios to %, bytes to MiB', () => {
  const [a] = parsePrometheus(fx('nvidia-gpu-exporter.txt'));
  assert.equal(a!.name, 'NVIDIA GeForce RTX 4090');
  assert.equal(a!.index, 0);
  assert.equal(a!.util, 98);
  assert.equal(a!.fan, 64);
  assert.equal(a!.memUsed, 22816);
  assert.equal(a!.memTotal, 24564);
  assert.equal(a!.powerLimit, 450);
});
