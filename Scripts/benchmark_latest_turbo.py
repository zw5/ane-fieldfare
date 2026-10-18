#!/usr/bin/env python3
"""Serial three-backend comparison using Turbo's real-generation-v1 protocol.

Build both release CLIs first, then run through with_benchmark_lock.sh. This
runner does not change either runtime, sampling, cache controls, or profiling.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
CASES = [('short-explanation', 20260721), ('medium-review', 20260722),
         ('long-synthesis', 20260723)]
ARMS = ['turbo', 'metal', 'hybrid-ane']
FOOTER = re.compile(r'\[stop=(.*?) prefill=(\d+)tok new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)\]')
MODEL_PROCESSES = ('TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|'
                   'TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|'
                   'ANEFieldfareHotpathCLI|ANEFieldfareQualityGate|ane-fieldfare-generate|mlx_lm|mlx-lm')


def capture(*command):
    return subprocess.check_output(command, text=True).strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def thermal():
    return subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True).stdout.strip()


def background_cpu():
    rows = capture('ps', '-axo', 'pcpu=,comm=').splitlines()
    parsed = []
    for row in rows:
        fields = row.strip().split(None, 1)
        if len(fields) == 2:
            parsed.append({'cpu_percent': float(fields[0]), 'executable': Path(fields[1]).name})
    return sorted(parsed, key=lambda r: r['cpu_percent'], reverse=True)[:10]


def source_identity(directory):
    paths = subprocess.check_output(['git', '-C', str(directory), 'ls-files', '-z',
                                     '--cached', '--others', '--exclude-standard',
                                     'Sources', 'Package.swift', 'Package.resolved'])
    h = hashlib.sha256()
    count = 0
    for name in sorted(set(paths.split(b'\0')) - {b''}):
        path = directory / os.fsdecode(name)
        if path.is_file():
            h.update(name + b'\0' + path.read_bytes() + b'\0')
            count += 1
    return {'git_head': capture('git', '-C', str(directory), 'rev-parse', 'HEAD'),
            'source_files': count, 'source_tree_sha256': h.hexdigest(),
            'git_status': capture('git', '-C', str(directory), 'status', '--short')}


def binary_identity(binary):
    h = hashlib.sha256()
    count = 0
    for bundle in sorted(binary.parent.glob('*.bundle')):
        for path in sorted(bundle.rglob('*.metal')):
            h.update(str(path.relative_to(binary.parent)).encode() + b'\0' + path.read_bytes())
            count += 1
    return {'path': str(binary), 'sha256': digest(binary),
            'metal_source_files': count, 'metal_resources_sha256': h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turbo-source', type=Path, required=True)
    parser.add_argument('--turbo-binary', type=Path, required=True)
    parser.add_argument('--candidate-binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--prior-warmups', type=Path,
                        help='Reuse completed warmups from a matching receipt without rebuilding.')
    parser.add_argument('--reverse-order', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    source = args.turbo_source.resolve()
    binaries = {'turbo': args.turbo_binary.resolve(), 'metal': args.candidate_binary.resolve(),
                'hybrid-ane': args.candidate_binary.resolve()}
    for binary in binaries.values():
        if not binary.is_file():
            parser.error(f'missing binary: {binary}')
    if not (ROOT / '.build/ane-fieldfare-benchmark.lock/owner').is_file():
        parser.error('run through Scripts/with_benchmark_lock.sh')
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'receipt.json').exists():
        parser.error('receipt already exists; choose a new output directory')
    model = ROOT / 'checkpoints/gturbo/gemma4.gturbo'
    fixtures = source / 'docs/benchmark-prompts/real-generation-v1'
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('ANE_FIELDFARE_', 'TURBO_FIELDFARE_', 'TURBOFIELDFARE_', 'MTL_'))}
    ancestors = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in ancestors:
        ancestors.add(parent)
        parent = int(capture('ps', '-p', str(parent), '-o', 'ppid='))
    receipt = {
        'schema': 'ane-fieldfare.latest-turbo-community-comparison.v1',
        'status': 'preflight', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runner_command': ['python3', str(Path(__file__).resolve())] + __import__('sys').argv[1:],
        'runner_sha256': digest(Path(__file__).resolve()),
        'host': platform.node(), 'cpu': capture('sysctl', '-n', 'machdep.cpu.brand_string'),
        'memory_bytes': int(capture('sysctl', '-n', 'hw.memsize')),
        'os_version': capture('sw_vers', '-productVersion'), 'os_build': capture('sw_vers', '-buildVersion'),
        'swift': capture('swift', '--version'), 'developer_dir': env.get('DEVELOPER_DIR'),
        'power': capture('pmset', '-g', 'batt'), 'power_settings': capture('pmset', '-g', 'custom'),
        'ane_architecture': {'isa_name': None, 'device_tree': capture('ioreg', '-p', 'IODeviceTree', '-n', 'ane', '-r', '-l', '-w', '0')},
        'upstream': source_identity(source), 'candidate': source_identity(ROOT),
        'binaries': {k: binary_identity(v) for k, v in binaries.items()},
        'protocol': {'max_new': 1024, 'max_context': 4096, 'temperature': 0.2, 'top_k': 64, 'top_p': 0.95,
                     'warmups_per_case_and_backend': 1, 'measured_repeats': args.repeats,
                     'profiling': False, 'process_per_run': 'fresh',
                     'warmup_receipt': str(args.prior_warmups.resolve()) if args.prior_warmups else None,
                     'reverse_order': args.reverse_order,
                     'order': 'All warmups precede measurements; backend order rotates with case and round, optionally reversed.'},
        'protocol_extensions_and_deviations': [
            'Same existing verified checkpoint path is shared without copying.',
            'Three backends interleaved serially; ANE Fieldfare retains its own production defaults and explicitly selects metal or hybrid-ane.',
            'Wall time, first stdout latency, and memory measured externally; profiling is disabled.',
            'Other user applications are not closed; background CPU, load average and thermal state are recorded.',
        ],
        'compile_count': None, 'load_count': None,
        'counter_boundary': 'Native ANE compile/load counters are not instrumented with profiling off. Separate numerical/lifecycle receipt: results/upstream-merge-20260915/receipt.json.',
        'claim_boundary': 'Product generation measurements. Compare speed ratios only when prompt/generated counts, settings and stop match; differing outputs also change expert routes. Coherent text is not an independent numerical-equivalence oracle. ANE hybrid offloads routed prefill; Metal executes companion stages and recurrent decode.',
        'runs': [],
    }

    def save():
        temporary = folder / 'receipt.json.tmp'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(folder / 'receipt.json')

    def preflight():
        active = subprocess.run(['pgrep', '-fl', MODEL_PROCESSES], capture_output=True, text=True)
        # The lock wrapper carries binary paths as arguments. Exclude only our
        # own ancestor processes, never another model or unrelated launcher.
        matches = [row for row in active.stdout.splitlines()
                   if int(row.split(None, 1)[0]) not in ancestors]
        if matches:
            raise RuntimeError('Another model process is active: ' + '\n'.join(matches))
        pressure = capture('memory_pressure', '-Q')
        free = int(re.search(r'free percentage: (\d+)', pressure).group(1))
        if free < 30:
            raise RuntimeError(f'Only {free}% free memory; require >=30%')
        disk = os.statvfs(ROOT)
        available = disk.f_bavail * disk.f_frsize
        if available < 2 * 1024 ** 3:
            raise RuntimeError('Less than 2 GiB disk headroom')
        return {'free_memory_percent': free, 'free_disk_bytes': available,
                'load_average': os.getloadavg(), 'thermal': thermal(), 'background_cpu': background_cpu()}

    save()
    # Watching our PID avoids passing CLI paths in caffeinate's own command
    # line, which macOS can retain in a separate assertion-holder process.
    subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    try:
        receipt['initial_preflight'] = preflight()
        manifest = json.loads((model / 'manifest.json').read_text())
        print('VERIFY existing checkpoint against manifest', flush=True)
        files = []
        for name, expected in manifest['files'].items():
            path = model / name
            if not path.is_file() or path.stat().st_size != expected['size'] or digest(path) != expected['sha256']:
                raise RuntimeError('Checkpoint integrity mismatch: ' + name)
            files.append({'path': name, 'size': expected['size'], 'sha256': expected['sha256']})
        receipt['model'] = {'path': str(model), 'manifest_sha256': digest(model / 'manifest.json'),
                            'verified_files': files, 'model_id': manifest['modelID']}
        receipt['fixtures'] = {case: {'path': str(fixtures / (case + '.json')),
                                      'sha256': digest(fixtures / (case + '.json')), 'seed': seed}
                               for case, seed in CASES}
        if args.prior_warmups:
            prior = json.loads(args.prior_warmups.read_text())
            if not prior['status'].startswith('completed'):
                raise RuntimeError('Prior measurement must be complete before reusing warmups')
            for arm in ARMS:
                if prior['binaries'][arm] != receipt['binaries'][arm]:
                    raise RuntimeError('Prior warmups used different binaries/resources: ' + arm)
                for case, _ in CASES:
                    warmups = [r for r in prior['runs'] if r['phase'] == 'warmup'
                               and r['arm'] == arm and r['case'] == case]
                    if not warmups or any(r['exit_status'] or r['stop'] != 'endOfTurn' for r in warmups):
                        raise RuntimeError('Missing successful warmup: ' + arm + '/' + case)
            if prior['model']['manifest_sha256'] != receipt['model']['manifest_sha256'] or prior['fixtures'] != receipt['fixtures']:
                raise RuntimeError('Prior warmups used different model or fixtures')
            for key in ('os_build', 'cpu', 'memory_bytes', 'swift'):
                if prior[key] != receipt[key]:
                    raise RuntimeError('Prior warmups used a different host/toolchain: ' + key)
            receipt['protocol_extensions_and_deviations'].append(
                'Reuses successful case/backend warmups from the matching earlier receipt; checkpoint hashes are reverified before the additional measured pass.')
        receipt['status'] = 'running'
        save()
        for round_index in range(1 if args.prior_warmups else 0, args.repeats + 1):
            phase = 'warmup' if round_index == 0 else f'measured-{round_index}'
            for case_index, (case, seed) in enumerate(CASES):
                rotation = (case_index + round_index) % len(ARMS)
                order = ARMS[rotation:] + ARMS[:rotation]
                if args.reverse_order:
                    order.reverse()
                for arm in order:
                    before = preflight()
                    cmd = [str(binaries[arm]), '--model', str(model), '--messages-file', str(fixtures / (case + '.json')),
                           '--max-new', '1024', '--max-context', '4096', '--temperature', '0.2',
                           '--top-k', '64', '--top-p', '0.95', '--seed', str(seed)]
                    if arm != 'turbo':
                        cmd += ['--backend', arm]
                    stem = f'{phase}-{case}-{arm}'
                    output, error = folder / (stem + '.stdout'), folder / (stem + '.stderr')
                    print('RUN', stem, flush=True)
                    start = time.monotonic()
                    first = None
                    reason = None
                    with output.open('wb') as out, error.open('wb') as err:
                        proc = subprocess.Popen(['/usr/bin/time', '-l'] + cmd, env=env, stdout=subprocess.PIPE,
                                                stderr=err, start_new_session=True)
                        selector = selectors.DefaultSelector()
                        selector.register(proc.stdout, selectors.EVENT_READ)
                        next_disk_check = start + 2
                        try:
                            while selector.get_map() or proc.poll() is None:
                                for key, _ in selector.select(timeout=0.1):
                                    block = os.read(key.fileobj.fileno(), 65536)
                                    if block:
                                        if first is None:
                                            first = time.monotonic() - start
                                        out.write(block)
                                        out.flush()
                                    else:
                                        selector.unregister(key.fileobj)
                                now = time.monotonic()
                                if now - start > 600:
                                    reason = '600-second wall watchdog'
                                    break
                                if now >= next_disk_check:
                                    disk = os.statvfs(ROOT)
                                    next_disk_check = now + 2
                                    if disk.f_bavail * disk.f_frsize < 2 * 1024 ** 3:
                                        reason = '2-GiB disk-headroom watchdog'
                                        break
                        finally:
                            selector.close()
                            if proc.poll() is None:
                                os.killpg(proc.pid, signal.SIGTERM)
                                try:
                                    proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            proc.stdout.close()
                    wall = time.monotonic() - start
                    stderr = error.read_text(errors='replace')
                    matches = list(FOOTER.finditer(stderr))
                    run = {'phase': phase, 'case': case, 'arm': arm, 'command': ['/usr/bin/time', '-l'] + cmd,
                           'environment_overrides': {}, 'exit_status': proc.returncode,
                           'stdout': str(output), 'stderr': str(error), 'stdout_sha256': digest(output),
                           'supervised_wall_seconds': wall, 'first_stdout_seconds': first,
                           'latency_boundary': 'Process spawn to first emitted UTF-8 bytes, including startup, loading, prefill and detokenizer buffering; not pure prefill time.',
                           'preflight': before, 'post_thermal': thermal(), 'watchdog': reason,
                           'compile_count': None if arm == 'hybrid-ane' else 0,
                           'load_count': None if arm == 'hybrid-ane' else 0}
                    if matches:
                        match = matches[-1]
                        stop, prompt, new, decode, rate = match.groups()
                        run.update(footer=match.group(0), stop=stop, prompt_tokens=int(prompt),
                                   new_tokens=int(new), decode_seconds=float(decode), tokens_per_second=float(rate))
                    measured_wall = re.search(r'([\d.]+)\s+real\s+[\d.]+\s+user\s+[\d.]+\s+sys', stderr)
                    if measured_wall:
                        run['process_wall_seconds'] = float(measured_wall.group(1))
                    for key, label in [('max_rss_bytes', 'maximum resident set size'), ('peak_footprint_bytes', 'peak memory footprint')]:
                        match = re.search(r'(\d+)\s+' + label, stderr)
                        run[key] = int(match.group(1)) if match else None
                    receipt['runs'].append(run)
                    save()
                    print('DONE', stem, run.get('footer'), 'wall', run.get('process_wall_seconds'),
                          'first stdout', round(first, 3) if first else None, flush=True)
                    if proc.returncode or reason or not matches:
                        raise RuntimeError('Incomplete run: ' + stem)
        receipt['all_measured_end_of_turn'] = all(r['stop'] == 'endOfTurn' for r in receipt['runs'] if r['phase'] != 'warmup')
        receipt['status'] = 'completed_pending_output_review'
        receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    except BaseException as error:
        receipt['status'] = 'incomplete'
        receipt['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
