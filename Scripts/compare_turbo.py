#!/usr/bin/env python3
"""Counterbalanced exact decode screen; requires the pinned local Turbo build and installed model.

Run through Scripts/with_benchmark_lock.sh. This is not the upstream community benchmark.
"""
import argparse,hashlib,json,os,pathlib,platform,re,signal,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--arms',default='turbo,pread16,pread16,turbo')
p.add_argument('--case',default='short-explanation')
p.add_argument('--tokens',type=int,default=128)
p.add_argument('--temperature',default='0')
p.add_argument('--tag',required=True)
p.add_argument('--profile',action='store_true')
p.add_argument('--toolchain',default='not recorded; do not infer matched compiler builds')
a=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.tag): p.error('tag must be a simple directory name')
if a.tokens < 1: p.error('tokens must be positive')
if a.case not in ('short-explanation','medium-review','long-synthesis'): p.error('unknown fixture')
for arm in a.arms.split(','):
 if not re.fullmatch(r'cliANE|simd(?:512|1024)|anesimd(?:512|1024)|hybrid(?:512|1024)|default|cached(?:512|1024)?|aneroute(?:1|30|512|1024)|ane(?:1|6|15|30)|aneprefill(?:1|6|15|30|512|1024)|turbo(?:8|16|24|32)?|(?:pread|mmap)(?:8|16|17|18|24|32)', arm): p.error('unsupported arm: '+arm)
model=ROOT/'checkpoints/gturbo/gemma4.gturbo'
if not (model/'manifest.json').is_file(): raise SystemExit('completed model is required')
folder=ROOT/'results'/a.tag
folder.mkdir(exist_ok=False)
fixture=ROOT/'.build/turbo-upstream/docs/benchmark-prompts/real-generation-v1'/f'{a.case}.json'
if not fixture.is_file(): raise SystemExit('unknown fixture')
base_env={k:v for k,v in os.environ.items() if not k.startswith('ANE_FIELDFARE_')}
footer_re=re.compile(r'\[stop=(.*?) prefill=(\d+)tok new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)\]')
receipt={'schema':'ane-fieldfare.decode-comparison.v1','declared_build_toolchain':a.toolchain,'host':platform.node(),'os_version':subprocess.check_output(['sw_vers','-productVersion'],text=True).strip(),'os_build':subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),'ane_architecture':'not exposed by this runner','counter_boundary':'Actual ANE compile/load/evaluation counts in candidate profiles when --profile is enabled; Metal shader libraries are separate','turbo_commit':subprocess.check_output(['git','-C',str(ROOT/'.build/turbo-upstream'),'rev-parse','HEAD'],text=True).strip(),'fixture':str(fixture),'max_new':a.tokens,'temperature':a.temperature,'wall_clock_boundary':'process_wall_seconds is /usr/bin/time real at 0.01s resolution; wall_seconds includes supervisor polling latency','counterbalanced_order':a.arms.split(',') == list(reversed(a.arms.split(','))) and len(set(a.arms.split(','))) > 1,'experiment_boundary':'optimization screen in the listed order, not the official 1024-token community protocol; candidate profiling '+('enabled' if a.profile else 'disabled'),'runs':[]}
device_tree = subprocess.run(['ioreg','-p','IODeviceTree','-n','ane','-r','-l','-w','0'],capture_output=True,text=True)
compat = re.search(r'"compatible" = <"([^"<>]+)">',device_tree.stdout)
ane_type = re.search(r'"ane-type" = <([0-9a-f]+)>',device_tree.stdout)
receipt['ane_architecture'] = {'isa_name':None,'io_device_tree_compatible':compat.group(1) if compat else None,'ane_type_hex_bytes':ane_type.group(1) if ane_type else None,'claim_boundary':'Raw device-tree identifiers; no undocumented ISA mapping is asserted.'}
def save():
 (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for index,arm in enumerate(a.arms.split(',')):
 env=base_env.copy()
 if arm.startswith('turbo'): binary=ROOT/'.build/turbo-reference/release/TurboFieldfareCLI'
 else:
  binary=ROOT/'.build/swift-hotpath/release/ANEFieldfareHotpathCLI'
  if a.profile: env['ANE_FIELDFARE_HOTPATH_PROFILE']='1'
  if arm not in ('default', 'cliANE'): env['ANE_FIELDFARE_PREFILL_ATTENTION']='full-tensorops-2d-preferred'
  if arm.startswith('cached'): env['ANE_FIELDFARE_HOTPATH_PREFILL_TRANSPORT']='cached-pread'
  if arm.startswith(('simd','anesimd')):
   env['ANE_FIELDFARE_PREFILL_ATTENTION']='causal-simd'
   if arm.startswith('anesimd'): env['ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS']='30'
  elif arm.startswith('hybrid'):
   env['ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS']='30'
   env['ANE_FIELDFARE_ANE_SHARED_LAYERS']='30'
   env['ANE_FIELDFARE_ANE_SHARED_MODE']='prefill'
  elif arm.startswith('aneroute'):
   env['ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS']='30' if arm in ('aneroute512','aneroute1024') else arm[8:]
  elif arm.startswith('ane'):
   env['ANE_FIELDFARE_ANE_SHARED_LAYERS']='30' if arm in ('aneprefill512','aneprefill1024') else re.search(r'\d+$',arm).group()
   env['ANE_FIELDFARE_ANE_SHARED_MODE']='prefill' if arm.startswith('aneprefill') else 'decode'
  if arm not in ('default', 'cliANE') and not arm.startswith(('cached','ane','hybrid','simd')):
   mode,slots=re.fullmatch(r'(pread|mmap)(\d+)',arm).groups()
   env['ANE_FIELDFARE_HOTPATH_CACHE_SLOTS']=slots
   env['ANE_FIELDFARE_HOTPATH_WINDOW_MMAP']='1' if mode=='mmap' else '0'
 cmd=[str(binary),'--model',str(model),'--messages-file',str(fixture),'--max-new',str(a.tokens),'--max-context','4096','--temperature',a.temperature,'--top-k','0' if a.temperature=='0' else '64','--top-p','1' if a.temperature=='0' else '0.95','--seed',{'short-explanation':'20260721','medium-review':'20260722','long-synthesis':'20260723'}[a.case]]
 if arm.startswith('turbo'):
  if arm != 'turbo': cmd+=['--expert-cache-slots',arm[5:]]
 else:
  if arm == 'cliANE': cmd += ['--backend', 'hybrid-ane', '--prefill-chunk', '1024', '--expert-top-k', '8']
  elif arm == 'default': cmd += ['--expert-top-k', '8']
  else: cmd+=['--prefill-chunk',re.search(r'(512|1024)$',arm).group() if re.search(r'(512|1024)$',arm) else '128','--expert-top-k','8']
 active=subprocess.run(['pgrep','-fl','TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|ANEFieldfareHotpathCLI|ane-fieldfare-generate|mlx_lm|mlx-lm'],capture_output=True,text=True)
 if active.returncode==0:
  receipt.update(status='rejected_preflight',preflight_rejection={'arm':arm,'reason':'another model process active','compile_count':0,'load_count':0,'exit_status':1})
  save(); raise SystemExit('another model process active: '+active.stdout)
 pressure=subprocess.check_output(['memory_pressure','-Q'],text=True)
 free=int(re.search(r'free percentage: (\d+)',pressure).group(1))
 if free<30:
  receipt.update(status='rejected_preflight',preflight_rejection={'arm':arm,'reason':'insufficient free memory','observed_free_memory_percent':free,'required_free_memory_percent':30,'compile_count':0,'load_count':0,'exit_status':1})
  save(); raise SystemExit('insufficient preflight free memory: '+pressure)
 stem=f'{index:02d}-{arm}'
 out,err=folder/(stem+'.stdout'),folder/(stem+'.stderr')
 shader_digest = hashlib.sha256()
 shader_count = 0
 for bundle in sorted(binary.parent.glob('*.bundle')):
  for shader in sorted(bundle.rglob('*.metal')):
   shader_digest.update(str(shader.relative_to(binary.parent)).encode()); shader_digest.update(shader.read_bytes()); shader_count += 1
 binary_identity = hashlib.sha256(binary.read_bytes()).hexdigest()
 start=time.monotonic(); reason=None
 print('RUN',stem,flush=True)
 with out.open('wb') as of,err.open('wb') as ef:
  proc=subprocess.Popen(['/usr/bin/time','-l']+cmd,env=env,stdout=of,stderr=ef,start_new_session=True)
  while proc.poll() is None:
   elapsed=time.monotonic()-start
   if elapsed>max(180,a.tokens/4): reason='wall watchdog';break
   if os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize<2*1024**3: reason='disk headroom watchdog';break
   time.sleep(1)
  if reason:
   os.killpg(proc.pid,signal.SIGTERM)
   try: proc.wait(timeout=5)
   except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 wall=time.monotonic()-start
 stderr=err.read_text(errors='replace'); matches=footer_re.findall(stderr)
 profiles=[]
 for line in stderr.splitlines():
  if line.startswith('{'):
   try: profiles.append(json.loads(line))
   except json.JSONDecodeError: pass
 run={'binary_sha256':binary_identity,'metal_resources_sha256':shader_digest.hexdigest(),'metal_resource_files':shader_count,'arm':arm,'command':['/usr/bin/time','-l']+cmd,'environment':{k:v for k,v in env.items() if k.startswith('ANE_FIELDFARE_')},'exit_status':proc.returncode,'wall_seconds':wall,'watchdog':reason,'stdout':str(out),'stderr':str(err),'preflight_free_memory_percent':free,'load_average':os.getloadavg(),'profiles':profiles}
 if matches:
  stop,prompt,new,decode,rate=matches[-1]
  run.update(stop=stop,prompt_tokens=int(prompt),new_tokens=int(new),decode_seconds=float(decode),tokens_per_second=float(rate),footer=matches[-1])
 m=re.search(r'([\d.]+)\s+real\s+[\d.]+\s+user\s+[\d.]+\s+sys',stderr)
 if m: run['process_wall_seconds']=float(m.group(1))
 for key,label in [('max_rss','maximum resident set size'),('peak_footprint','peak memory footprint')]:
  m=re.search(r'(\d+)\s+'+label,stderr)
  run[key]=int(m.group(1)) if m else None
 receipt['runs'].append(run);save()
 print('DONE',stem,'exit',proc.returncode,'tps',run.get('tokens_per_second'),'wall',round(wall,2),flush=True)
 if proc.returncode or reason or not matches: receipt['status']='rejected_incomplete_run';save();raise SystemExit(1)
reference=pathlib.Path(receipt['runs'][0]['stdout']).read_bytes()
receipt['all_output_text_identical']=all(pathlib.Path(r['stdout']).read_bytes()==reference for r in receipt['runs']) if len(receipt['runs']) > 1 else None
receipt['all_prompt_counts_identical']=len({r['prompt_tokens'] for r in receipt['runs']})==1
receipt['all_emitted_counts_identical']=len({r['new_tokens'] for r in receipt['runs']})==1
receipt['status']='completed_measurement_not_automatic_promotion'
receipt['claim_boundary']='Timing and exact UTF-8 output comparison on the stated workload. Turbo does not export token IDs; text equality is not asserted as a token-ID oracle. Routed ANE arms evaluate every top-8 expert during prefill using runtime-input weights; shared ANE arms use immutable FFNs. Metal executes companion stages. Inspect each profile for actual placement and invocation counts. Numerical equality is measured separately, not assumed.'
save(); print('TEXT_EQUAL',receipt['all_output_text_identical'],flush=True)
