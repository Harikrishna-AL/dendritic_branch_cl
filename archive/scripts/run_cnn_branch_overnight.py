"""Sequential, resumable overnight launcher for matched small-CNN experiments."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cnn_branch_comparison_20261027'
ACTIVE=None


def save(path,obj):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(obj,indent=2));temporary.replace(path)


def interrupt(signum,frame):
    global ACTIVE
    if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.terminate()
    raise KeyboardInterrupt


def main():
    global ACTIVE
    parser=argparse.ArgumentParser();parser.add_argument('--feature-epochs',type=int,default=20);parser.add_argument('--ood-epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=64);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    save(OUT/'process.json',{'pid':os.getpid(),'started_unix':time.time(),'python':sys.executable})
    worker=OUT/'worker_source.py'
    # Preserve the exact experimental implementation across future chat edits.
    if not worker.exists():shutil.copyfile(ROOT/'bdh_cl/scripts/cnn_branch_comparison.py',worker)
    shutil.copyfile(Path(__file__),OUT/'overnight_driver_source.py')
    spec=importlib.util.spec_from_file_location('frozen_cnn_comparison',worker);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source_hash=hashlib.sha256(worker.read_bytes()).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'external/DCNet',text=True).strip()
    (OUT/'upstream_portability.patch').write_text(subprocess.check_output(['git','diff'],cwd=ROOT/'external/DCNet',text=True))
    # Verify the existing archive before any full training launches.
    checksum=hashlib.md5()
    with (ROOT/'data_cache/cifar100/cifar-100-python.tar.gz').open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):checksum.update(chunk)
    assert checksum.hexdigest()=='eb9058c3a382ffc7106e4002c42a8d85'
    profile=ROOT/'results/cnn_branch_profile_final_20261027/cnn32_dcnet_hat_seed701/profile.json'
    profile_data=json.loads(profile.read_text())
    assert profile_data['device']=='mps' and profile_data['process_peak_RSS_bytes']<8_000_000_000
    checks={}
    for name in ['cnn16_branch_config_seed701','cnn32_dcnet_hat_seed701']:
        folder=ROOT/'results/cnn_branch_smoke_ready_20261027'/name
        smoke=json.loads((folder/'smoke_complete.json').read_text());assert smoke['passed']
        checks[name]=smoke
    save(OUT/'preflight.json',{'smoke_checks':checks,'largest_CNN_profile':profile_data,'dataset_MD5':checksum.hexdigest()})
    queue=[{'size':size,'arm':arm,'seed':seed} for seed in [701,702,703] for size in ['cnn16','cnn32'] for arm in ['branch_config','dcnet_hat']]
    protocol={'seeds':[701,702,703],'sizes':{'cnn16':[16,32,64],'cnn32':[32,64,128]},'arms':['branch_config','dcnet_hat'],'feature_epochs':args.feature_epochs,'ood_epochs':args.ood_epochs,'batch_size':args.batch_size,'warmup_epochs':min(3,max(1,args.feature_epochs//5)),'DAC_epoch':max(1,round(args.feature_epochs*400/700)),'fit_images':45000,'unused_validation_images':5000,'test_images':10000,'class_order_seed':2026,'class_order':module.np.random.default_rng(2026).permutation(100).tolist(),'task_split':'10x10 disjoint, same as earlier pilot','no_replay':True,'supplied_inference_task_ID':False,'external_pretraining':False,'CNN_trainable_during_representation_phase':True,'CNN_fixed_during_classifier_phase':True,'common_recipe':'Official DCNet IOE/DAC, 8 augmented views per representation update, 4 per classifier update, HAT/LARC protection, matching optimizer and data exposure across arms','primary_readout':'rotation_ensemble_joint_logits','secondary_readout':'rotation_ensemble_prototype_cosine; predeclared, no test selection','source_sha256':source_hash,'upstream_commit':commit,'queue':queue,'hardware':'Apple M4, 16 GiB unified memory, MPS','scope':'Matched small-CNN mechanism comparison; branch arm uses DCNet training/readout, not prior fixed-code raw-pixel recipe; no claim of published-DCNet reproduction'}
    if (OUT/'protocol.json').exists():
        old=json.loads((OUT/'protocol.json').read_text())
        for key in ['feature_epochs','ood_epochs','batch_size','source_sha256']:assert old[key]==protocol[key],f'Cannot change {key} while resuming'
    save(OUT/'protocol.json',protocol);save(OUT/'status.json',{'status':'running','queue_length':len(queue)});module.summary_report(OUT)
    completed=[];failed=[];environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT)+os.pathsep+environment.get('PYTHONPATH','');environment['PYTHONUNBUFFERED']='1'
    try:
        for item in queue:
            name=f"{item['size']}_{item['arm']}_seed{item['seed']}";folder=OUT/name
            if (folder/'result.json').exists():completed.append(name);continue
            if (folder/'failed.json').exists():(folder/'failed.json').rename(folder/f'previous_failure_{int(time.time())}.json')
            command=[sys.executable,str(worker),'--out',str(OUT),'--size',item['size'],'--arm',item['arm'],'--seed',str(item['seed']),'--feature-epochs',str(args.feature_epochs),'--ood-epochs',str(args.ood_epochs),'--batch-size',str(args.batch_size)]
            save(OUT/'status.json',{'status':'running','active':item,'completed':completed,'failed':failed});print('LAUNCH',name,flush=True)
            with (OUT/'logs'/f'{name}.log').open('a') as log:
                ACTIVE=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                subprocess.run(['taskpolicy','-B','-p',str(ACTIVE.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
                code=ACTIVE.wait()
            ACTIVE=None
            if code==0 and (folder/'result.json').exists():completed.append(name)
            else:
                failed.append(name)
                if not (folder/'failed.json').exists():module.save(folder/'failed.json',{'error':'Worker exited without completed result','exit_code':code})
            pairs={}
            for seed in [701,702,703]:
                for size in ['cnn16','cnn32']:
                    files=[OUT/f'{size}_{arm}_seed{seed}'/'run.json' for arm in ['branch_config','dcnet_hat']]
                    if all(p.exists() for p in files):
                        values=[json.loads(p.read_text())['initial_shared_sha256'] for p in files]
                        assert values[0]==values[1],f'Shared initialization mismatch: {size} {seed}'
                        pairs[f'{size}_seed{seed}']=True
            save(OUT/'shared_initialization_pairs.json',pairs);module.summary_report(OUT)
        save(OUT/'completion.json',{'finished':True,'all_successful':len(completed)==len(queue),'completed':completed,'failed':failed,'finished_unix':time.time()})
        save(OUT/'status.json',{'status':'finished','completed':completed,'failed':failed});module.summary_report(OUT)
        print('FINISHED',len(completed),'successful;',len(failed),'failed',flush=True)
    except KeyboardInterrupt:
        if ACTIVE is not None and ACTIVE.poll() is None:
            ACTIVE.terminate()
            try:ACTIVE.wait(timeout=15)
            except subprocess.TimeoutExpired:ACTIVE.kill();ACTIVE.wait()
        save(OUT/'status.json',{'status':'stopped','completed':completed,'failed':failed});module.summary_report(OUT)
        raise
    except BaseException as error:
        if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.terminate()
        save(OUT/'driver_failed.json',{'error':str(error),'traceback':traceback.format_exc()});module.summary_report(OUT)
        raise

if __name__=='__main__':main()
