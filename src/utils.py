import os
import sys
from subprocess import run
from textwrap import dedent

def submit_pbs(cfg):
    
    args = sys.argv
    
    if '--multirun' in args:
        raise NotImplementedError
        args.remove('multirun')
        # it's also necessary to replace multirun args like param=1,2,3
        exec_cmd = ' '.join(args)
    else:
        device = cfg.device or r'\`tail -c 2 \$PBS_GPUFILE\`'
        exec_cmd = dedent(f"""
            qsub <<EOT
            #PBS -d {os.path.dirname(__file__)}
            #PBS -N {cfg.run_name}
            #PBS -q {cfg.cluster.queue}
            #PBS -l nodes=1:ppn={cfg.cluster.procs}:gpus={cfg.cluster.gpus}:{cfg.cluster.queue}
            #PBS -l mem={cfg.cluster.mem},walltime={cfg.cluster.time}
            source ~/setup_dl.sh
            export CUDA_VISIBLE_DEVICES={device}
            {' '.join(args)} submit=False
            exit 0
            EOT
        """)
        
    if not cfg.dry_run:
        run(exec_cmd, shell=True, executable='/bin/bash')

    return exec_cmd
    
def ensure_device(x, device):
    """Recursively send tensors within nested structure to device"""
    if isinstance(x, list):
        return [ensure_device(e, device) for e in x]
    else:
        return x.to(device)    