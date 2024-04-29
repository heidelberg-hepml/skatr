import sys
from subprocess import run
from textwrap import dedent

def submit_pbs(cfg, hcfg, exp_dir):
    
    overrides = list(hcfg.overrides.task)
    overrides.remove('submit=True')
    overrides.append(f'hydra.run.dir={exp_dir}')
    device = cfg.device or r'\`tail -c 2 \$PBS_GPUFILE\`'
    exec_cmd = dedent(f"""
        qsub <<EOT
        #PBS -N {cfg.run_name}
        #PBS -q {cfg.cluster.queue}
        #PBS -l nodes={cfg.cluster.node}:ppn={cfg.cluster.procs}:gpus={cfg.cluster.num_gpus}:{cfg.cluster.queue}
        #PBS -l mem={cfg.cluster.mem},walltime={cfg.cluster.time}
        #PBS -o {exp_dir}/pbs.log
        #PBS -j oe
        cd {cfg.proj_dir}
        source setup.sh
        export CUDA_VISIBLE_DEVICES={device}
        python main.py {' '.join(overrides)} -cn {hcfg.job.config_name}
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
