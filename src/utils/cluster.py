import sys
from subprocess import run
from textwrap import dedent

def submit(cfg, hcfg, exp_dir, log):
    match cfg.cluster.scheduler:
        case 'pbs':
            exec_cmd = submit_pbs(cfg, hcfg, exp_dir)
            log.info(f'Executing in shell: {exec_cmd}')
        case _:
            log.error(f'Unknown cluster scheduler "{cfg.cluster.scheduler}"')
            sys.exit()

def submit_pbs(cfg, hcfg, exp_dir):
    
    ccfg = cfg.cluster
    overrides = list(hcfg.overrides.task)
    overrides.remove('submit=True')
    overrides.append(f'hydra.run.dir={exp_dir}')
    device = cfg.device or r'\`tail -c 2 \$PBS_GPUFILE\`'

    exec_cmd = dedent(f"""
        qsub <<EOT
        #PBS -N {cfg.run_name}
        #PBS -q {ccfg.queue}
        #PBS -l nodes={ccfg.node}:ppn={ccfg.procs}:gpus={ccfg.num_gpus}:{ccfg.queue}
        #PBS -l mem={ccfg.mem},walltime={ccfg.time}
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