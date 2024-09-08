import sys
from subprocess import check_output
from textwrap import dedent

def submit(cfg, hcfg, exp_dir, log):

    ccfg = cfg.cluster
    out_dir = hcfg.runtime.output_dir
    
    overrides = list(hcfg.overrides.task)
    overrides.remove('submit=True')
    overrides.append(f'hydra.run.dir={out_dir}')

    match cfg.cluster.scheduler:
        case 'pbs':
            exec_cmd, jobid = submit_pbs(cfg, ccfg, hcfg, overrides, out_dir)
        case 'moab':
            exec_cmd, jobid  = submit_moab(cfg, hcfg, exp_dir)
        case _:
            log.error(f'Unknown cluster scheduler "{cfg.cluster.scheduler}"')
            sys.exit()
    
    log.info(f'Executing in shell: {exec_cmd}')
    if jobid:
        log.info(f'Submitted job: {jobid}')

def submit_pbs(cfg, ccfg, hcfg, overrides, out_dir):
    
    device = cfg.device or r'\`tail -c 2 \$PBS_GPUFILE\`'
    dependency = f'#PBS -W depend={ccfg.dependency}' if ccfg.dependency else ''
    exec_cmd = dedent(f"""
        qsub <<EOT
        #PBS -N {cfg.run_name}
        #PBS -q {ccfg.queue}
        #PBS -l nodes={ccfg.node}:ppn={ccfg.procs or 1}:gpus={ccfg.num_gpus}:{ccfg.queue}
        #PBS -l walltime={ccfg.time},mem={ccfg.mem},vmem={ccfg.vmem}
        #PBS -o {out_dir}/pbs.log
        #PBS -j oe
        {dependency}
        cd {cfg.proj_dir}
        source setup.sh
        export CUDA_VISIBLE_DEVICES={device}
        python main.py -cn {hcfg.job.config_name} {' '.join(overrides)} 
        exit 0
        EOT
    """)
        
    jobid = None
    if not cfg.dry_run:
        jobid = check_output(exec_cmd, shell=True, executable='/bin/bash')
        jobid = jobid.decode().strip()

    return exec_cmd, jobid

def submit_moab(cfg, hcfg, exp_dir):
    
    ccfg = cfg.cluster
    out_dir = hcfg.runtime.output_dir
    
    overrides = list(hcfg.overrides.task)
    overrides.remove('submit=True')
    overrides.append(f'hydra.run.dir={out_dir}')

    exec_cmd = dedent(f"""
        msub <<EOT
        #MSUB -N {cfg.run_name}
        #MSUB -l nodes=1:ppn={ccfg.procs}:gpus={ccfg.num_gpus},feature={ccfg.feature},pmem={ccfg.pmem},walltime={ccfg.time}
        #MSUB -o {out_dir}/moab.log
        #MSUB -j oe
	cd {cfg.proj_dir}
        source setup.sh
        python main.py {' '.join(overrides)} -cn {hcfg.job.config_name}
        exit 0
        EOT
    """)
        
    jobid = None
    if not cfg.dry_run:
        jobid = check_output(exec_cmd, shell=True, executable='/bin/bash')
        jobid = jobid.decode().strip()

    return exec_cmd, jobid
