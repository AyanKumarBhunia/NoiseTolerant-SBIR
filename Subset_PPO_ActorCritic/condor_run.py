import os
from condor import condor, Job, Configuration
from datetime import datetime
# Provide required configuration of machine
# Condor commands:
# condor_q = query status of submitted jobs.
# condor_rm <job_id> = remove job
# ssh_to_job <job_id>



conf = Configuration(universe='docker',  # OR 'vanilla'
                     docker_image= 'registry.eps.surrey.ac.uk/deeprepo:master',
                     request_CPUs=8,
                     request_memory=1024*32,
                     request_GPUs=1,
                     gpu_memory_range=[10000, 24000],
                     cuda_capability=6.5)
# This is the (example) job to be submitted.
# python classifier.py --base ./ --root ${STORAGE}/datasets/quickdraw --batch_size 64 --n_classes 3 --epochs 5 --tag clsc3f7g10 --modelname clsc3f7g10


with condor('condor', project_space='ayanCV') as sess: #aisurrey-condor vs condor

    run_file = f'{os.getcwd()}/main.py'
    exp_name = 'FGSBIR_Subset_PPO'
    folder_name = '_'.join([exp_name, datetime.now().strftime("%b-%d_%H:%M:%S")])

    for bs in ['_']: # submit a bunch of jobs

        # It will autodetect the full path of your python executable
        j = Job('/vol/research/ayanCV/anaconda3/envs/ayanPY/bin/python', # if docker, use absolute path to specify executables inside container
            run_file,
            # all arguments to the executable should be in the dictionary as follows.
            # an entry 'epochs=30' in the dict will appear as 'python <file>.py --epochs 30'
            arguments=dict(
                base_dir=os.getcwd(),
                margin=0.2,
                batchsize=16,
                saved_models=os.path.join(os.getcwd(), f'./condor_output/{folder_name}'),
                reward='rank1'
                # topn=bs
                #root=os.environ['STORAGE'] + '/datasets/quickdraw',
                #batch_size=bs, # Here's the looped variable 'bs'
                #n_classes=3,
                #epochs=30,
                #modelname='clsc3f7g10'
                #splitTrain = 0.7
            ),
            stream_output=True,
            can_checkpoint=True,
            approx_runtime=8,  # in hours
            tag=exp_name,
            artifact_dir=f'./condor_output/{folder_name}'
        )
        # finally submit it
        sess.submit(j, conf)




