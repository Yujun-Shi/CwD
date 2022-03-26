device_id=0
SEED=0
bz=128
lr=0.1
mom=0.9
wd=1e-4
data=imagenet_100
network=resnet18
nepochs=90
n_exemplar=20

appr=lucir_cwd
lamb=10.0
nc_first=50
ntask=6

aux_coef=0.75
rej_thresh=1
first_task_lr=0.2
first_task_bz=128

CUDA_VISIBLE_DEVICES=$device_id python3 main_incremental.py --exp-name nc_first_${nc_first}_ntask_${ntask} \
         --datasets $data --num-tasks $ntask --nc-first-task $nc_first --network $network --seed $SEED \
         --nepochs $nepochs --batch-size $bz --lr $lr --momentum $mom --weight-decay $wd --decay-mile-stone 30 60 \
         --clipping -1 --results-path results --save-models \
         --approach $appr --lamb $lamb --num-exemplars-per-class $n_exemplar --exemplar-selection herding \
         --aux-coef $aux_coef --reject-threshold $rej_thresh \
         --first-task-lr $first_task_lr --first-task-bz $first_task_bz

CUDA_VISIBLE_DEVICES=$device_id python3 occ.py
