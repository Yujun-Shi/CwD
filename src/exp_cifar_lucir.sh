device_id=0
SEED=0
bz=64
lr=0.1
mom=0.9
wd=5e-4
data=cifar100_icarl
network=resnet18_cifar
nepochs=160

appr=lucir
lamb=5.0
nc_first=50
ntask=2

first_task_bz=128
first_task_lr=0.1


CUDA_VISIBLE_DEVICES=$device_id python3 main_incremental.py --exp-name ${nc_first}_${ntask}_${SEED} \
     --datasets $data --num-tasks $ntask --nc-first-task $nc_first --network $network --seed $SEED \
     --nepochs $nepochs --batch-size $bz --lr $lr --momentum $mom --weight-decay $wd --decay-mile-stone 80 120 \
     --clipping -1 --results-path results --save-models \
     --approach $appr --lamb $lamb --first-task-bz $first_task_bz --first-task-lr $first_task_lr \
     --num-exemplars-per-class 20 --exemplar-selection herding

CUDA_VISIBLE_DEVICES=$device_id python3 occ.py

