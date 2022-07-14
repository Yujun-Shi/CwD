device_id=7
SEED=1

bz=128
lr=0.1
mom=0.9
wd=1e-4
data=imagenet_100
network=resnet18
nepochs=90

appr=joint

nc_first=10
ntask=10

CUDA_VISIBLE_DEVICES=$device_id python3 main_incremental.py --exp-name nc_first_${nc_first}_ntask_${ntask} \
     --datasets $data --num-tasks $ntask --nc-first-task $nc_first --network $network --seed $SEED \
     --nepochs $nepochs --batch-size $bz --lr $lr --momentum $mom --weight-decay $wd --decay-mile-stone 30 60 \
     --clipping -1 --results-path results --save-models \
     --approach $appr


