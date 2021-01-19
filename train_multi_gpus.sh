# --nproc_per_node=NUM_GPUS_YOU_HAVE

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2  --master_port 29501  train.py \
--mgpus \
--batch_size 5 \
--num_workers 3 \
--num_epochs 60 \
--split full \
--data_path /data/KITTI_DAT
