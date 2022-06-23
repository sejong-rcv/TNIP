OMP_NUM_THREADS=1 python tnip.py \
    --dataset fivr5k \
    --data_root_path /path/to/raw_video_folder/ \
    --keyframe_path keyframes \
    --backbone R2_1D \
    --num_worker 1 \
    --temporal_flip \
    --res 112 \
    --gpu "0" \
    --ext_type "keyframe" \
    --clip_interval 2 \
    --save_path base
