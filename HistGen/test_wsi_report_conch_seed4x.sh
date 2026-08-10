model='histgen'
max_length=100
epochs=40
region_size=96
prototype_num=512

python /home/woody/iwi5/iwi5204h/HistGen/main_test_AllinOne_OG_plus.py \
    --image_dir /home/woody/iwi5/iwi5204h/CLAM/CONCH_1_5_features/pt_files/ \
    --ann_path /home/woody/iwi5/iwi5204h/HistGen/Data/Label/train_val_test.json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --threshold 1 \
    --batch_size 1 \
    --epochs $epochs \
    --step_size 10 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /home/woody/iwi5/iwi5204h/HistGen/Data/TestResult_conch/seed42/Best_38_seed47/ \
    --gamma 0.8 \
    --seed 46 \
    --log_period 1000 \
    --load /home/woody/iwi5/iwi5204h/HistGen/Data/TrainingResult_Conch_17_seed42/model_best_epoch38.pth \
    --beam_size 3 \
    --d_vf 1536 