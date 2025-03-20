export CUDA_VISIBLE_DEVICES=1

task_name=bnn
model_name=BNN
ignore_columns="Decremental bid Indicator,Region,Grid connection type,Resolution code,Offshore/onshore"
target="Measured & Upscaled"
result_rpath=/kaggle/working
root_path=/kaggle/input/all-data
data_path=20221201_20241201_Federal_utc.csv
date_column=Datetime
python -u run.py \
--task_name $task_name \
--loss MSE \
--scale 1 \
--is_training 1 \
--root_path $root_path \
--seasonal_patterns Monthly \
--model $model_name \
--data conv_ETTh1 \
--features MS \
--e_layers 2 \
--d_layers 1 \
--factor 6 \
--enc_in 15 \
--dec_in 15 \
--c_out 1 \
--batch_size 64 \
--d_model 16 \
--des Exp \
--learning_rate 0.001 \
--date_column $date_column \
--target "$target" \
--ignore_columns "$ignore_columns" \
--result_rpath $result_rpath \
--data_path $data_path \
--seq_len 96 \
--label_len 96 \
--pred_len 1 \
--num_workers 1 \
--train_epochs 300 \
--num_channels 48,48