export CUDA_VISIBLE_DEVICES=1

task_name=short_term_forecast
model_name=iTransformer
ignore_columns="Decremental bid Indicator,Region,Grid connection type,Resolution code,Offshore/onshore"
target="OT"
result_rpath=/kaggle/working
root_path=/kaggle/input/etth-small/ETT-small
data_path=ETTh1.csv
date_column=Datetime
data=former

python -u run.py \
--task_name $task_name \
--loss MSE \
--scale 1 \
--is_training 1 \
--root_path $root_path \
--seasonal_patterns Monthly \
--model $model_name \
--data $data \
--features MS \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 7 \
--dec_in 7 \
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
--seq_len 336 \
--label_len 96 \
--pred_len 12 \
--num_workers 1 \
--train_epochs 20
