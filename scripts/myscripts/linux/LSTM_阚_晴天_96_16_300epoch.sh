task_name=base
model_name=LSTM
ignore_columns=""
target="发电功率"
result_rpath=./result
root_path=./dataset/
data_path=last晴天.xlsx
date_column=date

python -u run.py
--task_name %task_name%
--loss MSE 
--scale 1 
--is_training 1 
--root_path %root_path%
--model %model_name% 
--data base
--features MS
--enc_in 7
--c_out 1
--batch_size 64
--learning_rate 0.0001
--date_column %date_column% 
--target "%target%" 
--ignore_columns %ignore_columns%
--result_rpath %result_rpath% 
--data_path %data_path% 
--seq_len 128
--label_len 128
--pred_len 1
--num_workers 1 
--train_epochs 300