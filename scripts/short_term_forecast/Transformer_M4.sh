export CUDA_VISIBLE_DEVICES=1

model_name=Transformer

python -u run.py 
  --task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Monthly' 
  --model_id m4_Monthly 
  --model Transformer
  --data m4 
  --features M 
  --e_layers 2 
  --d_layers 1 
  --factor 3 
  --enc_in 1 
  --dec_in 1 
  --c_out 1 
  --batch_size 16 
  --d_model 512 
  --des 'Exp' 
  --itr 1 
  --learning_rate 0.001 
  --loss 'SMAPE'

python -u run.py 
  --task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Yearly' 
  --model_id m4_Yearly 
  --model Transformer
  --data m4 
  --features M 
  --e_layers 2 
  --d_layers 1 
  --factor 3 
  --enc_in 1 
  --dec_in 1 
  --c_out 1 
  --batch_size 16 
  --d_model 512 
  --des 'Exp' 
  --itr 1 
  --learning_rate 0.001 
  --loss 'SMAPE'

python -u run.py 
  --task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Quarterly' 
  --model_id m4_Quarterly 
  --model Transformer
  --data m4 
  --features M 
  --e_layers 2 
  --d_layers 1 
  --factor 3 
  --enc_in 1 
  --dec_in 1 
  --c_out 1 
  --batch_size 16 
  --d_model 512 
  --des 'Exp' 
  --itr 1 
  --learning_rate 0.001 
  --loss 'SMAPE'

python -u run.py 
  --task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Weekly' 
  --model_id m4_Weekly 
  --model Transformer
  --data m4 
  --features M 
  --e_layers 2 
  --d_layers 1 
  --factor 3 
  --enc_in 1 
  --dec_in 1 
  --c_out 1 
  --batch_size 16 
  --d_model 512 
  --des 'Exp' 
  --itr 1 
  --learning_rate 0.001 
  --loss 'SMAPE'

python -u run.py 
  --task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Daily' 
  --model_id m4_Daily 
  --model Transformer
  --data m4 
  --features M 
  --e_layers 2 
  --d_layers 1 
  --factor 3 
  --enc_in 1 
  --dec_in 1 
  --c_out 1 
  --batch_size 16 
  --d_model 512 
  --des 'Exp' 
  --itr 1 
  --learning_rate 0.001 
  --loss 'SMAPE'

python -u run.py 
  --task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Hourly' 
  --model_id m4_Hourly 
  --model Transformer
  --data m4 
  --features M 
  --e_layers 2 
  --d_layers 1 
  --factor 3 
  --enc_in 1 
  --dec_in 1 
  --c_out 1 
  --batch_size 16 
  --d_model 512 
  --des 'Exp' 
  --itr 1 
  --learning_rate 0.001 
  --loss 'SMAPE'