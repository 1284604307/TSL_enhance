argsContent  = '''--model TCN --date_column timestamp --is_training 1 --root_path D:\深度学习\代码\A数据集/ --data_path pv_data.xlsx --model_id TimesNet_96_96 --seq_len 96 --label_len 48 --pred_len 1 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 1 --d_model 16 --d_ff 32 --des Exp --itr 1 --top_k 5 --task_name conv --data conv_ETTh1 --features MS --target 发电功率'''

argSplit = argsContent.split(" ")
argObjet = {}
for i in range(int(len(argSplit)/2)):
    argObjet[argSplit[i*2].replace("--","")] = argSplit[i*2+1]

rootPath = "/kaggle/input/pv-data/"
dataPath = "pv_data.xlsx"

argObjet['root_path'] = "/kaggle/input/pv-data/"
argObjet['data_path'] = "pv_data.xlsx"
argObjet['result_rpath'] = "/kaggle/working"
argObjet['model_id'] = "["+argObjet['data_path'].split(".")[0]+"]_"+argObjet['model']+"_"+argObjet['task_name']+"_"+argObjet['seq_len']+"_"+argObjet['pred_len']
argObjet['num_workers'] = "10"

args = []
for key,value in argObjet.items():
    args.append(f"--{key}")
    args.append(value)

print(args)
