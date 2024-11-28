argsContent  = '''--model HSFE_GRU_Informer_DSA --learning_rate 0.001 --date_column timestamp --is_training 1 
--root_path D:\深度学习\代码\A数据集/ --data_path pv_data.xlsx --model_id TimesNet_96_96 
--seq_len 96 --label_len 48 --pred_len 1 --e_layers 2 --d_layers 1 --factor 3 
--d_model 16 --d_ff 32 --des Exp --itr 1 --top_k 5 
--task_name conv --data conv_ETTh1 --features MS --target 发电功率'''

argsContent = argsContent.replace("\n","")
argSplit = argsContent.split(" ")
argObjet = {}
for i in range(int(len(argSplit)/2)):
    argObjet[argSplit[i*2].replace("--","")] = argSplit[i*2+1]

argObjet['model'] = "TCN-effKAN"
# 日期列
argObjet['date_column'] = "Sequence"
# 预测目标列
argObjet['target'] = "y"
#输入特征数量
argObjet['enc_in'] = "6"
argObjet['dec_in'] = argObjet['enc_in']
# 输出特征数量
argObjet['c_out'] = "1"
argObjet['root_path'] = "/kaggle/input/zenodo5516552-wt1"
argObjet['data_path'] = "Inland Wind Farm Dataset1(WT1).csv"
argObjet['result_rpath'] = "/kaggle/working"
argObjet['model_id'] = "["+argObjet['data_path'].replace(" ","-").split(".")[0]+"]_"+argObjet['model']+"_"+argObjet['task_name']+"_"+argObjet['seq_len']+"_"+argObjet['pred_len']
argObjet['num_workers'] = "10"

args = []
for key,value in argObjet.items():
    args.append(f"--{key}")
    args.append(value)
print("参数列")
print(args)

print("ide 配置参数列")

argObjet['root_path'] = "C:\dev\深度学习\数据集X"
argObjet['data_path'] = "Inland_Wind_Farm_Dataset1.csv"
argsStr = ""
for key,value in argObjet.items():
    argsStr+=(f" --{key} {value}")
print(argsStr)