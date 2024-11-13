argsContent  = '''--root_path
D:\深度学习\代码\A数据集\AEMO澳大利亚能源市场运营商电力需求数据集/
--data_path
AEMO_QLD_electricity_processed.csv
--ignore_columns
REGION,PERIODTYPE,SETTLEMENTDATE
--date_column
SETTLEMENTDATE
--target
TOTALDEMAND
--model
TCN-effKAN
--task_name
conv_ott
--is_training
1
--model_id
rlData_96_1
--data
conv_OnlyTrainTest
--features
MS
--batch_size
128
--seq_len
96
--pred_len
1
--enc_in
18
--c_out
1
--des
Exp
--itr
1
--num_channels
48,96,48
--learning_rate
0.001
--train_epochs
10'''

argSplit = argsContent.split("\n")
argObjet = {}
for i in range(int(len(argSplit)/2)):
    argObjet[argSplit[i*2].replace("--","")] = argSplit[i*2+1]
argObjet['root_path'] = "/kaggle/input/aemo-electricity/"
argObjet['data_path'] = "AEMO_QLD_electricity_processed.csv"
argObjet['result_rpath'] = "/kaggle/working"
argObjet['model_id'] = "["+argObjet['data_path'].split(".")[0]+"]_"+argObjet['model']+"_"+argObjet['task_name']+"_"+argObjet['seq_len']+"_"+argObjet['pred_len']

args = []
for key,value in argObjet.items():
    args.append(f"--{key}")
    args.append(value)
print(args)
