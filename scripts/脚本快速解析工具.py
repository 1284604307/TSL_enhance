from scripts import 参数集

argsContent = '''--task_name short_term_forecast 
  --is_training 1 
  --root_path ./dataset/m4 
  --seasonal_patterns 'Monthly' 
  --model_id m4_Monthly 
  --model $model_name 
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
  --des Exp
  --itr 1 
  --learning_rate 0.001 
  --loss SMAPE'''

argsContent = argsContent.replace("\n", "")
argSplit = argsContent.split(" ")
argObjet = {}
for i in range(int(len(argSplit) / 2)):
    argObjet[argSplit[i * 2].replace("--", "")] = argSplit[i * 2 + 1]
obj = 参数集.Inland_Wind_Farm_Dataset['Transformer']
for key in obj:
    argObjet[key] = obj[key]


argObjet['dec_in'] = argObjet['enc_in']
# 输出特征数量
argObjet['model_id'] = "[" + argObjet['data_path'].replace(" ", "-").split(".")[0] + "]_" + argObjet['model'] + "_" + \
                       argObjet['task_name'] + "_" + argObjet['seq_len'] + "_" + argObjet['pred_len']


args = []
for key, value in argObjet.items():
    args.append(f"--{key}")
    args.append(value)

print("参数列")
print(args)
print(argObjet)

print("ide 配置参数列")

argObjet['root_path'] = "D:\深度学习\代码\A数据集\AEMO澳大利亚能源市场运营商电力需求数据集"
argObjet['data_path'] = "AEMO_QLD_electricity.csv"
argObjet['data_path'] = "AEMO_QLD_electricity.csv"
argsStr = ""

for key, value in argObjet.items():
    argsStr += (f" --{key} {value}")
print(argsStr)
