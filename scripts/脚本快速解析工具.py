from scripts import 参数集

argsContent = '''--task_name short_term_forecast --loss SMAPE --scale 1 --is_training 1 --root_path D:\深度学习\代码\A数据集\AEMO澳大利亚能源市场运营商电力需求数据集 --seasonal_patterns Monthly --model_id [QLD]_Transformer_short_term_forecast_96_1 --model Transformer --data former --features MS --e_layers 2 --d_layers 1 --factor 6 --enc_in 11 --dec_in 11 --c_out 1 --batch_size 128 --d_model 16 --des Exp --learning_rate 0.001 --date_column SETTLEMENTDATE --target TOTALDEMAND --ignore_columns REGION,PERIODTYPE --data_path AEMO_QLD_electricity.xlsx --seq_len 96 --label_len 48 --pred_len 1 --result_rpath /kaggle/working --num_workers 1'''

argsContent = argsContent.replace("\n", "")
argSplit = argsContent.split(" ")
argObjet = {}
for i in range(int(len(argSplit) / 2)):
    argObjet[argSplit[i * 2].replace("--", "")] = argSplit[i * 2 + 1]

# obj = 参数集.Inland_Wind_Farm_Dataset['Transformer']
# for key in obj:
#     argObjet[key] = obj[key]


argObjet['dec_in'] = argObjet['enc_in']
# 输出特征数量


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
