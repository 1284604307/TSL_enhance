import os
import sys
import time

import netron
import torch
import random
import numpy as np
import argsUtil
from datetime import datetime, timezone, timedelta

from utils import drawUtil
from utils.drawUtil import getBaseOutputPath
from utils.tools import EarlyStopping, adjust_learning_rate

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if(sys.platform.startswith('win')):
        print("Windows 环境")

        args_cmd = argsUtil.getWinScriptParser().parse_args()
        # file_path = 'scripts/myscripts/windows/TCN-fftKAN_比利时_96_16.参数配置'
        file_path = 'scripts/myscripts/windows/'+args_cmd.script
        try:
            # 以只读模式打开文件
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                contents = content.split("python -u run.py")
                vars = contents[0].split("\n")
                itemVars = {}
                for item in vars:
                    itemVar = item.split("=")
                    # itemVars[itemVar[0]] = itemVar=[1]
                    if(len(itemVar) == 2):
                        contents[1] = contents[1].replace("%"+itemVar[0]+"%", itemVar[1])
                    else:
                        print("异常参数=>"+str(itemVar))
                contents[1] = contents[1].strip('\n').split("\n")
                args = []
                for item in contents[1]:
                    print(item)
                    escapeIndex = item.index(" ")
                    args.append(item[:escapeIndex].strip())
                    args.append(item[escapeIndex+1:].strip().strip("\""))
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到")
            exit(0)
        args = argsUtil.getArgsParser().parse_args(args)
    else:
        args = argsUtil.getArgsParser().parse_args()
    args = argsUtil.processAndPrintArgs(args)
    Exp = argsUtil.getExp(args)

    beijing = timezone(timedelta(hours=8))
    start_time = datetime.now().astimezone(beijing)
    print(f"开始时间：{start_time}")

    if args.is_training:
        for ii in range(args.itr):
            setting = argsUtil.getSettingsStr(args,ii)
            args.setting = setting
            # setting record of experiments
            exp = Exp(args)  # set experiments
            print(f"保存配置信息 -> {setting}")
            drawUtil.saveTxt(drawUtil.getBaseOutputPath(args)+'results/' + setting + f'/参数配置_itr{ii}.txt', argsUtil.args2txt(args))

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            train_data, train_loader = exp._get_data(flag='train')
            vali_data, vali_loader = exp._get_data(flag='val')
            test_data, test_loader = exp._get_data(flag='test')

            path = os.path.join(args.checkpoints, setting)
            if not os.path.exists(path):
                os.makedirs(path)
            train_steps = len(train_loader)
            time_now = time.time()
            completeEarlyStop = args.patience != 0
            if(completeEarlyStop):
                early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            for epoch in range(args.train_epochs):
                epoch_time = time.time()
                # todo 调用exp 训练
                train_loss = exp.trainOne(train_loader)

                print(f"Epoch: {epoch + 1} cost time: {(time.time() - epoch_time):.2f} | Train Loss: {np.average(train_loss):.7f}")

                if(completeEarlyStop):
                    print("计算vali_loss，test_loss以判断是否早停...")
                    # todo 调用exp 获取验证及测试结果
                    vali_loss = exp.vali(vali_data, vali_loader)
                    test_loss = exp.vali(test_data, test_loader)

                    print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {np.average(train_loss):.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
                    early_stopping(vali_loss, exp.model, path)
                    if early_stopping.early_stop:
                        print("Early  stopping")
                        break


                adjust_learning_rate(exp.model_optim, epoch + 1, args)


            if args.save_model:
                try:
                    best_model_path = path + '/' + 'checkpoint.pth'
                    exp.model.load_state_dict(torch.load(best_model_path))
                    print("模型已保存")
                except FileNotFoundError:
                    print("模型保存失败，未找到模型保存文件夹")

            # model = exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = argsUtil.getSettingsStr(args,ii)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        exp.getOnnxModel(setting)
        netron.start(getBaseOutputPath(args=args,setting=setting)+'cross.onnx')


    end_time = datetime.now().astimezone(beijing)
    print(f"开始时间：{start_time} , 结束时间:{end_time}")
