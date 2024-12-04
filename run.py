import torch
import random
import numpy as np
import argsUtil
from datetime import datetime, timezone, timedelta

from utils import drawUtil
from utils.drawUtil import getBaseOutputPath

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = argsUtil.getArgsParser().parse_args()
    args = argsUtil.processAndPrintArgs(args)
    Exp = argsUtil.getExp(args)

    beijing = timezone(timedelta(hours=8))
    start_time = datetime.now().astimezone(beijing)
    print(f"训练开始时间：{start_time}")

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = argsUtil.getSettingsStr(args,ii)
            print(f"保存配置信息 -> {setting}")
            drawUtil.saveTxt(drawUtil.getBaseOutputPath(args)+'results/' + setting + f'/参数配置_itr{ii}.txt', argsUtil.args2txt(args))

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = argsUtil.getSettingsStr(args,ii)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    end_time = datetime.now().astimezone(beijing)
    print(f"训练开始时间：{start_time} , 结束时间:{end_time}")