import os
import sys

import torch
import models
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, TCN
from models.model2024 import TCN_effKan

import importlib
import os



class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'TCN': TCN,
            'TCN-effKAN': TCN_effKan,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer
        }
        self.load_models()
        # print(self.model_dict)
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)


    def load_models(self):
        """
        遍历指定文件夹，查找Python模块（.py文件）并尝试导入其中的类进行实例化
        """
        # 获取文件夹下所有文件和文件夹名称
        files = os.listdir("./models")
        sys.path.append("./models")
        for file in files:
            # 只处理.py文件，忽略其他文件类型和文件夹
            if file.endswith('.py'):
                module_name = file[:-3]  # 去掉.py后缀获取模块名
                if(module_name in self.model_dict.keys()):
                    continue
                try:
                    module = importlib.import_module(module_name)
                    if("Model" in vars(module)):
                        print(f"自动导入模型类{module_name}")
                        self.model_dict[module_name] = module.Model
                    # 遍历模块中的所有属性（包括类等）
                    # for name, obj in vars(module.Model).items():
                    #     if isinstance(obj, type):  # 判断是否是类
                    #         print(f"自动导入模型类{module_name}")
                    #         self.model_dict[module_name] = module
                except ImportError:
                    print(f"无法导入模块 {module_name}，可能存在依赖问题或代码错误")
                    continue


    def _build_model(self):
        raise NotImplementedError
        return None

    def getModel(self,model_name):
        return self.model_dict[model_name]

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
