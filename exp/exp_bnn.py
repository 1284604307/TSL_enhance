
import datetime

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils import drawUtil
from utils.drawUtil import getBaseOutputPath
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_BNN(Exp_Basic):
    def __init__(self, args):
        super(Exp_BNN, self).__init__(args)
        print("Exp_BNN 暂不支持自定义损失函数")
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # imputation
                outputs = self.model(batch_x)
                if len(batch_y.shape) < len(outputs.shape):
                    # 补充一层，使模型输出shape = y shape
                    batch_y = batch_y.unsqueeze(1)
                # todo 多对多预测，调换预测的通道次序
                # if(self.args.features.endswith("M")):
                #     batch_y = batch_y.permute(0, 2, 1)
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def gaussian_log_likelihood(self,targets, pred_means, pred_stds=None):
        deltas = pred_means - targets
        if pred_stds is not None:
            lml = -((deltas / pred_stds) ** 2).sum(-1) * 0.5 \
                  - pred_stds.log().sum(-1) \
                  - np.log(2 * np.pi) * 0.5
        else:
            lml = -(deltas ** 2).sum(-1) * 0.5

        return lml

    def trainOne(self, train_loader):
        self.model.train()
        iter_count = 0
        train_losses = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            self.model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            # 还有一个参数
            outputs = self.model(batch_x, resample=True)
            # 移除值为1的维度
            if(len(outputs.shape)>2):
                outputs = torch.squeeze(outputs)
            batch_y = batch_y.float().to(self.device)

            # BNN 对分布平均计算损失
            # mean, log_std = outputs.split([batch_y.shape[1], batch_y.shape[1]], dim=-1)
            mean, log_std = outputs.split([48,48], dim=-1)

            loss = (-self.gaussian_log_likelihood(batch_y, mean, log_std.exp())
                    + 1e-2 *  self.model.regularization()).mean()

            # loss = self.criterion(outputs, batch_y)
            train_loss = loss.item()
            train_losses.append(train_loss)
            loss.backward()

            self.model_optim.step()
        return train_losses

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        input_xs = []
        folder_path = drawUtil.getBaseOutputPath(self.args) + 'test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # imputation
                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                if len(batch_y.shape) < len(outputs.shape):
                    # 补充一层，使模型输出shape = y shape
                    outputs = outputs.squeeze()
                    # batch_y = batch_y.unsqueeze()
                pred = outputs
                true = batch_y.cpu().numpy()


                input_xs.append(batch_x.cpu().numpy())
                preds.append(pred)
                trues.append(true)
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        input_xs = np.concatenate(input_xs, 0)

        print('test shape:', preds.shape, trues.shape)
        preds = np.reshape(preds, (preds.shape[0], -1))
        trues = np.reshape(trues, (trues.shape[0], -1))

        # result save
        folder_path = drawUtil.getBaseOutputPath(self.args, setting)

        drawUtil.drawResultCompare(
            result=preds,
            real=trues,
            tag=self.args.model,
            savePath=folder_path + '归一化预测对比',
            args=self.args
        )
        drawUtil.completeMSE(preds, trues)
        # drawUtil.metricAndSave(preds, trues, folder_path)
        drawUtil.saveResultCompare(preds, trues, drawUtil.getBaseOutputPath(self.args, setting))
        drawUtil.drawResultSample(input_data  = input_xs,pred = preds,real=trues,args=self.args)
        print("\n数据反归一化处理...")
        # if(len(preds.shape)==2):
        #     for i in range(preds.shape[1]):
        preds = test_data.labelScaler.inverse_transform(np.array(preds))
        trues = test_data.labelScaler.inverse_transform(np.array(trues))

        drawUtil.drawResultCompare(result=preds, real=trues, tag=self.args.model,
                                   savePath=folder_path + '反归一化后预测对比', )
        drawUtil.metricAndSave(preds=preds, trues=trues, folder_path=drawUtil.getBaseOutputPath(self.args, setting))
        drawUtil.saveResultCompare(preds, trues, drawUtil.getBaseOutputPath(self.args, setting))

        return

    def getOnnxModel(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        with torch.no_grad():
            iter_count = 0
            for i, (batch_x, batch_y) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_data = batch_x
                input_data = input_data.to(device)
                torch.onnx.export(self.model, input_data, f=getBaseOutputPath(args=self.args, setting=setting)
                                                            + 'cross.onnx')
                break
