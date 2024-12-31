import datetime

from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils import drawUtil
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


class Exp_Conv_OTT(Exp_Basic):
    def __init__(self, args):
        super(Exp_Conv_OTT, self).__init__(args)
        self.criterion = self._select_criterion()
        self.model_optim = self._select_optimizer()

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
                batch_x = batch_x.permute(0, 2, 1)
                # imputation
                outputs = self.model(batch_x)
                if len(batch_y.shape)<len(outputs.shape):
                    # 补充一层，使模型输出shape = y shape
                    batch_y = batch_y.unsqueeze(1)
                batch_y = batch_y.permute(0, 2, 1)
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            preds, trues = [],[]
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # 置换2，3，todo 应该是为了按通道训练
                batch_x = batch_x.permute(0, 2, 1)
                # todo 多对多预测，调换预测的通道次序
                if(self.args.features.endswith("M")):
                    batch_y = batch_y.permute(0, 2, 1)

                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                if len(batch_y.shape)<len(outputs.shape):
                    # 补充一层，使模型输出shape = y shape
                    batch_y = batch_y.unsqueeze(1)
                batch_y = batch_y.float().to(self.device)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if len(outputs.shape) >2:
                    preds = np.append(preds, outputs.detach()[:,0,0].reshape(-1).cpu().numpy())
                    trues = np.append(trues,batch_y.detach()[:,0,0].reshape(-1).cpu().numpy())
                else:
                    preds = np.append(preds, outputs.detach()[:,0].reshape(-1).cpu().numpy())
                    trues = np.append(trues,batch_y.detach()[:,0].reshape(-1).cpu().numpy())

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time} | Train Loss: {np.average(train_loss):.7f}")
            drawUtil.completeMSE(preds, trues)

        if (self.args.save_model):
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = drawUtil.getBaseOutputPath(self.args)+'test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.permute(0, 2, 1)
                # imputation
                outputs = self.model(batch_x)
                # outputs = outputs.detach().cpu().numpy()
                if len(batch_y.shape)<len(outputs.shape):
                    # 补充一层，使模型输出shape = y shape
                    batch_y = batch_y.unsqueeze(1)

                if len(outputs.shape) >2:
                    preds = np.append(preds, outputs.detach()[:,0,0].reshape(-1).cpu().numpy())
                    trues = np.append(trues, batch_y.detach()[:,0,0].reshape(-1).cpu().numpy())
                else:
                    preds = np.append(preds, outputs.detach()[:,0].reshape(-1).cpu().numpy())
                    trues = np.append(trues, batch_y.detach()[:,0].reshape(-1).cpu().numpy())
                # if i % 20 == 0:
                #     visual(true[0, :, -1], pred[0, :, -1], os.path.join(folder_path, str(i) + '.pdf'))

        # preds = np.concatenate(preds, 0)
        # trues = np.concatenate(trues, 0)
        print('test shape:', preds.shape, trues.shape)
        # preds = np.reshape(preds, (preds.shape[0], -1))
        # trues = np.reshape(trues, (trues.shape[0], -1))
        drawUtil.completeMSE(
            predicted=preds,
            real=trues,
        )
        # result save
        folder_path = drawUtil.getBaseOutputPath(self.args)+'results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = drawUtil.metricAndSave(preds, trues, folder_path)

        drawUtil.drawResultCompare(
            result=preds,
            real=trues,
            tag=f"{self.args.model_id} {mae*100:.4f}%",
            savePath=folder_path+f"{self.args.model_id} {mae*100:.4f}.png"
        )

        return
