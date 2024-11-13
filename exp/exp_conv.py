import datetime

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


class Exp_Conv(Exp_Basic):
    def __init__(self, args):
        super(Exp_Conv, self).__init__(args)

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

    def vali(self, vali_data, vali_loader, criterion):
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
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.detach().cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
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
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:]

                # loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time/1000))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("计算vali_loss，test_loss...")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
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
        folder_path = './test_results/' + setting + '/'
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
                outputs = outputs.detach().cpu().numpy()
                if len(batch_y.shape)<len(outputs.shape):
                    # 补充一层，使模型输出shape = y shape
                    batch_y = batch_y.unsqueeze(1)
                pred = outputs
                true = batch_y.cpu().numpy()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    visual(true[0, :, -1], pred[0, :, -1], os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        print('test shape:', preds.shape, trues.shape)
        preds = np.reshape(preds, (preds.shape[0], -1))
        trues = np.reshape(trues, (trues.shape[0], -1))
        drawUtil.completeMSE(
            predicted=preds,
            real=trues,
        )

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        drawUtil.drawResultCompare(
            result=preds,
            real=trues,
            tag="tcn test",
            savePath=folder_path+'test.png',
        )
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.savetxt(folder_path + 'pred.csv', preds, delimiter=',')
        np.savetxt(folder_path + 'trues.csv', trues, delimiter=',')
        return
