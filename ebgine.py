import torch
import math
import os
import time
import copy
import numpy as np
from tqdm import tqdm
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tensorboardX import SummaryWriter
import torch.nn.functional as F

class Engine(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        self.current_epoch = 0 
        super(Engine, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, '{}_{}_best_model.pth'.format(args.dataset, args.model))
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)  # run.log
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
         # Create a SummaryWriter for TensorBoard
        self.writer = SummaryWriter(log_dir="/content/AFDGCN_BerNet/logs_appnp_pems")
        

    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_mape = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :1]
            label = target[..., :1].to(data.device)    # (..., 1)
            # data and target shape: B, T, N, F; output shape: B, T, N, F
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            output = output.to(label.device)
            loss = self.loss(output, label)
            mae = torch.abs(output - label).mean()
            rmse = torch.sqrt(F.mse_loss(output, label))
            mape = (torch.abs(output - label) / (label + 1e-7)).mean() * 100
            total_mae += mae.item()
            total_rmse += rmse.item()
            total_mape += mape.item()
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        
        train_epoch_loss = total_loss / self.train_per_epoch
        train_epoch_mae = total_mae / self.train_per_epoch
        train_epoch_rmse = total_rmse / self.train_per_epoch
        train_epoch_mape = total_mape / self.train_per_epoch

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
       
        return train_epoch_loss,train_epoch_mae, train_epoch_rmse, train_epoch_mape


    def val_epoch(self, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        total_val_mae = 0
        total_val_rmse = 0
        total_val_mape = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :1]
                label = target[..., :1].to(data.device) 
                output = self.model(data)
                output = output.to(label.device)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                mae = torch.abs(output - label).mean()
                rmse = torch.sqrt(F.mse_loss(output, label))
                mape = (torch.abs(output - label) / (label + 1e-7)).mean() * 100

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                total_val_mae += mae.item()
                total_val_rmse += rmse.item()
                total_val_mape += mape.item()
        num_batches = len(val_dataloader)
        val_loss = total_val_loss / num_batches
        val_mae = total_val_mae / num_batches
        val_rmse = total_val_rmse / num_batches
        val_mape = total_val_mape / num_batches

        return val_loss, val_mae, val_rmse, val_mape


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            self.current_epoch = epoch
            t1 = time.time()
            train_epoch_loss, train_epoch_mae, train_epoch_rmse, train_epoch_mape = self.train_epoch()
            self.writer.add_scalar('Loss/Train', train_epoch_loss, epoch)
            self.writer.add_scalar('Metrics/MAE_Train', train_epoch_mae, epoch)
            self.writer.add_scalar('Metrics/RMSE_Train', train_epoch_rmse, epoch)
            self.writer.add_scalar('Metrics/MAPE_Train', train_epoch_mape, epoch)
            t2 = time.time()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            t3 = time.time()
            val_epoch_loss, val_epoch_mae, val_epoch_rmse, val_epoch_mape= self.val_epoch(val_dataloader)
            self.writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
            self.writer.add_scalar('Metrics/MAE_Val', val_epoch_mae, epoch)
            self.writer.add_scalar('Metrics/RMSE_Val', val_epoch_rmse, epoch)
            self.writer.add_scalar('Metrics/MAPE_Val', val_epoch_mape, epoch)
            t4 = time.time()
            self.logger.info('Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f} secs.'.format(epoch, train_epoch_loss, val_epoch_loss, (t2 - t1)))
            print("Inference Time: {:.4f} secs.", (t4 - t3))
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)


    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :1].to(args.device)
                label = target[..., :1].to(args.device)
                output = model(data)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).to(args.device)
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0).to(args.device).to(args.device)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0)).to(args.device)
        print(y_true.cpu().numpy().shape)
        # np.save('./NewNet_{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        # np.save('./NewNet_{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())

        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.rmse_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
            # Store metric values in dictionaries
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.rmse_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))
