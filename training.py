import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from IQADataset import IQADataset
from Network import CNNIQAnet
from sklearn.linear_model import LinearRegression
from torch.optim import Adam
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from scipy import stats


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def loss_fn(y_pred, y):
    return F.l1_loss(y_pred, y[0])

class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._y_pred = []
        self._y = []
        self._y_std = []

    def update(self, output):
        y_pred, y = output
        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        self._y_pred.append(torch.mean(y_pred).item())

    def compute(self):
        sq = np.asarray(self._y)
        q = np.asarray(self._y_pred)

        # Calculate SROCC and KROCC
        SROCC = stats.spearmanr(sq, q)[0]
        KROCC = stats.kendalltau(sq, q)[0]

        # Perform non-linear fitting
        reg = LinearRegression()
        reg.fit(q.reshape(-1, 1), sq.reshape(-1, 1))
        q_pred = reg.predict(q.reshape(-1, 1))

        # Calculate PLCC, RMSE, MAE, OR after non-linear fitting
        PLCC = stats.pearsonr(sq, q_pred.flatten())[0]
        RMSE = np.sqrt(((sq - q_pred.flatten()) ** 2).mean())
        MAE = np.abs(sq - q_pred.flatten()).mean()
        OR = (np.abs(sq - q_pred.flatten()) > 2 * np.std(sq - q_pred.flatten())).mean()

        return SROCC, KROCC, PLCC, RMSE, MAE, OR

def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=4)
    print("Train loader created")

    val_dataset = IQADataset(config, 'val')
    val_loader = DataLoader(val_dataset)
    print("Val loader created")

    if config['test_ratio']:
        test_dataset = IQADataset(config, 'test')
        test_loader = DataLoader(test_dataset)
        print("Test loader created")

        return train_loader, val_loader, test_loader

    return train_loader, val_loader

from torch.utils.tensorboard import SummaryWriter

def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir, trained_model_file, save_result_file, disable_gpu=False):
    if config['test_ratio']:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    device = torch.device("cpu")
    model = CNNIQAnet(patch_size=32).to(device)
    writer = SummaryWriter(log_dir=log_dir)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)
    best_epoch = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        writer.add_scalar("validation/OR", OR, engine.state.epoch)
        global best_epoch
        if SROCC > best_criterion:
            best_criterion = SROCC
            best_epoch = engine.state.epoch  # Update the best epoch variable
            torch.save(model.state_dict(), trained_model_file)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            print("Testing Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)
            writer.add_scalar("testing/OR", OR, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"]:
            model.load_state_dict(torch.load(trained_model_file))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            if save_result_file:
                np.save(str(save_result_file), (SROCC, KROCC, PLCC, RMSE, MAE, OR))

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_testing_results)

    trainer.add_event_handler(Events.COMPLETED, final_testing_results)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()
