import numpy as np
import torch
from calculateError import calculate_error

def evaluate(fcstnet, test_data, test_loader, pred_len, batch_size, return_lists=False):
    batch_mase_cost = []
    batch_smape_cost = []
    batch_mae_cost = []
    batch_mape_cost = []
    batch_mse_cost = []
    fcstnet.model.eval()

    # Load model parameters
    # checkpoint = torch.load(fcstnet.save_file, map_location=fcstnet.device)
    # fcstnet.model.load_state_dict(checkpoint['model_state_dict'])
    # fcstnet.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.double()
        batch_y = batch_y.double()
        batch_x_mark = batch_x_mark.double()
        batch_y_mark = batch_y_mark.double()
        batch_y = batch_y[:, -pred_len:, :]
        batch_y = batch_y.transpose(0, 1)
        with torch.no_grad():
            y_pred_list = []
            y_pred = fcstnet.model(input=batch_x, input_emb=batch_x_mark, target=batch_y, target_emb=batch_y_mark,
                                       is_training=False)
            mase_list = []
            smape_list = []
            nrmse_list = []
            mae_list = []
            mape_list = []
            mse_list = []
            for i in range(batch_size):
                mase, se, smape, nrmse, mae, mse, rmse, mape, mspe = calculate_error(
                    y_pred[:, i, :].detach().cpu().numpy(), batch_y[:, i, :].detach().numpy())
                mase_list.append(mase)
                smape_list.append(smape)
                nrmse_list.append(nrmse)
                mae_list.append(mae)
                mse_list.append(mse)
            mase = np.mean(mase_list)
            smape = np.mean(smape_list)
            mae = np.mean(mae_list)
            mse = np.mean(mse_list)
            batch_mase_cost.append(mase)
            batch_smape_cost.append(smape)
            batch_mae_cost.append(mae)
            batch_mse_cost.append(mse)

        if return_lists:
            return np.ndarray.flatten(np.array(mase_list)), np.ndarray.flatten(
                np.array(smape_list)), np.ndarray.flatten(
                np.array(nrmse_list))
        else:
            mase = np.mean(batch_mase_cost)
            smape = np.mean(batch_smape_cost)
            mae = np.mean(batch_mae_cost)
            mse = np.mean(batch_mse_cost)
            return mase, smape, mse, mae