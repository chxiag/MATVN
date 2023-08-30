import numpy as np
import warnings
def calculate_error(Yhat, Y, print_errors=False):
    # Ensure arrays are 2D
    assert np.ndim(Y) <= 3, 'Y must be one, two, or three dimensional, with the sequence on the first dimension'
    assert np.ndim(Yhat) <= 3, 'Yhat must be one, two, or three dimensional, with the sequence on the first dimension'
    assert np.ndim(Y) <= np.ndim(Yhat), 'Y has a different shape to Yhat'

    # Prepare Y and Yhat based on their number of dimensions
    if np.ndim(Y) == 1:
        n_sequences = 1
        Y = np.expand_dims(Y, axis=1)
        Yhat = np.expand_dims(Yhat, axis=1)
    elif np.ndim(Y) == 2:
        n_sequences = Y.shape[1]
    elif np.ndim(Y) == 3 and Y.shape[2] == 1:
        assert Y.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Y = np.squeeze(Y, axis=2)
        assert Yhat.shape[2] == 1, 'For a three dimensional array, Y.shape[2] == 1'
        Yhat = np.squeeze(Yhat, axis=2)
        n_sequences = Y.shape[1]
    elif np.ndim(Y) == 3 and Y.shape[2] > 1:
        return calculate_miltidim_error(Yhat=Yhat, Y=Y, print_errors=print_errors)
    else:
        raise Warning('Error in dimensions')

    # Symmetric Mean Absolute Percentage Error (M4 comp)
    smape = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = np.absolute(Y[:, i] - Yhat[:, i])
        denominator = (np.absolute(Y[:, i]) + np.absolute(Yhat[:, i]))
        # Remove any elements with zeros in the denominator
        non_zeros = denominator != 0
        numerator = numerator[non_zeros]
        denominator = denominator[non_zeros]
        # Sequence length
        length = numerator.shape[0]
        # Calculate error
        smape.append(200.0 / length * np.sum(numerator / denominator))
    smape = np.array(smape)
    if print_errors:
        print('Symmetric mean absolute percentage error (sMAPE) = ', smape)

    # Mean absolute scaled error
    se = []
    mase = []
    for i in range(n_sequences):
        numerator = (Y[:, i] - Yhat[:, i])
        denominator = np.sum(np.absolute(Y[1:, i] - Y[0:-1, i]), axis=0)
        # Check if denominator is zero
        if denominator == 0:
            warnings.warn("The denominator for the MASE is zero")
            se.append(np.NaN * np.ones(length))
            mase.append(np.NaN)
            continue
        # Sequence length
        length = numerator.shape[0]
        # Scaled Error
        scaled_error = (length - 1) * numerator / denominator
        se.append(scaled_error)
        mase.append(np.mean(np.absolute(scaled_error)))
    mase = np.array(mase)
    if print_errors:
        print('Scaled error (SE) = ', se)
        print('Mean absolute scaled error (MASE) = ', mase)

    # Normalised Root Mean Squared Error
    nrmse = []
    for i in range(n_sequences):
        # Compute numerator and denominator
        numerator = 100 * np.sqrt(np.mean(np.square(Y[:, i] - Yhat[:, i])))
        denominator = np.max(Y[:, i]) - np.min(Y[:, i])
        # Remove any elements with zeros in the denominator
        non_zeros = denominator != 0
        numerator = numerator[non_zeros]
        denominator = denominator[non_zeros]
        # Calculate error
        nrmse.append(numerator / denominator)
    nrmse = np.array(nrmse)
    if print_errors:
        print('Normalised root mean squared error (NRMSE) = ', nrmse)

    def RSE(pred, true):
        return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

    def CORR(pred, true):
        u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
        return (u / d).mean(-1)

    def MAE(pred, true):
        return np.mean(np.abs(pred - true))

    def MSE(pred, true):
        return np.mean((pred - true) ** 2)

    def RMSE(pred, true):
        return np.sqrt(MSE(pred, true))

    def MAPE(pred, true):
        return np.mean(np.abs((pred - true) / true))

    def MSPE(pred, true):
        return np.mean(np.square((pred - true) / true))

    mae = MAE(Yhat, Y)
    mse = MSE(Yhat, Y)
    rmse = RMSE(Yhat, Y)
    mape = MAPE(Yhat, Y)
    mspe = MSPE(Yhat, Y)
    # return mase, se, smape, nrmse
    return mase, se, smape, nrmse, mae, mse, rmse, mape, mspe
    # return se, smape, nrmse