from dataHelpers import _get_data
from evaluate import evaluate
from train import train
from MATVN import MATVN
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt

seq_len = in_seq_length = predict_start = 48
pred_len = out_seq_length =output_len =  24
input_dim = input_size = 7
output_dim = 7
batch_size= 16
n_epochs=1
learning_rate = 0.001
dropout = 0.1
embedding_size = 96
hidden_size = 128
learneddim = 8
n_encoder_layers = 3

enc_attn_type ='full'
encoder_attention ='full'
dec_attn_type = 'full'
decoder_attention = 'full'

window_size = [7,14,21]

train_data, train_loader = _get_data(data='ETTh1', flag='train',seq_len = in_seq_length, pred_len = pred_len)
valid_data, valid_loader = _get_data(data='ETTh1', flag='val', seq_len = in_seq_length, pred_len = pred_len)
test_data, test_loader = _get_data(data='ETTh1', flag='test', seq_len = in_seq_length, pred_len = pred_len)

fcstnet = MATVN(in_seq_length=in_seq_length, out_seq_length=out_seq_length, input_dim=input_dim,
                      output_dim=output_dim,  batch_size=batch_size,n_encoder_layers=n_encoder_layers,
                      window_size = window_size,learneddim = learneddim, hidden_size=hidden_size, embedding_size=embedding_size,
                      n_epochs=1, learning_rate=learning_rate, save_file='./forecastnet.pt')

training_costs, validation_costs = train(fcstnet, train_data, train_loader, valid_data, valid_loader, batch_size, pred_len,seq_len,
                                       restore_session=False)

mase, smape, mse, mae = evaluate(fcstnet, test_data, test_loader, pred_len = 24, batch_size = 16, return_lists=False)

print("")
print('MSE:', mse)
print('MAE:', mae)














