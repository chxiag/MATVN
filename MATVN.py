import torch
from TVarchitecture import TVarchitecture
from dataHelpers import format_input
from optimizer import OpenAIAdam
import numpy as np
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt
class MATVN:
    def __init__(self, in_seq_length, out_seq_length, input_dim, output_dim, batch_size, n_encoder_layers, window_size,learneddim, hidden_size, embedding_size,
                 n_epochs=1, learning_rate=0.0001, save_file='./forecastnet.pt'):
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.n_encoder_layers = n_encoder_layers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.save_file = save_file
        self.window_size = window_size
        self.learneddim = learneddim
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TVarchitecture(self.input_dim, self.output_dim, self.in_seq_length, self.out_seq_length, self.n_encoder_layers,
                                    self.window_size, self.learneddim, self.hidden_size, self.embedding_size,self.device)

        self.model.to(self.device)
        lr_warmup = True
        n_updates_total = 100  # (batch_x.__len__() // batch_size) * n_epochs
        lr_schedule = 'warmup_constant'

        self.optimizer = OpenAIAdam(self.model.parameters(),
                                      lr=self.learning_rate,
                                      schedule=lr_schedule,
                                      warmup=lr_warmup,
                                      t_total=n_updates_total,
                                      b1=0.9,
                                      b2=0.999,
                                      e=1e-8,
                                      l2=0.01,
                                      vector_l2='store_true',
                                      max_grad_norm=1)


    def forecast(self, test_x, test_x_emb):
        self.model.eval()

        # Load model parameters
        checkpoint = torch.load(self.save_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with torch.no_grad():
            if type(test_x) is np.ndarray:
                test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
            # Format the inputs
            test_x = format_input(test_x)
            test_x_emb = format_input(test_x_emb)
            # Dummy output
            empty_y = torch.empty(
                (self.out_seq_length, test_x[:, :self.in_seq_length].shape[1], self.output_dim))
            test_x = test_x.to(self.device)
            empty_y = empty_y.to(self.device)

            # Compute the forecast
            y_hat = self.model(test_x[:, :self.in_seq_length], empty_y, is_training=False)
        return y_hat.cpu().numpy()