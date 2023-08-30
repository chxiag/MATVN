import time
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np

def train(fcstnet, train_data, train_loader, vali_data, vali_loader, batch_size, pred_len,seq_len,restore_session=False):
    # Initialise model with predefined parameters
    if restore_session:
        # Load model parameters
        checkpoint = torch.load(fcstnet.save_file)
        fcstnet.model.load_state_dict(checkpoint['model_state_dict'])
        fcstnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # List to hold the training costs over each epoch    training_costs = []
    validation_costs = []

    # Set in training mode
    fcstnet.model.train()
    train_window = seq_len + pred_len
    predict_start = seq_len

    for epoch in range(fcstnet.n_epochs):
        t_start = time.time()
        print('Epoch: %i of %i' % (epoch + 1, fcstnet.n_epochs))
        batch_cost = []
        training_costs = []
        count = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.double()
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double()
            batch_y_mark = batch_y_mark.double()
            valid = torch.autograd.Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = torch.autograd.Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            batch_y = batch_y.transpose(0, 1)
            fcstnet.optimizer.zero_grad()
            outputs = fcstnet.model(input=batch_x, input_emb=batch_x_mark, target=batch_y, target_emb=batch_y_mark,
                                    is_training=True)
            outputs = outputs.double()
            batch_y = batch_y.double()
            outputs = outputs#.cuda()
            batch_x = batch_x#.cuda()
            batch_y = batch_y#.cuda()


            loss = F.mse_loss(input=outputs, target=batch_y)
            batch_cost.append(loss.item())
            loss = loss.to(torch.float32)

            loss.backward()
            fcstnet.optimizer.step()

        epoch_cost = np.mean(batch_cost)
        training_costs.append(epoch_cost)

        if plot_train_progress:
            plt.cla()
            plt.plot(np.arange(input.shape[0], input.shape[0] + batch_y.shape[0]), batch_y[:, 0, 0])
            temp = outputs.detach()
            plt.plot(np.arange(input.shape[0], input.shape[0] + batch_y.shape[0]), temp[:, 0, 0])
            plt.pause(0.1)

        if vali_data is not None:
            fcstnet.model.eval()
            with torch.no_grad():
                batch_valid_cost = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.double()
                    batch_y = batch_y.double()
                    batch_x_mark = batch_x_mark.double()
                    batch_y_mark = batch_y_mark.double()
                    batch_y = batch_y.transpose(0, 1)
                    batch_y = batch_y#.cuda()
                    y_valid_prediction = fcstnet.model(input=batch_x, input_emb=batch_x_mark, target=batch_y,
                                                       target_emb=batch_y_mark,
                                                       is_training=False)
                    # Calculate the loss
                    # y_valid_prediction=y_valid_prediction.float()
                    y_valid_prediction = y_valid_prediction#.cuda()
                    y_valid_prediction = y_valid_prediction.float()
                    batch_y = batch_y.float()
                    loss = F.mse_loss(input=y_valid_prediction, target=batch_y)  #
                    batch_valid_cost.append(loss.item())
                epochvalid_cost = np.mean(batch_valid_cost)
                validation_costs.append(epochvalid_cost)


        print("Average epoch training cost: ", epoch_cost)
        if vali_data is not None:
            print('Average validation cost:     ', validation_costs[-1])
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((fcstnet.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (fcstnet.n_epochs - epoch - 1) * (time.time() - t_start)))
        best_result = False
        if vali_data is None:
            if training_costs[-1] == min(training_costs):
                best_result = True
        else:
            if validation_costs[-1] == min(validation_costs):
                best_result = True
        if best_result:
            torch.save({
                'model_state_dict': fcstnet.model.state_dict(),
                'optimizer_state_dict': fcstnet.optimizer.state_dict(),
            }, fcstnet.save_file)
            print("Model saved in path: %s" % fcstnet.save_file)
    return training_costs, validation_costs
