import sys
import pickle
import torch
import numpy as np
from model import CNNModel


class DANN:
    def __init__(self, lr=0.001):
        with open('./data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.my_net = CNNModel()
        self.optimizer = torch.optim.SGD(self.my_net.parameters(), lr=lr)
        self.loss_class = torch.nn.MSELoss()
        self.loss_domain = torch.nn.MSELoss()

    def acc_rate(self, predict, labels):
        counter = 0
        zeross = torch.zeros(1, predict.shape[1])
        for i, j in zip(predict, labels):
            temp = torch.zeros(1, labels.shape[1])
            temp[:, int(i.argmax(dim=0))] = 1
            # print(temp*j)
            if zeross.equal(temp * j):
                pass
            else:
                counter += 1
        return counter / predict.shape[0]

    def fetch_data(self, fold):
        train_key, train_index = list(self.data.keys()), list(range(domain_num))
        valid_key, _ = train_key.pop(fold), train_index.pop(fold)
        valid_x, valid_y = self.data[valid_key]['data'], self.data[valid_key]['label']
        train_x, train_y = np.vstack([self.data[k]['data'] for k in train_key]), np.hstack(
            [self.data[k]['label'] for k in train_key])
        valid_d, train_d = np.ones(valid_y.size) * fold, np.repeat(train_index, valid_y.size)
        valid_y = torch.nn.functional.one_hot(torch.tensor(valid_y + 1).to(torch.int64))
        valid_d = torch.nn.functional.one_hot(torch.tensor(valid_d).to(torch.int64), num_classes=5)
        # to_categorical(train_y + 1).astype(np.float32)
        train_y = torch.nn.functional.one_hot(
            torch.tensor(train_y + 1).to(torch.int64))
        # to_categorical(train_d, num_classes=domain_num).astype(np.float32)
        train_d = torch.nn.functional.one_hot(torch.tensor(train_d).to(torch.int64), num_classes=5)
        train_x, valid_x = train_x.astype(np.float32), valid_x.astype(np.float32)
        return (torch.tensor(train_x), torch.tensor(train_y), torch.tensor(train_d)), (
            torch.tensor(valid_x), torch.tensor(valid_y), torch.tensor(valid_d))

    def test(self, test):
        # to do
        t_img, t_label, valid_d = test
        t_img, t_label, valid_d = t_img, t_label, valid_d
        p = float(len(t_label) * len(t_label))
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        class_output, valid_output = self.my_net(t_img.to(torch.float32), alpha)
        acc_test = self.acc_rate(class_output, t_label)
        acc_domain = self.acc_rate(valid_output, valid_d)
        print('\ntest acc is {} the acc_domain is {}\n'.format(acc_test, acc_domain))

    def train(self, n_epoch, batch_size, train, test):
        s_img, s_label, domain_label = train
        # print(domain_label.shape,'\n',s_label.shape),s_img.shape,s_label.shape)
        # sys.exit()
        t_img, t_label, domain_t = test
        for epoch in range(n_epoch):
            index_batch = 0
            while index_batch * batch_size < len(s_label):
                p = float(index_batch + epoch * len(s_label)) / n_epoch / len(s_label)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # training model using source data
                self.my_net.zero_grad()

                if (index_batch + 1) * batch_size < len(s_label):
                    s_img_mini = s_img[index_batch * batch_size:(index_batch + 1) * batch_size, :]
                    s_label_mini = s_label[index_batch * batch_size:(index_batch + 1) * batch_size, :]
                    domain_label_mini = domain_label[index_batch * batch_size:(index_batch + 1) * batch_size, :]
                    # print('\n\nid_num+index_batch*batch_size: \n\n',id_num+index_batch*batch_size,'\n',id_num+(index_batch+1)*batch_size)
                elif index_batch * batch_size <= len(s_label):
                    s_img_mini = s_img[index_batch * batch_size:, :]
                    s_label_mini = s_label[index_batch * batch_size:, :]
                    domain_label_mini = domain_label[index_batch * batch_size:, :]
                    # print('\n\nid_num+index_batch*batch_size: \n\n',id_num+index_batch*batch_size,'\n',id_num+index_batch*batch_size+len(s_label_mini))
                # print(s_img_mini.shape)
                class_output, domain_output = self.my_net(s_img_mini.to(torch.float32), alpha)
                # print('classes\t',class_output.shape)
                # print('domain_output\t',class_output.shape,s_label_mini.shape)
                # sys.exit()
                # print(domain_label_mini.shape,s_label_mini.shape,s_img_mini.shape)
                err_s_label = self.loss_class(class_output.to(torch.float32), s_label_mini.to(torch.float32))
                err_s_domain = self.loss_domain(domain_output.to(torch.float32), domain_label_mini.to(torch.float32))

                if (index_batch + 1) * batch_size < len(s_label):
                    t_img_mini = t_img[int(index_batch * batch_size / 4):int((index_batch + 1) * batch_size / 4),
                                       :]
                    domain_t_label_mini = domain_t[
                                          int(index_batch * batch_size / 4):int((index_batch + 1) * batch_size / 4),
                                          :]

                elif index_batch * batch_size <= len(s_label):
                    t_img_mini = t_img[int(index_batch * batch_size / 4):, :]
                    domain_t_label_mini = domain_t[int(index_batch * batch_size / 4):, :]

                # print('t_img_mini\t',t_img_mini.shape)

                _, domain_output = self.my_net(t_img_mini.to(torch.float32), alpha)
                # print('domain_out\t',domain_output.shape,domain_t_label_mini.shape)
                # sys.exit()
                err_t_domain = self.loss_domain(domain_output.to(torch.float32), domain_t_label_mini.to(torch.float32))
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                self.optimizer.step()

                sys.stdout.write(
                    '\r epoch: %d, error_s_label: %f, error_s_domain: %f, error_t_domain: %f'
                    % (epoch+1, err_s_label.data.numpy(),
                       err_s_domain.data.numpy(), err_t_domain.data.item()))
                sys.stdout.flush()
                index_batch += 1
            print("\n accurate rate is %.4f" % (self.acc_rate(class_output, s_label_mini)))

    def run(self):
        for folder in range(domain_num):
            train, test = self.fetch_data(folder)
            self.train(50, 400, train, test)
            self.test(test)
            # self.test_init(test)


inputs_dim, feature_dim = 310, 128
labels_dim, domain_dim = 64, 64
labels_num, domain_num = 3, 5

if __name__ == '__main__':
    model = DANN()
    model.run()
