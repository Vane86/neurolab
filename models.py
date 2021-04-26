import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import accuracy_score


class FCLayer(torch.nn.Module):

    def __init__(self, inputs, outputs, dropout_p):
        super().__init__()

        self.batch_norm = torch.nn.BatchNorm1d(inputs)
        self.fc = torch.nn.Linear(inputs, outputs)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class FCNN(torch.nn.Module):

    def __init__(self, inputs, outputs, hidden_size, dropout_p):
        super().__init__()
        self.l1 = FCLayer(inputs, hidden_size, dropout_p=dropout_p)
        self.l2 = FCLayer(hidden_size, hidden_size, dropout_p=dropout_p)
        self.l3 = FCLayer(hidden_size, hidden_size, dropout_p=dropout_p)
        self.l4 = FCLayer(hidden_size, outputs, dropout_p=dropout_p)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1) + x1
        x3 = self.l3(x2) + x2
        x4 = self.l4(x3)
        return x4


class LSTMNN(torch.nn.Module):

    def __init__(self, inputs, outputs, hidden_size=32, num_layers=1):
        super().__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=inputs, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, outputs)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


class EEGDataset(Dataset):

    def __init__(self, features, labels):
        self._features = features
        self._labels = labels

    def __getitem__(self, item):
        return self._features[item], self._labels[item]

    def __len__(self):
        return len(self._features)


class Classifier:

    def __init__(self, model, epochs, learning_rate, lamb, batch_size):
        self._model = model
        self._epochs = epochs
        self._lr = learning_rate
        self._lamb = lamb
        self._batch_size = batch_size

    def fit(self, features, labels):

        self._model.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._lamb)
        loss_function = torch.nn.CrossEntropyLoss()

        data_loader = DataLoader(EEGDataset(features, labels),
                                 batch_size=self._batch_size,
                                 shuffle=True, drop_last=True)
        val_acc_history = list()
        for epoch in range(self._epochs):
            print(f'Starting epoch {epoch} / {self._epochs}')
            for i, batch in enumerate(data_loader):

                n_batches = len(data_loader)
                batch_features, batch_labels = batch

                optimizer.zero_grad()

                loss = loss_function(self._model(batch_features), batch_labels)
                loss.backward()

                optimizer.step()

                if i % (n_batches // 5) == 1:
                    self._model.eval()
                    val_acc = torch.tensor(self.predict(batch_features) == batch_labels, dtype=torch.float32).mean().item()
                    print(f'Batch {i} / {n_batches} processed.')
                    print(f'Train acc: {val_acc}')
                    print(f'Train loss: {loss.item()}')
                    self._model.train()
        self._model.eval()
        return self

    def predict(self, features):
        return self._model(features).argmax(axis=-1).data


if __name__ == '__main__':
    clf = Classifier(LSTMNN(1, 3), 100, 0.0001, 0)
    data = torch.arange(0, 100, 0.5).resize(2, 100, 1)
    print(data)
    print(clf.predict(data))
