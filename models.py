import torch


class FCNN(torch.nn.Module):

    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = torch.nn.Linear(inputs, 32)
        self.act1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.85)
        self.fc2 = torch.nn.Linear(32, 16)
        self.act2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.85)
        self.fc3 = torch.nn.Linear(16, outputs)
        self.act3 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout1(x)
        x = self.act2(self.fc2(x))
        x = self.dropout2(x)
        x = self.act3(self.fc3(x))
        return x


class Classifier:

    def __init__(self, model, epochs, learning_rate, lamb):
        self._model = model
        self._epochs = epochs
        self._lr = learning_rate
        self._lamb = lamb

    def fit(self, features, labels):

        self._model.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._lamb)
        loss_function = torch.nn.CrossEntropyLoss()

        # features, labels = torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32)
        val_acc_history = list()
        for epoch in range(self._epochs):
            optimizer.zero_grad()
            loss = loss_function(self._model(features), labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}/{self._epochs}')

            if epoch % 100 == 99:
                self._model.eval()
                val_acc = torch.tensor(self.predict(features) == labels, dtype=torch.float32).mean().item()
                print(f'Train acc: {val_acc}')
                self._model.train()
        self._model.eval()
        return self

    def predict(self, features):
        return self._model(features).argmax(axis=1).data

