import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import RandomOverSampler
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# random_seed = 16
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

train_df = pd.read_csv('done_train_df.csv')
train_df.drop('Customer_ID', axis=1, inplace=True)

obj_to_num_dict = {}
num_to_obj_dict = {}


def convert_str_to_num(df=train_df):
    # df=pd.get_dummies(df,columns=['Occupation'])
    for column in (df.select_dtypes(include='object').columns):
        obj_to_num_dict[column] = {}
        num_to_obj_dict[column] = {}
        for n, i in enumerate(df[column].unique()):
            df[column] = df[column].replace(i, n + 1)
            obj_to_num_dict[column][i] = n + 1
            num_to_obj_dict[column][n + 1] = i
        df[column] = df[column].astype('uint8')
    return df


train_df = convert_str_to_num(df=train_df)


# train_df.replace({'Credit_Score':{1:0,2:1,3:2}})

def split_and_balance(df, target='Credit_Score', test_size=0.2, random_state=0,
                      augmentation=RandomOverSampler(random_state=16)):
    test = pd.DataFrame()
    nuniq_labels = df[target].nunique()

    for i, l in enumerate(df[target].unique()):
        if i == 0:
            test = df.query(f'Credit_Score=={l}').sample(n=int((df.shape[0] * test_size) // nuniq_labels),
                                                         random_state=random_state)
        else:
            test = pd.concat([test,
                              df.query(f'Credit_Score=={l}').sample(n=int((df.shape[0] * test_size) // nuniq_labels),
                                                                    random_state=random_state)])
    train = df.drop(test.index)

    if augmentation:
        xtrain, ytrain = augmentation.fit_resample(train.drop(target, axis=1), train[target])

    else:
        xtrain, ytrain = train.drop(target, axis=1), train[target]

    xtest, ytest = test.drop(target, axis=1), test[target]

    return xtrain, xtest, ytrain, ytest


x_train, x_test, y_train, y_test = split_and_balance(train_df, test_size=0.2,
                                                     augmentation=RandomOverSampler(random_state=17))

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


class Fc(nn.Module):
    def __init__(self, input_size, output_size, n_layers, hidden_sizes, l_activations, l_dropouts):
        super().__init__()
        layers = []
        for _, size, activ, drop in zip(range(n_layers), hidden_sizes, l_activations, l_dropouts):
            if _ == 0:
                layers.append(nn.Linear(input_size, size))
                layers.append(nn.Dropout(drop))
                layers.append(getattr(nn, activ)())
            else:
                layers.append(nn.Linear(hidden_sizes[_ - 1], size))
                layers.append(nn.Dropout(drop))
                layers.append(getattr(nn, activ)())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.fcnn_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcnn_model(x)


class Data(Dataset):
    def __init__(self):
        self.x = torch.tensor(x_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.float32).view(-1)
        self.instances = self.x.shape[0]

    def __len__(self):
        return self.instances

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def objective(trial):
    n_layers = trial.suggest_int('nlayers', 2, 3, 1)
    # Set for 2 layers
    # n_layers=2
    hidden_sizes = []
    # Set activations
    l_activations = []
    l_dropouts = [0.3, 0.2]
    if n_layers != 2:
        l_dropouts.append(0.2)

    for l in range(n_layers):
        hidden_sizes.append(trial.suggest_int(f'size_l{l}', 32 * 6, 128 * 10, 32))
        l_activations.append(trial.suggest_categorical(f'activation_l{l}', ['LeakyReLU', 'ReLU']))
    # set learning rate as 1e-4
    # learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1,step=1e-7)
    model = Fc(input_size=x_train.shape[1], output_size=1, n_layers=n_layers, hidden_sizes=hidden_sizes,
               l_activations=l_activations, l_dropouts=l_dropouts).to(device=device)
    criterion = nn.MSELoss()
    # Set weight_decay (L2 penalty) to avoid over fitting.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    df = Data()
    epochs = 30
    train_loader = DataLoader(dataset=df, batch_size=x_train.shape[0] // 6, shuffle=True)
    loss = None
    for epoch in range(epochs):
        for batch, (features, labels) in enumerate(train_loader):
            # Adding data to cuda
            features = features.to(device=device)
            labels = labels.to(device=device)
            predictions = model.forward(features).view(-1).to(device=device)
            loss = torch.sqrt(criterion(predictions, labels).to(device=device))
            print(f"batch:{batch}\nepoch:{epoch} loss: {loss.to(device='cpu')}")
            # Backprop
            loss.backward()
            # Optimizer step
            optimizer.step()
            # Reset gradient
            optimizer.zero_grad(set_to_none=True)
    if loss_check[-1] > loss:
        for param, param_val in zip(['n_layers', 'hidden_sizes', 'l_activations', 'l_dropouts'],
                                    [n_layers, hidden_sizes, l_activations, l_dropouts]):
            best_trial_params[param] = param_val
        torch.save(model.state_dict(), 'best_trial_st_dict.pth')
        print(f'old loss:{loss_check[-1]}\nnew loss: {loss}')
        loss_check.append(loss)
    return loss


def detailed_objective(trial):
    n_layers = trial.suggest_int('nlayers', 2, 3, 1)
    # n_layers = 2
    hidden_sizes = []
    # Set activations
    l_activations = ['LeakyReLU', 'LeakyReLU']
    # To avoid over fitting.
    l_dropouts = [0.3, 0.2]
    if n_layers != 2:
        l_dropouts.append(0.2)

    # l_dropouts = [0.3, 0.25]

    for l in range(n_layers):
        hidden_sizes.append(trial.suggest_int(f'size_l{l}', 32 * 5, 128 * 10, 32))
        l_activations.append(trial.suggest_categorical(f'activation_l{l}', ['LeakyReLU', 'ReLU']))

    # learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1,step=1e-7)
    model = Fc(input_size=x_train.shape[1], output_size=1, n_layers=n_layers, hidden_sizes=hidden_sizes,
               l_activations=l_activations, l_dropouts=l_dropouts).to(device=device)
    model.load_state_dict(torch.load('best_trial_st_dict.pth'))
    criterion = nn.MSELoss()
    # Set weight_decay (L2 penalty) to avoid over fitting.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    df = Data()
    epochs = 5000
    train_loader = DataLoader(dataset=df, batch_size=x_train.shape[0] // 6, shuffle=True)
    loss = None
    for epoch in range(epochs):
        for batch, (features, labels) in enumerate(train_loader):
            # Adding data to cuda
            features = features.to(device=device)
            labels = labels.to(device=device)
            predictions = model.forward(features).view(-1).to(device=device)
            loss = torch.sqrt(criterion(predictions, labels).to(device=device))
            if loss_check[-1] > loss:
                for param, param_val in zip(['n_layers', 'hidden_sizes', 'l_activations', 'l_dropouts'],
                                            [n_layers, hidden_sizes, l_activations, l_dropouts]):
                    best_trial_params[param] = param_val
                torch.save(model.state_dict(), 'best_trial_st_dict.pth')

                print(f'old loss:{loss_check[-1]}\nnew loss: {loss}')
                loss_check.append(loss)
            acc_train = (model(torch.tensor(x_train, dtype=torch.float32).to(device=device)).round().to(
                device='cpu').view(-1) == torch.tensor(y_train.to_numpy())).sum() / y_train.shape[0]
            acc_test = (model(torch.tensor(x_test, dtype=torch.float32).to(device=device)).round().to(
                device='cpu').view(-1) == torch.tensor(y_test.to_numpy())).sum() / y_test.shape[0]
            print(
                f"batch:{batch}\nepoch:{epoch}\nloss: {loss.to(device='cpu')}\naccuracy: train: {acc_train}, test:{acc_test}")
            # Backprop
            loss.backward()
            # Optimizer step
            optimizer.step()
            # Reset gradient
            optimizer.zero_grad(set_to_none=True)

    return model


if __name__ == '__main__':
    loss_check = [1]
    best_trial_params = {}
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    print(f'best value:{study.best_value} (params: {study.best_params})')
    with open('study_best_trail.txt', 'w') as f:
        f.write('dict = ' + repr(study.best_params) + '\n')
    best_params = study.best_params
    best_trial_model = detailed_objective(study.best_trial)
