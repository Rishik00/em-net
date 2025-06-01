import os
from torch.utils.data import DataLoader
import mlflow
import torch
import typer
from dotenv import load_dotenv
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

## Local imports
from data import SyntheticTimeSeriesDataset, SyntheticTimeSeriesTensorDataset, DatasetConfig
from model import DenseNetwork, ModelConfig

load_dotenv()

def train_model(train_dataloader, model, optim, loss_fn, device):
    for batch_idx, (inp, target) in enumerate(train_dataloader):
        inp, target = inp.to(device), target.to(device)

        pred = model(inp)
        loss = loss_fn(pred, target)

        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx
            mlflow.log_metric("loss", loss, step=(batch_idx // 100))

            print(
                f"loss: {loss:3f} [{current} / {len(train_dataloader)}]"
            )

def test_model():
    pass

def main(num_epochs, batch_size, lr):
    mlflow.set_tracking_uri(f"{os.environ("MLFLOW_HOST"):{os.environ("MLFLOW_PORT")}}")
    mlflow.set_experiment("Torch basic time series")


    dataset_config = DatasetConfig(
        date_start='2021/10/11 00:00:00', 
        date_end='2021/11/11 00:00:00',
        sampling='H',
        setting='simple',
        seq_length=24,
        horizon=1,
        split_ratio=0.2
    )
    
    gen = SyntheticTimeSeriesDataset(
        config = dataset_config
    )

    X, y = gen.create_univariate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    train_ds, test_ds = SyntheticTimeSeriesTensorDataset(dataset_config, X_train, y_train), SyntheticTimeSeriesTensorDataset(dataset_config, X_test, y_test)
 
    model_config = ModelConfig(
        input_dim= X.shape[1],
        hidden_dim= 256,
        output_dim= 1,
        lr = lr,
        num_epochs= num_epochs,
        batch_size = batch_size,
    )

    loss_fn = torch.nn.L1Loss()
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=1)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseNetwork(config=model_config).to(device)
    
    print(f"[INFO] device set to: {device}")

    with mlflow.start_run():
        mlflow.log_params(model_config.__dict__)

        for t in range(num_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_model(train_dl, model, loss_fn, optimizer, device)

        test_model()


if __name__ == "__main__":

    typer.run(main)