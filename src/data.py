from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass

#TODO: integrate this for a data script
import typer


@dataclass
class DatasetConfig:
    date_start: str='2021/10/11 00:00:00' 
    date_end: str='2021/11/11 00:00:00'
    sampling: str='H'
    setting: str='simple'
    seq_length: int=24
    horizon: int=1
    split_ratio: float=0.2

class SyntheticTimeSeriesDataset:
    def __init__(
        self, config: DatasetConfig
    ):
        
        self.config = config
        torch.manual_seed(0)

    def create_univariate_dataset(self):

        if isinstance(self.config.date_start, str) and isinstance(self.config.date_end, str):
            self.config.date_start = pd.to_datetime(self.config.date_start)
            self.config.date_end = pd.to_datetime(self.config.date_end)

        num_days = self.config.date_end - self.config.date_start
        print('Number of days in the time delta: ', num_days)

        if self.config.sampling == 'H':
            self.config.num_samples = num_days.days * 24
            print("Sampling for number of samples: ", self.config.num_samples)

        if self.config.sampling == 'D':
            self.config.num_samples = num_days

        index = pd.date_range(start=self.config.date_start, end=self.config.date_end, freq=self.config.sampling)
        series = torch.rand(len(index))  
        print('Index Length: ',len(index), index[0], index[1])

        df = pd.DataFrame({
            'time_series': series
        }, index=index)

        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
        df['day'] = df.index.day

        print("dataframe created: ", df['time_series'])
        self.viz(df['time_series'], df.index)

        cols = [col for col in df.columns if col != 'time_series']
        return df[cols], df['time_series']
    
    def preprocess(self):
        return NotImplemented("TODO")

    def viz(self, series, date_time_index: pd.DatetimeIndex):
        plt.figure(figsize=(12, 6))
        plt.plot(date_time_index, series, color='royalblue', linewidth=2)
        plt.title("Time Series Visualization", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

class SyntheticTimeSeriesTensorDataset(Dataset):
    def __init__(self, config, X, y):
        self.config = config
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X) - self.config.seq_length - self.config.horizon + 1

    def __getitem__(self, index):
        ## X_seq = index ==> index + window_size
        x_seq = self.X[index : index + self.config.seq_length]

        ## y_seq = index + seq_length ==> index + seq_length + horizon
        y_seq = self.y[index + self.config.seq_length : index + self.config.seq_length + self.config.horizon]

        return torch.tensor(x_seq.values, dtype=torch.float32), torch.tensor(y_seq.values, dtype=torch.float32)


if __name__ == "__main__":
    config = DatasetConfig(
        date_start='2021/10/11 00:00:00', 
        date_end='2021/11/11 00:00:00',
        sampling='H',
        setting='simple',
        seq_length=24,
        horizon=1,
        split_ratio=0.2
    )
    syn_data = SyntheticTimeSeriesDataset(
        config = config
    )

