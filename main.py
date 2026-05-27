import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import yfinance as yf
except ImportError:
    yf = None


FIGURE_DIR = Path("figures")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def show_or_close_figure():
    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close()
    else:
        plt.show()


def parse_yyyymmdd(date_text):
    return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"


def fetch_stock_data(stock_code, start_date, end_date):
    if ak is not None:
        try:
            data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if data is not None and not data.empty:
                print("当前使用 akshare 获取数据。")
                return data
        except Exception as e:
            print(f"akshare 获取数据失败，改用 yfinance。原因：{e}")

    if yf is not None:
        suffix = ".SZ" if stock_code.startswith(("0", "3")) else ".SS"
        ticker = f"{stock_code}{suffix}"
        data = yf.download(
            ticker,
            start=parse_yyyymmdd(start_date),
            end=parse_yyyymmdd(end_date),
            auto_adjust=False,
            progress=False,
        )
        if data is not None and not data.empty:
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            data = data.rename(
                columns={
                    "Date": "日期",
                    "Open": "开盘",
                    "Close": "收盘",
                    "High": "最高",
                    "Low": "最低",
                    "Volume": "成交量",
                }
            )
            print("当前使用 yfinance 获取数据。")
            return data

    raise ImportError("未安装 akshare 或 yfinance，无法获取数据。")


def get_close_price_frame(data):
    if "收盘" in data.columns:
        return data[["收盘"]].astype(float)
    if "close" in data.columns:
        return data[["close"]].rename(columns={"close": "收盘"}).astype(float)
    if "Close" in data.columns:
        return data[["Close"]].rename(columns={"Close": "收盘"}).astype(float)
    raise KeyError("找不到收盘价列，请检查数据列名。")


def get_date_series(data):
    if "日期" in data.columns:
        return pd.to_datetime(data["日期"])
    if "Date" in data.columns:
        return pd.to_datetime(data["Date"])
    raise KeyError("找不到日期列，请检查数据列名。")


def explore_and_plot(data, close_df):
    print("数据前 5 行：")
    print(data.head())
    print("\n数据后 5 行：")
    print(data.tail())
    print("\n数据统计摘要：")
    print(data.describe(include="all"))

    FIGURE_DIR.mkdir(exist_ok=True)
    figure_path = FIGURE_DIR / "stock_price_history.png"
    plt.figure(figsize=(14, 5))
    plt.plot(close_df["收盘"].values, label="Actual Price")
    plt.title("Stock Price History")
    plt.xlabel("Trading Day")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    show_or_close_figure()
    return figure_path


def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def preprocess_data(data, time_step=60, test_ratio=0.2):
    close_df = get_close_price_frame(data)

    if len(close_df) <= time_step + 1:
        raise ValueError(f"数据量不足，至少需要超过 {time_step + 1} 条收盘价记录。")

    split_index = int(len(close_df) * (1 - test_ratio))
    if split_index <= time_step:
        raise ValueError("训练集过小，无法构造有效的时间窗口。")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_df.iloc[:split_index].values)
    scaled_data = scaler.transform(close_df.values)

    X_train, y_train = create_dataset(scaled_data[:split_index], time_step=time_step)

    X_test, y_test = [], []
    for target_idx in range(split_index, len(scaled_data)):
        start_idx = target_idx - time_step
        X_test.append(scaled_data[start_idx:target_idx, 0])
        y_test.append(scaled_data[target_idx, 0])

    X_train = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(np.array(X_test)).unsqueeze(-1)
    y_test = torch.FloatTensor(np.array(y_test))

    return X_train, y_train, X_test, y_test, scaler, close_df, split_index


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, dropout=0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(25, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear1(x[:, -1, :])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def train_model(
    X_train,
    y_train,
    model_name="lstm_base",
    hidden_size=50,
    dropout=0.0,
    epochs=20,
    batch_size=64,
    lr=0.001,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(hidden_size=hidden_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)
        print(f"[{model_name}] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    FIGURE_DIR.mkdir(exist_ok=True)
    figure_path = FIGURE_DIR / f"{model_name}_loss_curve.png"
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, label="Training Loss")
    plt.title(f"Training Loss Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    show_or_close_figure()

    return model, losses, figure_path


def predict_scaled(model, X, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        return model(X.to(device)).cpu().numpy()


def evaluate_predictions(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def predict_and_visualize(
    model,
    X_train,
    X_test,
    scaler,
    close_df,
    split_index,
    model_name="lstm_base",
    time_step=60,
    device=None,
):
    train_predict = predict_scaled(model, X_train, device=device)
    test_predict = predict_scaled(model, X_test, device=device)

    train_predict_price = scaler.inverse_transform(train_predict).flatten()
    test_predict_price = scaler.inverse_transform(test_predict).flatten()

    train_actual = close_df.iloc[time_step:split_index, 0].values
    test_actual = close_df.iloc[split_index:, 0].values

    train_metrics = evaluate_predictions(train_actual, train_predict_price)
    test_metrics = evaluate_predictions(test_actual, test_predict_price)

    print("\n训练集评估：", train_metrics)
    print("测试集评估：", test_metrics)

    FIGURE_DIR.mkdir(exist_ok=True)
    figure_path = FIGURE_DIR / f"{model_name}_prediction.png"
    plt.figure(figsize=(14, 5))
    plt.plot(close_df.iloc[:, 0].values, label="Actual Price")
    plt.plot(np.arange(time_step, split_index), train_predict_price, label="Train Predicted Price")
    plt.plot(
        np.arange(split_index, split_index + len(test_predict_price)),
        test_predict_price,
        label="Test Predicted Price",
    )
    plt.title(f"Stock Price Prediction - {model_name}")
    plt.xlabel("Trading Day")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    show_or_close_figure()

    return train_metrics, test_metrics, figure_path


def predict_recent_rolling_prices(
    model,
    date_series,
    close_df,
    scaler,
    future_days=30,
    time_step=60,
    model_name="lstm_base",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    required_points = time_step + future_days
    if len(close_df) < required_points:
        raise ValueError(f"数据量不足，至少需要 {required_points} 条记录。")

    recent_close = close_df.iloc[-required_points:].copy()
    recent_dates = date_series.iloc[-required_points:].reset_index(drop=True)
    recent_scaled = scaler.transform(recent_close.values)

    rolling_inputs = []
    rolling_actual = []
    for idx in range(future_days):
        rolling_inputs.append(recent_scaled[idx : idx + time_step, 0])
        rolling_actual.append(recent_close.iloc[idx + time_step, 0])

    X_recent = torch.FloatTensor(np.array(rolling_inputs)).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        recent_predict_scaled = model(X_recent.to(device)).cpu().numpy()

    future_prices = scaler.inverse_transform(recent_predict_scaled).flatten()
    actual_prices = np.array(rolling_actual, dtype=float)
    target_dates = date_series.iloc[-future_days:].reset_index(drop=True)

    FIGURE_DIR.mkdir(exist_ok=True)
    figure_path = FIGURE_DIR / f"{model_name}_rolling_30_days.png"
    plt.figure(figsize=(14, 5))
    plt.plot(target_dates, actual_prices, label="Actual Price")
    plt.plot(target_dates, future_prices, label="Predicted Price")
    plt.title("Rolling 30-Day Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    show_or_close_figure()

    return future_prices, actual_prices, figure_path


def main():
    stock_code = "000001"
    start_date = "20200101"
    end_date = "20250101"
    time_step = 60
    test_ratio = 0.2
    epochs = 20
    batch_size = 64
    lr = 0.001
    future_days = 30

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练设备：{device}")

    data = fetch_stock_data(stock_code, start_date, end_date)
    date_series = get_date_series(data)
    close_df = get_close_price_frame(data)
    explore_and_plot(data, close_df)

    X_train, y_train, X_test, y_test, scaler, close_df, split_index = preprocess_data(
        data, time_step=time_step, test_ratio=test_ratio
    )
    print(f"\nX_train: {tuple(X_train.shape)}, X_test: {tuple(X_test.shape)}")

    model, _, _ = train_model(
        X_train,
        y_train,
        model_name="lstm_base",
        hidden_size=50,
        dropout=0.0,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    train_metrics, test_metrics, _ = predict_and_visualize(
        model,
        X_train,
        X_test,
        scaler,
        close_df,
        split_index,
        model_name="lstm_base",
        time_step=time_step,
        device=device,
    )

    future_prices, future_actual, _ = predict_recent_rolling_prices(
        model,
        date_series,
        close_df,
        scaler,
        future_days=future_days,
        time_step=time_step,
        model_name="lstm_base",
        device=device,
    )

    print("\n最近 30 个交易日滚动预测：")
    for day, (actual, predicted) in enumerate(zip(future_actual, future_prices), start=1):
        print(f"第 {day:02d} 天，实际收盘价：{actual:.4f}，预测收盘价：{predicted:.4f}")

    print("\n开始比较两个模型：")
    print("基础模型测试集评估：", test_metrics)

    set_seed()
    better_model, _, _ = train_model(
        X_train,
        y_train,
        model_name="lstm_hidden64_dropout",
        hidden_size=64,
        dropout=0.2,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    _, better_test_metrics, _ = predict_and_visualize(
        better_model,
        X_train,
        X_test,
        scaler,
        close_df,
        split_index,
        model_name="lstm_hidden64_dropout",
        time_step=time_step,
        device=device,
    )
    print("改进模型测试集评估：", better_test_metrics)


if __name__ == "__main__":
    main()
