"""A small recurrent network that reads the last three days of hourly solar wind and Kp.

One network predicts all lead times at once. Inputs are standardized with statistics from
the rows it's fitted on; gaps become the training mean.
"""

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class StormGRU(torch.nn.Module):
    """A GRU over the hourly sequence, joined with the daily features, then a small head per lead."""

    def __init__(self, channels, daily_features, leads, hidden):
        super().__init__()
        self.gru = torch.nn.GRU(channels, hidden, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + daily_features, 32), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(32, leads))

    def forward(self, sequence, daily):
        _, last = self.gru(sequence)
        return self.head(torch.cat([last[-1], daily], dim=1))


def fit_predict(sequences, daily, targets, fit_rows, stop_rows, predict_rows, hidden=32, seeds=(0, 1, 2),
                max_epochs=40, patience=5):
    """Train on `fit_rows`, stop early on `stop_rows`' loss, and return storm probabilities for
    `predict_rows` (one column per lead), averaged over a few random seeds."""
    imputer = SimpleImputer(strategy="median").fit(daily[fit_rows])
    scaler = StandardScaler().fit(imputer.transform(daily[fit_rows]))
    x = torch.from_numpy(scaler.transform(imputer.transform(daily)).astype(np.float32))
    mean = np.nanmean(sequences[fit_rows], axis=(0, 1))
    std = np.nanstd(sequences[fit_rows], axis=(0, 1)) + 1e-6
    s = torch.from_numpy(np.nan_to_num((sequences - mean) / std).astype(np.float32))
    y = torch.from_numpy(targets.astype(np.float32))
    fit_idx, stop_idx = np.flatnonzero(fit_rows), np.flatnonzero(stop_rows)

    predictions = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = StormGRU(s.shape[2], x.shape[1], y.shape[1], hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, waited = np.inf, None, 0
        for _ in range(max_epochs):
            model.train()
            for batch in np.array_split(rng.permutation(fit_idx), max(1, len(fit_idx) // 128)):
                optimizer.zero_grad()
                loss_fn(model(s[batch], x[batch]), y[batch]).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                stop_loss = loss_fn(model(s[stop_idx], x[stop_idx]), y[stop_idx]).item()
            if stop_loss < best - 1e-4:
                best, best_state, waited = stop_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                waited += 1
                if waited >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions.append(torch.sigmoid(model(s[predict_rows], x[predict_rows])).numpy())
    return np.mean(predictions, axis=0)
