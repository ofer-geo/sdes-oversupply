"""Neural forecasters of parked scooters per cell.

Every cell's recent history is encoded by one shared GRU. Without graphs each cell is then
forecast on its own. With graphs, which makes the model a spatio-temporal graph neural network
(GNN), cells also receive information from the cells they touch and optionally from the cells
that send them scooters. All variants output a Poisson rate for every cell and horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from oversupply.features import MAX_ALLOWED_HORIZON, Panel

WINDOW = 24            # hours of history per cell
N_DYNAMIC = 7          # parked, arrivals, departures, provider, city total, hour sin/cos
N_NODE_HORIZON = 3     # usual level, yesterday and last week at the target hour
N_GLOBAL_HORIZON = 12  # hour sin/cos, weekday one-hot, holiday, eve, summer
N_WEATHER = 4


def normalized_adjacency(adjacency: sp.csr_matrix, device: str) -> torch.Tensor:
    """Sparse matrix giving each cell the weighted average over its linked cells."""
    a = adjacency.tocoo()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    values = a.data.astype(np.float32) / degree[a.row]
    indices = torch.as_tensor(np.vstack([a.row, a.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.as_tensor(values, dtype=torch.float32),
                                   a.shape, device=device,
                                   check_invariants=False).coalesce()


def load_graphs(kind: str, adjacency: sp.csr_matrix, flow_path, device: str) -> list[torch.Tensor]:
    """The graphs of a model variant: none (gru), touching cells (gnn), plus flows (gnn_flow)."""
    graphs = []
    if kind in ("gnn", "gnn_flow"):
        graphs.append(normalized_adjacency(adjacency, device))
    if kind == "gnn_flow":
        graphs.append(normalized_adjacency(sp.load_npz(flow_path), device))
    return graphs


def busy_weights(y: torch.Tensor, strength: float, cap: float = 40.0) -> torch.Tensor:
    """Loss weight per target: 1 for an empty cell, growing with the count up to ``cap``."""
    return 1.0 + y.clamp(max=cap) * strength


class TensorPanel:
    """The panel on the training device, producing leak-free inputs for a batch of origins."""

    def __init__(self, panel: Panel, horizons: int, device: str):
        self.horizons = horizons
        self.device = device

        def log_tensor(matrix: np.ndarray) -> torch.Tensor:
            return torch.log1p(torch.as_tensor(matrix, dtype=torch.float32, device=device))

        self.parked_raw = torch.as_tensor(panel.parked.astype(np.float32), device=device)
        self.parked = torch.log1p(self.parked_raw)
        self.arrivals = log_tensor(panel.arrivals.astype(np.float32))
        self.departures = log_tensor(panel.departures.astype(np.float32))
        self.provider = log_tensor(panel.provider.astype(np.float32))
        self.profile = log_tensor(panel.profile)
        self.city = torch.log1p(torch.as_tensor(panel.city_parked, dtype=torch.float32,
                                                device=device)) / 10.0
        self.slot = torch.as_tensor(panel.slot, dtype=torch.long, device=device)

        calendar = panel.calendar
        hour = torch.as_tensor(calendar["hour"].to_numpy(copy=True), dtype=torch.float32,
                               device=device)
        angle = 2 * torch.pi * hour / 24
        self.hour_sin, self.hour_cos = torch.sin(angle), torch.cos(angle)
        dow = torch.as_tensor(calendar["dow"].to_numpy().astype(np.int64), device=device)
        flags = torch.as_tensor(
            calendar[["is_holiday", "is_holiday_eve", "school_summer"]].to_numpy(
                dtype=np.float32, copy=True),
            device=device)
        self.global_features = torch.cat([
            self.hour_sin[:, None], self.hour_cos[:, None],
            nn.functional.one_hot(dow, 7).float(), flags], dim=1)

        weather = panel.weather.to_numpy(dtype=np.float32)
        weather = (weather - weather.mean(0)) / (weather.std(0) + 1e-6)
        self.weather = torch.as_tensor(weather, device=device)

        statics = panel.statics.copy()
        for column in ("dist_to_dock_m", "dist_to_infra_m"):
            statics[column] = np.log1p(statics[column])
        values = statics.to_numpy(dtype=np.float32)
        values = (values - values.mean(0)) / (values.std(0) + 1e-6)
        self.statics = torch.as_tensor(values, device=device)
        self.n_nodes, self.n_static = values.shape[0], values.shape[1]

    def batch(self, origins: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inputs for origins (B). Observed values never come from after the origin."""
        if self.horizons > MAX_ALLOWED_HORIZON:
            raise ValueError("Horizon too long: the daily lag would look into the future")
        steps = torch.arange(WINDOW - 1, -1, -1, device=self.device)
        window = origins[:, None] - steps                                 # B x W
        dynamic = torch.stack([
            self.parked[:, window], self.arrivals[:, window], self.departures[:, window],
            self.provider[:, window],
            self.city[window][None].expand(self.n_nodes, -1, -1),
            self.hour_sin[window][None].expand(self.n_nodes, -1, -1),
            self.hour_cos[window][None].expand(self.n_nodes, -1, -1),
        ], dim=-1).permute(1, 0, 2, 3)                                    # B x N x W x C

        ahead = torch.arange(1, self.horizons + 1, device=self.device)
        targets = origins[:, None] + ahead                                # B x H
        node_horizon = torch.stack([
            self.profile[:, self.slot[targets]],
            self.parked[:, targets - 24],
            self.parked[:, targets - 168],
        ], dim=-1).permute(1, 0, 2, 3)                                    # B x N x H x 3
        return {
            "dynamic": dynamic,
            "node_horizon": node_horizon,
            "global_horizon": self.global_features[targets],              # B x H x 12
            "weather": self.weather[origins],                             # B x 4
            "last_parked": self.parked[:, origins].T,                     # B x N
        }

    def targets(self, origins: torch.Tensor) -> torch.Tensor:
        ahead = torch.arange(1, self.horizons + 1, device=self.device)
        return self.parked_raw[:, origins[:, None] + ahead].permute(1, 0, 2)  # B x N x H


@torch.no_grad()
def predict_rates(model: nn.Module, tensors: TensorPanel, origins: np.ndarray,
                  batch_size: int = 16) -> np.ndarray:
    """Forecast rates as origins x cells x horizons."""
    model.eval()
    parts = []
    for start in range(0, len(origins), batch_size):
        chunk = torch.as_tensor(origins[start:start + batch_size], device=tensors.device)
        parts.append(torch.exp(model(tensors.batch(chunk))).cpu().numpy())
    return np.concatenate(parts)


class GraphLayer(nn.Module):
    """Updates each cell from its links in one or more graphs."""

    def __init__(self, hidden: int, n_graphs: int, dropout: float):
        super().__init__()
        self.mix = nn.Linear((1 + n_graphs) * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, graphs: list[torch.Tensor]) -> torch.Tensor:
        b, n, d = h.shape
        flat = h.permute(1, 0, 2).reshape(n, b * d)
        messages = [torch.sparse.mm(g, flat).reshape(n, b, d).permute(1, 0, 2) for g in graphs]
        update = torch.relu(self.mix(torch.cat([h, *messages], dim=-1)))
        return self.norm(h + self.dropout(update))


@dataclass
class ModelSize:
    hidden: int = 64
    embedding: int = 16
    graph_layers: int = 2
    dropout: float = 0.1


class Forecaster(nn.Module):
    def __init__(self, n_nodes: int, n_static: int, horizons: int, statics: torch.Tensor,
                 graphs: list[torch.Tensor], size: ModelSize):
        super().__init__()
        d = size.hidden
        self.horizons = horizons
        self.register_buffer("statics", statics)
        self.graphs = graphs
        self.encoder = nn.GRU(N_DYNAMIC, d, batch_first=True)
        self.node_embedding = nn.Embedding(n_nodes, size.embedding)
        self.node_mix = nn.Linear(d + size.embedding + n_static, d)
        layers = size.graph_layers if graphs else 0
        self.graph = nn.ModuleList(GraphLayer(d, len(graphs), size.dropout)
                                   for _ in range(layers))
        self.horizon_embedding = nn.Embedding(horizons, 8)
        self.decoder = nn.Sequential(
            nn.Linear(d + N_NODE_HORIZON + N_GLOBAL_HORIZON + N_WEATHER + 8 + 1, d),
            nn.ReLU(), nn.Dropout(size.dropout), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Log rate per batch, cell and horizon (B x N x H)."""
        b, n, w, c = x["dynamic"].shape
        _, last = self.encoder(x["dynamic"].reshape(b * n, w, c))
        h = last[0].reshape(b, n, -1)
        identity = self.node_embedding.weight[None].expand(b, -1, -1)
        statics = self.statics[None].expand(b, -1, -1)
        h = torch.relu(self.node_mix(torch.cat([h, identity, statics], dim=-1)))
        for layer in self.graph:
            h = layer(h, self.graphs)

        H = self.horizons
        ahead = self.horizon_embedding.weight[None, None].expand(b, n, -1, -1)
        decoded = self.decoder(torch.cat([
            h[:, :, None].expand(-1, -1, H, -1),
            x["node_horizon"],
            x["global_horizon"][:, None].expand(-1, n, -1, -1),
            x["weather"][:, None, None].expand(-1, n, H, -1),
            ahead,
            x["last_parked"][:, :, None, None].expand(-1, -1, H, -1),
        ], dim=-1))
        return decoded.squeeze(-1).clamp(-8.0, 7.0)
