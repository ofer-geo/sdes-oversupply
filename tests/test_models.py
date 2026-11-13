import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from oversupply.features import Panel, calendar_frame  # noqa: E402
from oversupply.models import (  # noqa: E402
    Forecaster,
    ModelSize,
    TensorPanel,
    busy_weights,
    normalized_adjacency,
)

N_CELLS, N_HOURS, H = 4, 400, 6
ADJACENCY = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]))


def small_panel() -> Panel:
    rng = np.random.default_rng(0)
    times = pd.Series(pd.date_range("2024-01-01", periods=N_HOURS, freq="h"))

    def m() -> np.ndarray:
        return rng.integers(0, 20, (N_CELLS, N_HOURS)).astype(np.int16)

    return Panel(m(), m(), m(), m(), m(), m(),
                 pd.DataFrame({"dist_to_dock_m": rng.random(N_CELLS),
                               "dist_to_infra_m": rng.random(N_CELLS)}),
                 calendar_frame(times),
                 pd.DataFrame({"temperature_2m": rng.random(N_HOURS),
                               "precipitation": rng.random(N_HOURS),
                               "wind_speed_10m": rng.random(N_HOURS),
                               "relative_humidity_2m": rng.random(N_HOURS)}),
                 rng.random((N_CELLS, 168)).astype(np.float32))


def build(n_graphs: int):
    tensors = TensorPanel(small_panel(), H, "cpu")
    torch.manual_seed(0)
    graphs = [normalized_adjacency(ADJACENCY, "cpu")] * n_graphs
    model = Forecaster(N_CELLS, tensors.n_static, H, tensors.statics, graphs,
                       ModelSize(hidden=8, embedding=4, graph_layers=2, dropout=0.0))
    return tensors, model.eval()


@pytest.mark.parametrize("n_graphs", [0, 1, 2])
def test_output_has_one_rate_per_batch_cell_and_horizon(n_graphs):
    tensors, model = build(n_graphs)
    origins = torch.tensor([200, 250, 300])
    out = model(tensors.batch(origins))
    assert out.shape == (3, N_CELLS, H)
    assert torch.isfinite(out).all()
    assert tensors.targets(origins).shape == (3, N_CELLS, H)


def test_targets_are_the_counts_ahead_of_the_origin():
    tensors, _ = build(0)
    y = tensors.targets(torch.tensor([200]))
    assert y[0, 2, 0] == tensors.parked_raw[2, 201]
    assert y[0, 2, 5] == tensors.parked_raw[2, 206]


def test_inputs_ignore_everything_after_the_origin():
    tensors, _ = build(2)
    origin = torch.tensor([300])
    before = {k: v.clone() for k, v in tensors.batch(origin).items()}
    for matrix in (tensors.parked, tensors.arrivals, tensors.departures, tensors.provider):
        matrix[:, 301:] = 99.0
    tensors.city[301:] = 99.0
    tensors.weather[301:] = 99.0
    after = tensors.batch(origin)
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_graph_layers_share_information_between_neighbours():
    tensors, gnn = build(1)
    origin = torch.tensor([300])
    x = tensors.batch(origin)
    changed = {k: v.clone() for k, v in x.items()}
    changed["dynamic"][0, 0] += 5.0              # only cell 0's history
    delta = (gnn(changed) - gnn(x)).abs().sum(dim=(0, 2))
    assert delta[1] > 0                          # its neighbour responds
    assert delta[3] == 0                         # three steps away: out of reach of two layers

    tensors, gru = build(0)
    delta = (gru(changed) - gru(x)).abs().sum(dim=(0, 2))
    assert delta[0] > 0 and delta[1] == 0        # without a graph, cells stay independent


def test_adjacency_rows_average_their_links_by_weight():
    weights = sp.csr_matrix(np.array([[0, 3, 1], [0, 0, 0], [2, 0, 0]], dtype=np.float32))
    a = normalized_adjacency(weights, "cpu").to_dense()
    assert a.tolist() == [[0.0, 0.75, 0.25], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_busy_weights_grow_with_the_count_up_to_the_cap():
    y = torch.tensor([0.0, 10.0, 40.0, 100.0])
    assert busy_weights(y, 0.2).tolist() == [1.0, 3.0, 9.0, 9.0]


def test_too_long_horizon_is_refused():
    tensors = TensorPanel(small_panel(), 25, "cpu")
    with pytest.raises(ValueError):
        tensors.batch(torch.tensor([200]))
