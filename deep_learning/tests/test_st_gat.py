"""ST-GAT: the graph switch must change only information flow, never model size."""
import torch

from deep_learning.dl import engine
from deep_learning.tests.test_models import ADJ, tiny_data
from deep_learning.dl.models import build


def test_graph_switch_keeps_parameters_and_changes_information_flow():
    data = tiny_data()
    torch.manual_seed(0); with_graph = build("st_gat", ADJ, d_model=8, dropout=0.0, attn_dropout=0.0).eval()
    torch.manual_seed(0); no_graph = build("st_gat_noedge", ADJ, d_model=8, dropout=0.0, attn_dropout=0.0).eval()
    assert engine.count_parameters(with_graph) == engine.count_parameters(no_graph)
    b = data.batch(data.origins["train"][:2])
    poked = {k: v.clone() for k, v in b.items()}
    poked["x"][:, :, 0, 0] += 3.0                      # change only sensor 0's history
    for model, should_move in ((no_graph, False), (with_graph, True)):
        delta = (model(poked) - model(b)).abs().amax(dim=(0, 1))
        assert (delta[2] > 1e-6) == should_move          # sensor 2 is two hops downstream of sensor 0
        assert delta[4] < 1e-6                           # the isolated sensor never moves
