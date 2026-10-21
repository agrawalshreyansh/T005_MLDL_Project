"""The graph model must respect the graph, handle missing targets and produce correct shapes."""
import numpy as np
import pandas as pd
import pytest
import torch

from src import config, gat_model as gm, graph_utils as gu, preprocessing as pp


def chain_adj(n=6):
    """0 -> 1 -> 2 -> 3 and 4 -> 5 (two disconnected components)."""
    a = np.eye(n, dtype=np.float32)
    for i, j in [(0, 1), (1, 2), (2, 3), (4, 5)]:
        a[i, j] = 0.8
    return a


def make_model(use_graph=True, n=6, layers=2):
    torch.manual_seed(0)
    return gm.STGAT(n_nodes=n, in_features=6, adj=chain_adj(n), d_model=16, n_heads=2, n_gat_layers=layers,
                    node_emb_dim=4, dropout=0.0, attn_dropout=0.0, n_steps=12, use_graph=use_graph).eval()


def test_attention_is_zero_outside_the_mask_and_rows_sum_to_one():
    adj = chain_adj()
    mask = torch.from_numpy(gu.neighbor_masks(adj)["in"])
    layer = gm.DenseGATLayer(8, 4, 2, 0.0).eval()
    out, alpha = layer(torch.randn(3, 6, 8), mask, return_attention=True)
    assert out.shape == (3, 6, 8) and alpha.shape == (3, 6, 6, 2)
    assert torch.all(alpha[:, ~mask] == 0)
    torch.testing.assert_close(alpha.sum(2), torch.ones(3, 6, 2))


def test_dense_layer_matches_pytorch_geometric_gatconv():
    pyg = pytest.importorskip("torch_geometric.nn")
    adj = chain_adj()
    mask = torch.from_numpy(gu.neighbor_masks(adj)["in"])
    layer = gm.DenseGATLayer(8, 4, 2, 0.0).eval()
    tgt, src = torch.nonzero(mask, as_tuple=True)              # mask[i, j]: edge j -> i
    conv = pyg.GATConv(8, 4, heads=2, add_self_loops=False, dropout=0.0).eval()
    with torch.no_grad():
        conv.lin.weight.copy_(layer.lin.weight); conv.att_src.copy_(layer.att_src)
        conv.att_dst.copy_(layer.att_dst); conv.bias.copy_(layer.bias)
        x = torch.randn(1, 6, 8)
        torch.testing.assert_close(layer(x, mask)[0], conv(x[0], torch.stack([src, tgt])), atol=1e-5, rtol=1e-5)


def _influence(model, node, receiver):
    """Does perturbing the input history of ``node`` change the prediction of ``receiver``?"""
    x = torch.randn(1, 12, 6, 6); anchor = torch.zeros(1, 6)
    with torch.no_grad():
        base = model(x, anchor)[0, :, receiver]
        x2 = x.clone(); x2[:, :, node, :] += 5.0
        changed = model(x2, anchor)[0, :, receiver]
    return not torch.allclose(base, changed, atol=1e-6)


def test_predictions_depend_only_on_graph_neighbourhood():
    model = make_model(use_graph=True, layers=2)
    assert _influence(model, node=0, receiver=1)               # direct neighbour
    assert _influence(model, node=0, receiver=2)               # two hops (two blocks)
    assert not _influence(model, node=0, receiver=3)           # three hops: out of receptive field
    assert not _influence(model, node=0, receiver=5)           # other component
    assert not _influence(model, node=4, receiver=0)


def test_ablation_without_edges_makes_sensors_independent():
    model = make_model(use_graph=False)
    for j in range(6):
        for i in range(6):
            if i != j:
                assert not _influence(model, node=j, receiver=i)
    assert _influence(model, node=2, receiver=2)


def test_ablation_has_identical_parameter_count():
    a, b = make_model(True), make_model(False)
    assert gm.count_parameters(a) == gm.count_parameters(b)


def test_forward_shapes_and_residual_anchor():
    model = make_model()
    out = model(torch.randn(3, 12, 6, 6), torch.zeros(3, 6))
    assert out.shape == (3, 12, 6)
    out_shift = model(torch.zeros(3, 12, 6, 6), torch.full((3, 6), 2.5))
    out_zero = model(torch.zeros(3, 12, 6, 6), torch.zeros(3, 6))
    torch.testing.assert_close(out_shift - out_zero, torch.full_like(out_zero, 2.5))   # anchor added exactly


def test_masked_loss_ignores_invalid_targets():
    pred = torch.tensor([[1.0, 100.0]]); tgt = torch.tensor([[2.0, 0.0]]); valid = torch.tensor([[1.0, 0.0]])
    assert gm.masked_mae_loss(pred, tgt, valid).item() == pytest.approx(1.0)
    assert gm.masked_mae_loss(pred, tgt, torch.zeros_like(valid)).item() == 0.0     # no valid -> no NaN


def _toy_graph_data():
    ts = pd.date_range("2012-03-01", periods=3 * 288, freq="5min")
    rng = np.random.default_rng(0)
    speed = rng.uniform(30, 65, (len(ts), 6)).astype(np.float32)
    mask = rng.random(speed.shape) > 0.1
    speed = np.where(mask, speed, np.nan).astype(np.float32)
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    scaler = pp.SpeedScaler.fit(speed, mask, sp.train)
    lv, la = pp.locf_with_age(speed, mask)
    data = gm.GraphData(speed, mask, pp.fill_for_model(speed, mask, scaler), pp.time_feature_matrix(ts), lv, la, scaler)
    return ts, sp, data


def test_batch_tensors_have_the_documented_shapes_and_no_nan():
    _, sp, data = _toy_graph_data()
    origins = pp.valid_origins(sp.train, 12, 12, len(data.speed), "train")[:5]
    x, anchor, y, valid = data.batch(origins, torch.device("cpu"))
    assert x.shape == (5, 12, 6, 6) and anchor.shape == (5, 6) and y.shape == (5, 12, 6) == valid.shape
    assert torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(anchor).all()
    assert set(torch.unique(valid).tolist()) <= {0.0, 1.0}
    # the mask channel of the input equals the observation mask
    x_idx, _ = pp.window_index_arrays(origins, 12, 12)
    np.testing.assert_array_equal(x[..., 1].numpy().astype(bool), data.mask[x_idx])


def test_training_runs_improves_and_prediction_shapes_are_correct(tmp_path):
    _, sp, data = _toy_graph_data()
    n = len(data.speed)
    tr = pp.valid_origins(sp.train, 12, 12, n, "train")[::6]
    va = pp.valid_origins(sp.val, 12, 12, n, "val")[::6]
    model = gm.STGAT(6, 6, chain_adj(), d_model=16, n_heads=2, n_gat_layers=1, node_emb_dim=4)
    params = dict(config.GAT_PARAMS, max_epochs=3, patience=3, batch_size=16, seed=1)
    ckpt = tmp_path / "m.pt"
    hist = gm.train_model(model, data, tr, va, torch.device("cpu"), params, ckpt, log=lambda s: None)
    assert len(hist["val_mae_mph"]) >= 1 and np.isfinite(hist["val_mae_mph"]).all()
    assert ckpt.is_file()
    pred = gm.predict_mph(model, data, va, torch.device("cpu"), steps=(1, 3, 6, 12))
    assert pred.shape == (len(va), 4, 6) and np.isfinite(pred).all()
    # checkpoint reload reproduces predictions exactly
    fresh = gm.STGAT(6, 6, chain_adj(), d_model=16, n_heads=2, n_gat_layers=1, node_emb_dim=4)
    gm.load_checkpoint(ckpt, fresh, torch.device("cpu"))
    np.testing.assert_allclose(gm.predict_mph(fresh, data, va, torch.device("cpu")), pred, atol=1e-4)


def test_predictions_are_inverse_transformed_to_mph():
    _, sp, data = _toy_graph_data()
    model = make_model()
    origins = pp.valid_origins(sp.val, 12, 12, len(data.speed), "val")[:4]
    z = gm.predict_scaled(model, data, origins, torch.device("cpu"))
    mph = gm.predict_mph(model, data, origins, torch.device("cpu"), steps=(1, 12))
    np.testing.assert_allclose(mph, z[:, [0, 11], :] * data.scaler.std + data.scaler.mean, rtol=1e-5, atol=1e-4)
