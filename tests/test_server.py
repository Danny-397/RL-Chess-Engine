"""The web API caps what a request may ask the searcher to do."""

import chess

import search
from web import server


def test_depth_and_top_n_are_capped(monkeypatch):
    seen = {}
    monkeypatch.setattr(search, "analyze",
                        lambda board, depth, top_n: seen.update(depth=depth, top_n=top_n) or {})
    server._analyse(server.PositionRequest(fen=chess.STARTING_FEN, depth=50, top_n=10_000))
    assert seen == {"depth": server._MAX_DEPTH, "top_n": server._MAX_TOP_N}
    server._analyse(server.PositionRequest(fen=chess.STARTING_FEN, depth=-3, top_n=0))
    assert seen == {"depth": 1, "top_n": 1}


def test_the_ui_depths_pass_through_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(search, "analyze", lambda board, depth, top_n: seen.append(depth) or {})
    for depth in (2, 3, 4):
        server._analyse(server.PositionRequest(fen=chess.STARTING_FEN, depth=depth))
    assert seen == [2, 3, 4]


def test_a_bad_fen_is_a_400_not_a_crash():
    from fastapi.testclient import TestClient

    r = TestClient(server.app).post("/analyze", json={"fen": "not a position"})
    assert r.status_code == 400 and r.json() == {"detail": "invalid FEN"}
