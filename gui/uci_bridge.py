from __future__ import annotations

import atexit
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.engine
import numpy as np
import torch
from django.core.cache import cache

PROJECT_ROOT = Path("~/PycharmProjects/chess_hybrid_rl").expanduser().resolve()
BEST_MODEL_PATH = PROJECT_ROOT / "chess_hybrid_rl_runs" / "checkpoints" / "best" / "best_model.pt"
STOCKFISH_PATH = Path("/usr/games/stockfish")

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MCTSConfig, ModelConfig
from src.encoding import BoardEncoder, MoveEncoder
from src.mcts import MCTS
from src.model import PolicyValueNet
from dataclasses import asdict

CACHE_PREFIX = "noshchessrlx"
DEFAULT_SIMULATIONS = 128
WHITE = "white"
BLACK = "black"


@dataclass(slots=True)
class StockfishLine:
    multipv: int
    uci: str
    san: str
    score_text: str
    cp_white: Optional[int]
    mate_white: Optional[int]
    depth: int
    nodes: int
    nps: int


class ChessEngineService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model_lock = threading.RLock()
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.model = self._load_model()
        self.base_mcts_cfg = MCTSConfig(
            simulations=DEFAULT_SIMULATIONS,
            c_puct=1.75,
            dirichlet_alpha=0.30,
            dirichlet_eps=0.25,
            temperature_start=0.0,
            temperature_end=0.0,
            temperature_decay_plies=1,
            root_noise=False,
        )

    def _load_model(self) -> PolicyValueNet:
        model_cfg = ModelConfig(input_planes=18, channels=128, num_blocks=10, value_head_hidden=256, dropout=0.0, action_size=20480)
        model = PolicyValueNet(model_cfg).to(self.device)
        checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"{BEST_MODEL_PATH} does not contain 'model_state_dict'.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        torch.set_float32_matmul_precision("high")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "compile"):
                model = torch.compile(model, mode="reduce-overhead")
        return model

    def choose_move(self, board: chess.Board, simulations: int) -> Dict[str, Any]:
        cfg = MCTSConfig(
            simulations=max(int(simulations), 1),
            c_puct=self.base_mcts_cfg.c_puct,
            dirichlet_alpha=self.base_mcts_cfg.dirichlet_alpha,
            dirichlet_eps=self.base_mcts_cfg.dirichlet_eps,
            temperature_start=0.0,
            temperature_end=0.0,
            temperature_decay_plies=1,
            root_noise=False,
        )
        with self.model_lock:
            mcts = MCTS(
                model=self.model,
                board_encoder=self.board_encoder,
                move_encoder=self.move_encoder,
                config=cfg,
                device=self.device,
                amp_dtype=self.dtype,
            )
            result = mcts.run(board, ply=board.ply())
        move = self.move_encoder.index_to_move(result.action)
        if move not in board.legal_moves:
            legal_moves = list(board.legal_moves)
            move = legal_moves[0]
        board_copy = board.copy(stack=False)
        san = board_copy.san(move)
        return {
            "uci": move.uci(),
            "san": san,
            "root_value": float(result.root_value),
            "nodes": int(result.nodes),
            "simulations": int(simulations),
        }


class TeacherAnalysisService:
    def __init__(self, engine_path: Path) -> None:
        self.engine_path = str(engine_path)
        self.lock = threading.RLock()
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.engine.configure({"Threads": 4, "Hash": 512})

    def close(self) -> None:
        with self.lock:
            try:
                self.engine.quit()
            except Exception:
                pass

    def _cp_for_color(self, score: chess.engine.PovScore, color: chess.Color) -> int:
        cp = score.pov(color).score(mate_score=100000)
        return int(cp if cp is not None else 0)

    def _score_payload(self, score: chess.engine.PovScore) -> Dict[str, Any]:
        cp_white = score.white().score()
        mate_white = score.white().mate()
        if mate_white is not None:
            text = f"M{mate_white}"
            return {
                "kind": "mate",
                "value": int(mate_white),
                "cp_white": None,
                "mate_white": int(mate_white),
                "text": text,
            }
        cp_value = int(cp_white if cp_white is not None else 0)
        text = f"{cp_value / 100.0:+.2f}"
        return {
            "kind": "cp",
            "value": cp_value,
            "cp_white": cp_value,
            "mate_white": None,
            "text": text,
        }

    def _line_to_payload(self, board: chess.Board, info: Dict[str, Any], pv_index: int) -> StockfishLine:
        pv_moves = list(info.get("pv", []))
        temp = board.copy(stack=False)
        san_moves: List[str] = []
        for move in pv_moves[:12]:
            san_moves.append(temp.san(move))
            temp.push(move)
        score = info["score"]
        return StockfishLine(
            multipv=int(info.get("multipv", pv_index)),
            uci=" ".join(move.uci() for move in pv_moves[:12]),
            san=" ".join(san_moves),
            score_text=self._score_payload(score)["text"],
            cp_white=self._score_payload(score)["cp_white"],
            mate_white=self._score_payload(score)["mate_white"],
            depth=int(info.get("depth", 0)),
            nodes=int(info.get("nodes", 0)),
            nps=int(info.get("nps", 0)),
        )

    def analyze(self, board: chess.Board, multipv: int = 3, movetime_ms: int = 120) -> Dict[str, Any]:
        with self.lock:
            limit = chess.engine.Limit(time=max(movetime_ms, 1) / 1000.0)
            info = self.engine.analyse(board, limit, multipv=max(multipv, 1))
        rows = info if isinstance(info, list) else [info]
        rows = [row for row in rows if row.get("pv")]
        if not rows:
            return {
                "score": {"kind": "cp", "value": 0, "cp_white": 0, "mate_white": None, "text": "+0.00"},
                "top_lines": [],
                "depth": 0,
                "nodes": 0,
                "nps": 0,
            }
        main = rows[0]
        lines = [asdict(self._line_to_payload(board, row, idx + 1)) for idx, row in enumerate(rows[:max(multipv, 1)])]
        return {
            "score": self._score_payload(main["score"]),
            "top_lines": lines,
            "depth": int(main.get("depth", 0)),
            "nodes": int(main.get("nodes", 0)),
            "nps": int(main.get("nps", 0)),
        }

    def classify_move(self, board_before: chess.Board, move: chess.Move, multipv: int = 3, movetime_ms: int = 120) -> Dict[str, Any]:
        before_info = self.analyze(board_before, multipv=multipv, movetime_ms=movetime_ms)
        before_cp = before_info["score"]["mate_white"]
        if before_info["score"]["kind"] == "mate":
            before_eval = 100000 if int(before_cp) > 0 else -100000
        else:
            before_eval = int(before_info["score"]["cp_white"] or 0)

        board_after = board_before.copy(stack=False)
        san = board_after.san(move)
        board_after.push(move)
        after_info = self.analyze(board_after, multipv=multipv, movetime_ms=movetime_ms)
        if after_info["score"]["kind"] == "mate":
            after_eval_white = 100000 if int(after_info["score"]["mate_white"]) > 0 else -100000
        else:
            after_eval_white = int(after_info["score"]["cp_white"] or 0)

        mover = board_before.turn
        before_for_mover = before_eval if mover == chess.WHITE else -before_eval
        after_for_mover = after_eval_white if mover == chess.WHITE else -after_eval_white
        cp_loss = max(before_for_mover - after_for_mover, 0)

        best_uci = None
        if before_info["top_lines"]:
            top_uci_line = before_info["top_lines"][0]["uci"].split()
            best_uci = top_uci_line[0] if top_uci_line else None

        if move.uci() == best_uci or cp_loss <= 20:
            label = "Great"
        elif cp_loss >= 250:
            label = "Blunder"
        elif cp_loss >= 80:
            label = "Inaccuracy"
        else:
            label = "Good"

        return {
            "label": label,
            "cp_loss": int(cp_loss),
            "played_move_uci": move.uci(),
            "played_move_san": san,
            "best_move_uci": best_uci,
            "best_move_san": before_info["top_lines"][0]["san"].split()[0] if before_info["top_lines"] and before_info["top_lines"][0]["san"] else None,
            "before": before_info,
            "after": after_info,
        }


class AsyncGameCoordinator:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="noshchessrlx")
        self.tasks: dict[str, Future] = {}
        self.lock = threading.RLock()
        self.engine_service = ChessEngineService()
        self.stockfish_service = TeacherAnalysisService(STOCKFISH_PATH)

    def shutdown(self) -> None:
        self.stockfish_service.close()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def submit_turn_resolution(self, game_state: Dict[str, Any]) -> str:
        task_id = uuid.uuid4().hex
        snapshot = {
            "game_id": game_state["game_id"],
            "version": game_state["version"],
            "moves_uci": list(game_state["moves_uci"]),
            "simulations": int(game_state.get("simulations", DEFAULT_SIMULATIONS)),
            "human_color": game_state.get("human_color", WHITE),
            "last_player_move_uci": game_state.get("last_player_move_uci"),
        }
        future = self.executor.submit(self._resolve_turn, snapshot)
        with self.lock:
            self.tasks[task_id] = future
        return task_id

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        with self.lock:
            future = self.tasks.get(task_id)
        if future is None:
            return {"status": "missing"}
        if not future.done():
            return {"status": "pending"}
        try:
            result = future.result()
            return {"status": "ready", "result": result}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        finally:
            with self.lock:
                self.tasks.pop(task_id, None)

    def _resolve_turn(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        board_before_player = chess.Board()
        if snapshot["moves_uci"]:
            for move_uci in snapshot["moves_uci"][:-1]:
                board_before_player.push_uci(move_uci)
        board_after_player = chess.Board()
        for move_uci in snapshot["moves_uci"]:
            board_after_player.push_uci(move_uci)

        player_move = chess.Move.from_uci(snapshot["last_player_move_uci"])
        classification = self.stockfish_service.classify_move(board_before_player, player_move, multipv=3, movetime_ms=150)
        analysis_after_player = classification["after"]
        player_turn_terminal = board_after_player.is_game_over(claim_draw=True)

        ai_move_payload: Optional[Dict[str, Any]] = None
        analysis_after_ai: Optional[Dict[str, Any]] = None
        moves_uci = list(snapshot["moves_uci"])

        if not player_turn_terminal:
            ai_move_payload = self.engine_service.choose_move(board_after_player.copy(stack=False), snapshot["simulations"])
            board_after_player.push_uci(ai_move_payload["uci"])
            moves_uci.append(ai_move_payload["uci"])
            analysis_after_ai = self.stockfish_service.analyze(board_after_player, multipv=3, movetime_ms=150)

        final_board = board_after_player
        return {
            "version": snapshot["version"],
            "moves_uci": moves_uci,
            "fen": final_board.fen(),
            "analysis": analysis_after_ai or analysis_after_player,
            "player_classification": classification,
            "ai_move": ai_move_payload,
            "is_game_over": final_board.is_game_over(claim_draw=True),
            "result": final_board.result(claim_draw=True) if final_board.is_game_over(claim_draw=True) else None,
            "status_text": "Game over" if final_board.is_game_over(claim_draw=True) else "Your move",
        }


def game_cache_key(game_id: str) -> str:
    return f"{CACHE_PREFIX}:game:{game_id}"


def default_game_state(game_id: str) -> Dict[str, Any]:
    board = chess.Board()
    return {
        "game_id": game_id,
        "version": 1,
        "fen": board.fen(),
        "moves_uci": [],
        "human_color": WHITE,
        "simulations": DEFAULT_SIMULATIONS,
        "pending": False,
        "task_id": None,
        "status_text": "Your move",
        "player_classification": None,
        "analysis": None,
        "ai_move": None,
        "last_player_move_uci": None,
        "result": None,
        "is_game_over": False,
    }


def build_board_from_state(state: Dict[str, Any]) -> chess.Board:
    board = chess.Board()
    for move_uci in state.get("moves_uci", []):
        board.push_uci(move_uci)
    return board


def state_to_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    board = build_board_from_state(state)
    return {
        **state,
        "pgn": moves_to_pgn(state.get("moves_uci", [])),
        "legal_moves": [move.uci() for move in board.legal_moves],
        "turn": WHITE if board.turn == chess.WHITE else BLACK,
    }


def moves_to_pgn(moves_uci: List[str]) -> str:
    board = chess.Board()
    chunks: List[str] = []
    for idx, move_uci in enumerate(moves_uci):
        move = chess.Move.from_uci(move_uci)
        san = board.san(move)
        if idx % 2 == 0:
            chunks.append(f"{(idx // 2) + 1}. {san}")
        else:
            chunks[-1] = f"{chunks[-1]} {san}"
        board.push(move)
    return " ".join(chunks)


def get_or_create_game(session: Dict[str, Any]) -> Dict[str, Any]:
    game_id = session.get("noshchessrlx_game_id")
    if not game_id:
        game_id = uuid.uuid4().hex
        session["noshchessrlx_game_id"] = game_id
        state = default_game_state(game_id)
        cache.set(game_cache_key(game_id), state, timeout=60 * 60 * 24)
        return state
    state = cache.get(game_cache_key(game_id))
    if state is None:
        state = default_game_state(game_id)
        cache.set(game_cache_key(game_id), state, timeout=60 * 60 * 24)
    return state


def save_game_state(state: Dict[str, Any]) -> None:
    cache.set(game_cache_key(state["game_id"]), state, timeout=60 * 60 * 24)


def start_new_game(session: Dict[str, Any], human_color: str = WHITE, simulations: int = DEFAULT_SIMULATIONS) -> Dict[str, Any]:
    game_id = uuid.uuid4().hex
    session["noshchessrlx_game_id"] = game_id
    state = default_game_state(game_id)
    state["human_color"] = human_color if human_color in {WHITE, BLACK} else WHITE
    state["simulations"] = max(int(simulations), 1)
    state["status_text"] = "Your move" if state["human_color"] == WHITE else "Model is thinking"
    save_game_state(state)
    return state


COORDINATOR = AsyncGameCoordinator()
atexit.register(COORDINATOR.shutdown)
