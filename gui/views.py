from __future__ import annotations

from typing import Any, Dict

import chess
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .uci_bridge import (
    COORDINATOR,
    build_board_from_state,
    get_or_create_game,
    save_game_state,
    start_new_game,
    state_to_payload,
)


def _json_error(message: str, status: int = 400, **extra: Any) -> JsonResponse:
    payload: Dict[str, Any] = {"ok": False, "error": message}
    payload.update(extra)
    return JsonResponse(payload, status=status)


@ensure_csrf_cookie
@require_GET
def index(request: HttpRequest):
    state = get_or_create_game(request.session)
    if state.get("analysis") is None and not state.get("pending"):
        board = build_board_from_state(state)
        state["analysis"] = COORDINATOR.stockfish_service.analyze(board, multipv=3, movetime_ms=120)
        save_game_state(state)
    return render(
        request,
        "gui/index.html",
        {
            "initial_state": state_to_payload(state),
            "default_simulations": state.get("simulations", 128),
        },
    )


@require_GET
def game_state(request: HttpRequest) -> JsonResponse:
    state = get_or_create_game(request.session)
    if state.get("analysis") is None and not state.get("pending"):
        board = build_board_from_state(state)
        state["analysis"] = COORDINATOR.stockfish_service.analyze(board, multipv=3, movetime_ms=120)
        save_game_state(state)
    return JsonResponse({"ok": True, "state": state_to_payload(state)})


@require_POST
def new_game(request: HttpRequest) -> JsonResponse:
    simulations = request.POST.get("simulations", 128)
    try:
        simulations_int = max(int(simulations), 1)
    except ValueError:
        return _json_error("Invalid simulations value.")

    state = start_new_game(request.session, human_color="white", simulations=simulations_int)
    board = build_board_from_state(state)
    state["analysis"] = COORDINATOR.stockfish_service.analyze(board, multipv=3, movetime_ms=120)
    save_game_state(state)
    return JsonResponse({"ok": True, "state": state_to_payload(state)})


@require_POST
def player_move(request: HttpRequest) -> JsonResponse:
    state = get_or_create_game(request.session)
    if state.get("pending"):
        return _json_error("An analysis task is already running.", status=409)
    if state.get("is_game_over"):
        return _json_error("The game is already over.", status=409)

    move_uci = (request.POST.get("move") or "").strip()
    if not move_uci:
        return _json_error("No move provided.")

    simulations = request.POST.get("simulations")
    if simulations is not None:
        try:
            state["simulations"] = max(int(simulations), 1)
        except ValueError:
            return _json_error("Invalid simulations value.")

    board = build_board_from_state(state)
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return _json_error("Move format is invalid.")

    if move not in board.legal_moves:
        return _json_error("Illegal move.", status=422)

    board.push(move)
    state["moves_uci"].append(move_uci)
    state["fen"] = board.fen()
    state["last_player_move_uci"] = move_uci
    state["pending"] = True
    state["version"] = int(state.get("version", 0)) + 1
    state["status_text"] = "Analyzing position and waiting for model response"
    state["analysis"] = None
    state["ai_move"] = None
    save_game_state(state)

    task_id = COORDINATOR.submit_turn_resolution(state)
    state["task_id"] = task_id
    save_game_state(state)
    return JsonResponse({"ok": True, "task_id": task_id, "state": state_to_payload(state)})


@require_GET
def task_status(request: HttpRequest, task_id: str) -> JsonResponse:
    state = get_or_create_game(request.session)
    if state.get("task_id") != task_id:
        return _json_error("Task does not belong to the current game.", status=404)

    task = COORDINATOR.get_task_result(task_id)
    status_value = task["status"]
    if status_value == "pending":
        return JsonResponse({"ok": True, "status": "pending"})
    if status_value == "missing":
        return _json_error("Task not found.", status=404)
    if status_value == "error":
        state["pending"] = False
        state["task_id"] = None
        state["status_text"] = "Analysis failed"
        save_game_state(state)
        return _json_error(task.get("error", "Unknown task error."), status=500)

    result = task["result"]
    if int(result["version"]) == int(state.get("version", -1)):
        state["moves_uci"] = result["moves_uci"]
        state["fen"] = result["fen"]
        state["analysis"] = result["analysis"]
        state["player_classification"] = result["player_classification"]
        state["ai_move"] = result["ai_move"]
        state["is_game_over"] = result["is_game_over"]
        state["result"] = result["result"]
        state["status_text"] = result["status_text"]
        state["pending"] = False
        state["task_id"] = None
        save_game_state(state)
    return JsonResponse({"ok": True, "status": "ready", "state": state_to_payload(state)})
