from __future__ import annotations

from django.urls import path

from . import views

app_name = "gui"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/state/", views.game_state, name="game_state"),
    path("api/new-game/", views.new_game, name="new_game"),
    path("api/player-move/", views.player_move, name="player_move"),
    path("api/task/<str:task_id>/", views.task_status, name="task_status"),
]
