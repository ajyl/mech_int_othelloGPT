# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

"""
Empty logit attribution experiment.
"""
import os
import random
from collections import defaultdict
import copy

import numpy as np
import torch
import plotly.express as px
from fancy_einsum import einsum
from jaxtyping import Float

from transformer_lens import (
    ActivationCache,
)
import transformer_lens.utils as utils
from data.othello import OthelloBoardState
from mech_int.tl_othello_utils import (
    load_hooked_model,
    to_string,
    to_int,
)
from constants import OTHELLO_HOME


random.seed(42)


# %%


def build_corrupt_games(board_seqs_string, exclude_move, num_moves):
    """
    Construct stack of board-states.
    This function will also filter out corrputed game-sequences.
    """
    clean_states = []
    corrupt_states = []
    clean_moves = []
    corrupt_moves = []

    for idx, seq in enumerate(board_seqs_string):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()

        board = OthelloBoardState()
        _clean_state = []
        _corrupt_state = []
        for move in seq[:-1]:
            board.umpire(move)
            _clean_state.append(np.copy(board.state))
            _corrupt_state.append(np.copy(board.state))

        clean_board = copy.deepcopy(board)
        corrupt_board = board

        # Find alternative move to A4 (exclude_move)
        valid_moves = [
            _move
            for _move in corrupt_board.get_valid_moves()
            if _move != to_string(exclude_move)
        ]
        if len(valid_moves) < 1:
            continue
        first_valid_move = random.sample(valid_moves, k=1)[0]

        clean_board.umpire(seq[-1])
        corrupt_board.umpire(first_valid_move)

        _clean_state.append(np.copy(clean_board.state))
        _corrupt_state.append(np.copy(corrupt_board.state))

        _additional_clean_states = []
        _additional_corrupt_states = []
        _common_valid_moves = []
        for _ in range(num_moves):
            clean_valid_moves = clean_board.get_valid_moves()
            corrupt_valid_moves = corrupt_board.get_valid_moves()
            valid_moves = [
                _move
                for _move in clean_valid_moves
                if _move in corrupt_valid_moves
            ]
            if len(valid_moves) < 1:
                break

            valid_move = random.sample(valid_moves, k=1)[0]
            clean_board.umpire(valid_move)
            corrupt_board.umpire(valid_move)
            _additional_clean_states.append(np.copy(clean_board.state))
            _additional_corrupt_states.append(np.copy(corrupt_board.state))
            _common_valid_moves.append(valid_move)

        assert len(_additional_clean_states) == len(_additional_corrupt_states)
        assert len(_additional_clean_states) == len(_common_valid_moves)

        if len(_common_valid_moves) != num_moves:
            # Failed to find :num_moves: number of common valid moves
            continue

        _clean_state.extend(_additional_clean_states)
        _corrupt_state.extend(_additional_corrupt_states)

        clean_states.append(np.stack(_clean_state, axis=0))
        corrupt_states.append(np.stack(_corrupt_state, axis=0))

        clean_moves.append(seq + _common_valid_moves)
        corrupt_moves.append(
            seq[:-1] + [first_valid_move] + _common_valid_moves
        )

    if len(clean_states) < 1:
        return [None] * 4
    return (
        torch.tensor(np.stack(clean_states)),
        torch.tensor(np.stack(corrupt_states)),
        torch.tensor(clean_moves),
        torch.tensor(corrupt_moves),
    )


# %%


def decomps_to_prob_diff(
    clean_components: Float[torch.Tensor, "n_components batch seq d_model"],
    corrupt_components: Float[torch.Tensor, "n_components batch seq d_model"],
    linear_probe: Float[torch.Tensor, "d_model options"],
    move_idx,
) -> float:
    """
    "Logit diff" when projecting decompositions into probe direction.
    """
    clean_components = clean_components[..., move_idx, :]
    corrupt_components = corrupt_components[..., move_idx, :]
    clean_logits = einsum(
        "n_components batch d_model, d_model options -> n_components batch options",
        clean_components,
        linear_probe,
    )
    corrupt_logits = einsum(
        "n_components batch d_model, d_model options -> n_components batch options",
        corrupt_components,
        linear_probe,
    )

    clean_blank_probs = clean_logits.softmax(-1)[..., 0]
    corrupt_blank_probs = corrupt_logits.softmax(-1)[..., 0]

    return corrupt_blank_probs - clean_blank_probs


# %%

othello_gpt = load_hooked_model("synthetic")
board_seqs_int = torch.load(
    os.path.join(OTHELLO_HOME, "data/board_seqs_int_valid.pth")
)
board_seqs_string = torch.load(
    os.path.join(OTHELLO_HOME, "data/board_seqs_string_valid.pth")
)
linear_probe = torch.load("probes/linear/resid_0_linear.pth")

# %%

cell = "A4"

cell_str = to_string(cell)
row_idx, col_idx = cell_str // 8, cell_str % 8
probe = linear_probe[0, ..., row_idx, col_idx, :]
val_size = 512

indices = torch.arange(val_size)
all_games_int = board_seqs_int[indices]
all_games_str = board_seqs_string[indices]


cell_str = to_string(cell)
cell_idxs = (all_games_str == cell_str).nonzero()

idx_group = defaultdict(list)
for idx_pair in cell_idxs:
    idx_group[idx_pair[1].item()].append(idx_pair[0].item())


all_decomp_diffs = {}

for num_pad_moves in range(0, 10):
    print(f"Num pad moves: {num_pad_moves}")
    decomp_diffs = []
    corrupted_games = []
    num_games = 0
    for move_idx in sorted(idx_group.keys()):
        game_idxs = idx_group[move_idx]
        clean_games_str = all_games_str[game_idxs, : move_idx + 1]
        (
            clean_states,
            corrupt_states,
            clean_games_str,
            corrupt_games_str,
        ) = build_corrupt_games(clean_games_str, cell, num_pad_moves)
        if clean_states is None:
            continue
        num_games += len(clean_games_str)

        corrupted_games.append((clean_games_str, corrupt_games_str))

        clean_games_int = torch.tensor(to_int(clean_games_str))
        corrupt_games_int = torch.tensor(to_int(corrupt_games_str))

        _, clean_cache = othello_gpt.run_with_cache(
            clean_games_int.cuda(), return_type=None
        )
        _, corrupt_cache = othello_gpt.run_with_cache(
            corrupt_games_int.cuda(), return_type=None
        )

        # [n_components, batch, seq_len, d_model]
        clean_decomps, labels = clean_cache.get_full_resid_decomposition(
            layer=1, expand_neurons=False, return_labels=True
        )
        clean_decomps = clean_decomps.cpu()
        corrupt_decomps, labels = corrupt_cache.get_full_resid_decomposition(
            layer=1, expand_neurons=False, return_labels=True
        )
        corrupt_decomps = corrupt_decomps.cpu()

        decomps_diff = decomps_to_prob_diff(
            clean_decomps, corrupt_decomps, probe.cpu(), -1
        )
        decomp_diffs.append(decomps_diff.cpu())

    print(num_games)
    diffs = torch.cat(decomp_diffs, dim=1)

    all_decomp_diffs[num_pad_moves] = [diffs.detach().cpu(), num_games]

# %%

odds = []
evens = []
all_games = 0
for num_pad_moves, diff_objs in all_decomp_diffs.items():
    all_games += diff_objs[1]
    if num_pad_moves % 2 == 0:
        evens.append(diff_objs[0])
    else:
        odds.append(diff_objs[0])


evens = torch.cat(evens, dim=1).mean(dim=1)[:8]
odds = torch.cat(odds, dim=1).mean(dim=1)[:8]


fig = px.imshow(
    utils.to_numpy(torch.stack([evens, odds])),
    x=[
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ],
    y=["A4 Mine", "A4 Yours"],
    labels=dict(
        x="Attention Head",
    ),
    zmin=-0,
    zmax=0.8,
    color_continuous_scale="greens",
    aspect="auto",
)
fig.layout.yaxis.type = "category"
fig.layout.xaxis.type = "category"
fig.update_layout(font=dict(size=18))
fig.update_layout(width=500, height=270)
fig.show()
