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

import os
import numpy as np
import torch
import random
from tqdm import tqdm

from fancy_einsum import einsum
from data.othello import OthelloBoardState
from mech_int.tl_othello_utils import (
    load_hooked_model,
    state_stack_to_one_hot_threeway,
    ITOS,
)
from constants import OTHELLO_HOME

random.seed(42)


DATA_DIR = os.path.join(OTHELLO_HOME, "data")

# %%


def seq_to_state_stack(str_moves):
    if isinstance(str_moves, torch.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    valid_moves = []
    all_flipped = []
    for move in str_moves:
        flipped = board.umpire_return_flipped(move)
        states.append(np.copy(board.state))
        valid_moves.append(board.get_valid_moves())
        all_flipped.append(flipped)
    states = np.stack(states, axis=0)
    return states, valid_moves, all_flipped


def build_state_stack(board_seqs_string):
    """
    Construct stack of board-states.
    This function will also filter out corrputed game-sequences.
    """
    state_stack = []
    moves = []
    flipped = []
    for idx, seq in enumerate(board_seqs_string):
        _stack, _moves, _flipped = seq_to_state_stack(seq)
        state_stack.append(_stack)
        moves.append(_moves)
        flipped.append(_flipped)
    return state_stack, moves, flipped


# %%


eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


def get_min_boardstate(board, moves):
    """
    Get minimum board-state that needs to be satisfied to correctly derive moves.
    """
    me = 2
    you = 1

    min_board = torch.zeros((8, 8))
    for move in moves:
        r, c = move // 8, move % 8
        for direction in eights:
            cur_r, cur_c = r, c
            inbetween = False
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                elif board[cur_r, cur_c] == 0:
                    break
                elif board[cur_r, cur_c] == you:
                    inbetween = True
                    continue

                elif board[cur_r, cur_c] == me:
                    if not inbetween:
                        break
                    min_board[cur_r, cur_c] = me
                    while not (cur_r == r and cur_c == c):
                        cur_r = cur_r - direction[0]
                        cur_c = cur_c - direction[1]
                    min_board[cur_r, cur_c] = 0
                    break

    return min_board


# %%

othello_gpt = load_hooked_model("synthetic")
board_seqs_int = torch.load(
    os.path.join(
        DATA_DIR,
        "board_seqs_int_valid.pth",
    )
)

board_seqs_string = torch.load(
    os.path.join(
        DATA_DIR,
        "board_seqs_string_valid.pth",
    )
)

# %%

test_size = 1000
board_seqs_int = board_seqs_int[-test_size:]
board_seqs_string = board_seqs_string[-test_size:]

games_int = board_seqs_int
games_str = board_seqs_string
all_indices = torch.arange(test_size)
print("Building state stacks...")
orig_state_stack, valid_moves, flips = build_state_stack(games_str)

# %%

pos_start = 0
pos_end = othello_gpt.cfg.n_ctx - 0

unembed = othello_gpt.unembed

probes = [
    torch.load(
        os.path.join(
            OTHELLO_HOME,
            f"mech_int/probes/linear/resid_{layer}_linear.pth",
        )
    )
    for layer in range(8)
]
probes = torch.stack(probes)


# %%

earliest_layers_move = []
earliest_layers_board = []
batch_size = 128
for idx in tqdm(range(0, test_size, batch_size)):
    indices = all_indices[idx : idx + batch_size]
    _games_int = games_int[indices]

    state_stack = torch.tensor(np.stack(orig_state_stack))[
        indices, pos_start:pos_end, :, :
    ]
    state_stack_one_hot = state_stack_to_one_hot_threeway(state_stack).cuda()

    logits, cache = othello_gpt.run_with_cache(
        _games_int.cuda()[:, :-1], return_type="logits"
    )
    _valid_moves = [valid_moves[_idx] for _idx in indices]
    _flips = [flips[_idx] for _idx in indices]

    for batch_idx in tqdm(range(indices.shape[0])):
        for move_idx in range(59):
            move_groundtruth = sorted(_valid_moves[batch_idx][move_idx])
            board_groundtruth = state_stack_one_hot.argmax(-1)
            curr_board_gold = board_groundtruth[0, batch_idx, move_idx]

            min_board_state = get_min_boardstate(
                curr_board_gold, move_groundtruth
            )
            debug_board = curr_board_gold.clone()
            for _move in move_groundtruth:
                debug_board[_move // 8, _move % 8] = 99
            earliest_layer_board = 9
            earliest_layer_move = 9
            for layer in range(8):
                resid_post = cache["resid_post", layer][:, pos_start:pos_end]
                probe_out = einsum(
                    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                    resid_post,
                    probes[layer],
                )
                unembedded = einsum(
                    "batch pos d_model, d_model vocab -> batch pos vocab",
                    resid_post,
                    unembed.W_U,
                )

                # Board match.
                board_preds = probe_out.argmax(-1)[0, batch_idx, move_idx]
                if (
                    torch.equal(
                        board_preds,
                        curr_board_gold,
                    )
                    and layer < earliest_layer_board
                ):
                    earliest_layer_board = layer

                # Moves match.
                topk_preds, topk_indices = unembedded[
                    batch_idx, move_idx
                ].topk(k=len(move_groundtruth))
                move_preds = sorted([ITOS[x.item()] for x in topk_indices])
                if (
                    move_groundtruth == move_preds
                ) and layer < earliest_layer_move:
                    earliest_layer_move = layer

                if earliest_layer_move != 9 and earliest_layer_board != 9:
                    break

            earliest_layers_move.append(earliest_layer_move)
            earliest_layers_board.append(earliest_layer_board)


# %%

earliest_moves = torch.tensor(earliest_layers_move).reshape(test_size, -1)
earliest_boards = torch.tensor(earliest_layers_board).reshape(test_size, -1)
assert len(earliest_moves) == len(earliest_boards)

wrong_move = 0
wrong_board = 0
board_first = 0
same = 0
move_first = 0
earliers = []
sames = []
laters = []
wrong_moves = []
wrong_boards = []

for idx in range(earliest_moves.shape[1]):
    move_layers = earliest_moves[:, idx]
    board_layers = earliest_boards[:, idx]
    mask = board_layers.ne(9) * move_layers.ne(9)
    mask_idxs = mask.nonzero().squeeze()
    _move_layers = move_layers[mask_idxs]
    _board_layers = board_layers[mask_idxs]

    assert (_move_layers == 9).sum() == 0
    assert (_board_layers == 9).sum() == 0

    earlier = (
        _board_layers.lt(_move_layers).sum() / board_layers.shape[0]
    ).item()
    same = (
        _board_layers.eq(_move_layers).sum() / board_layers.shape[0]
    ).item()
    later = (
        _board_layers.gt(_move_layers).sum() / board_layers.shape[0]
    ).item()

    wrong_move = ((move_layers == 9).sum() / move_layers.shape[0]).item()
    wrong_board = ((board_layers == 9).sum() / board_layers.shape[0]).item()

    earliers.append(earlier)
    sames.append(same)
    laters.append(later)
    wrong_moves.append(wrong_move)
    wrong_boards.append(wrong_board)

# %%

_mask = earliest_boards.ne(9)
avg_first_layer_board = (earliest_boards * _mask).sum(dim=0) / _mask.sum(dim=0)
_mask = earliest_moves.ne(9)
avg_first_layer_move = (earliest_moves * _mask).sum(dim=0) / _mask.sum(dim=0)


# %%

from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import plotly.express as px

greys = px.colors.sequential.gray

data = {
    "Before": earliers,
    "Same": sames,
    "After": laters,
    "Incorrect (Move)": wrong_moves,
    "Incorrect (Boards)": wrong_boards,
}

INCLUDE_EARLIEST_LAYER_BOARDSTATE = 1
INCLUDE_EARLIEST_LAYER_MOVES = 1

BARCHART_KEYS = [
    "Before",
    "Same",
    "After",
    "Incorrect (Move)",
    "Incorrect (Boards)",
]
COLORS = [
    greys[9],
    greys[5],
    greys[1],
    "blue",
    "#d62728",
]


widths = [1] * len(earliers)
fig = make_subplots(specs=[[{"secondary_y": True}]])
for idx, key in enumerate(BARCHART_KEYS):
    fig.add_trace(
        go.Bar(
            name=key,
            y=data[key],
            x=list(range(len(earliers))),
            width=widths,
            offset=0,
            marker_color=COLORS[idx],
        )
    )

if INCLUDE_EARLIEST_LAYER_BOARDSTATE:
    fig.add_trace(
        dict(
            x=list(range(len(earliers))),
            y=avg_first_layer_board,
            name="Earliest Layer, Board-state",
            type="scatter",
            line=dict(color="greenyellow"),
        ),
        secondary_y=True,
    )

if INCLUDE_EARLIEST_LAYER_MOVES:
    fig.add_trace(
        dict(
            x=list(range(len(earliers))),
            y=avg_first_layer_move,
            name="Earliest Layer, Moves",
            type="scatter",
            line=dict(color="aqua"),
        ),
        secondary_y=True,
    )

fig.update_layout(
    barmode="stack",
)
fig.update_yaxes(range=[0, 7], secondary_y=True)
fig.update_layout(
    yaxis1_tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
    yaxis2_tickvals = [0, 1, 2, 3, 4, 5, 6, 7],
)
fig.show()


# %%
