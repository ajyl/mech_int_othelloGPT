"""
Utility functions for mech int experiments.
"""
import os
import numpy as np
import torch
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)
import transformer_lens.utils as utils
from data.othello import OthelloBoardState
from constants import OTHELLO_HOME


STARTING_SQUARES = [27, 28, 35, 36]
ALPHA = "ABCDEFGH"
COLUMNS = [str(_) for _ in range(1, 9)]


def build_itos():

    """
    Build itos mapping.
    Handles 27, 28, 35, 36 squares (starting squares).
    """
    itos = {0: -100}
    for idx in range(1, 28):
        itos[idx] = idx - 1

    for idx in range(28, 34):
        itos[idx] = idx + 1

    for idx in range(34, 61):
        itos[idx] = idx + 3
    return itos


def build_stoi():
    """
    Build stoi mapping.
    Handles 27, 28, 35, 36 squares (starting squares).
    """
    _itos = build_itos()
    stoi = {y: x for x, y in _itos.items()}
    stoi[-1] = 0
    for sq in STARTING_SQUARES:
        assert sq not in stoi
    return stoi


ITOS = build_itos()
stoi = build_stoi()


def to_string(x):
    """
    Confusingly, maps x (board cell)to an int, but a board position
    label not a token label.
    (token labels have 0 == pass, and middle board cells don't exist)
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_string(x.item())

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return [to_string(i) for i in x]

    if isinstance(x, int):
        return itos[x]

    if isinstance(x, str):
        x = x.upper()
        return 8 * ALPHA.index(x[0]) + int(x[1])

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def to_int(x):
    """
    Convert x (board cell) to 'int' representation (model's vocabulary).
    Calls itself recursively.
    """
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_int(x.item())

    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return [to_int(i) for i in x]

    if isinstance(x, int):
        return stoi[x]

    if isinstance(x, str):
        x = x.upper()
        return to_int(to_string(x))

    raise RuntimeError(f"Unknown type for x: {type(x)}.")


def load_hooked_model(model_type):
    """
    Load a HookedTransformer model pulled from Huggingface.
    :model_type: must be either "synthetic" or "champsionship"
    """

    if model_type not in ["synthetic"]:
        raise ValueError(f"Invalid 'model_type': {model_type}.")

    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)

    if model_type == "synthetic":
        model_path = os.path.join(OTHELLO_HOME, "synthetic_model.pth")
    else:
        raise RuntimeError("Unsupported..")

    sd = torch.load(model_path)
    model.load_state_dict(sd)
    return model


def state_stack_to_one_hot_threeway(state_stack):
    one_hot = torch.zeros(
        1,  # even vs. odd vs. all (mode)
        state_stack.shape[0],
        state_stack.shape[1],
        8,  # rows
        8,  # cols
        3,  # the 2 options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[..., 0] = state_stack == 0
    one_hot[0, :, 0::2, ..., 1] = (state_stack == 1)[:, 0::2]
    one_hot[0, :, 1::2, ..., 1] = (state_stack == -1)[:, 1::2]

    one_hot[0, :, 0::2, ..., 2] = (state_stack == -1)[:, 0::2]
    one_hot[0, :, 1::2, ..., 2] = (state_stack == 1)[:, 1::2]

    return one_hot


def seq_to_state_stack(str_moves):
    if isinstance(str_moves, torch.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states


def build_state_stack(board_seqs_string):
    """
    Construct stack of board-states.
    """
    state_stack = []
    for idx, seq in enumerate(board_seqs_string):
        _stack = seq_to_state_stack(seq)
        state_stack.append(_stack)
    return torch.tensor(np.stack(state_stack))


def to_board_label(idx):
    return f"{ALPHA[idx//8]}{COLUMNS[idx%8]}"


def run_with_cache_and_hooks(
    model,
    fwd_hooks,
    *model_args,
    **model_kwargs,
):
    """
    Runs the model and returns model output and a Cache object.
    Applies all hooks in fwd_hooks.
    """
    cache_dict = model.add_caching_hooks()
    for name, hook in fwd_hooks:
        if type(name) == str:
            model.mod_dict[name].add_hook(hook, dir="fwd")
        else:
            # Otherwise, name is a Boolean function on names
            for hook_name, hp in model.hook_dict.items():
                if name(hook_name):
                    hp.add_hook(hook, dir="fwd")

    model_out = model(*model_args, **model_kwargs)

    model.reset_hooks(False, including_permanent=False)
    return model_out, cache_dict
