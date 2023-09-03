""""
Experiments for interventions.
"""

import os

import random
from functools import partial
import pickle
import numpy as np
import torch
from tqdm import tqdm

from data import get_othello
from data.othello import OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mech_int.tl_othello_utils import (
    load_hooked_model,
    to_board_label,
    ITOS,
    run_with_cache_and_hooks,
)
from constants import OTHELLO_HOME
from mingpt.utils import set_seed

set_seed(44)
random.seed(42)


with open(
    os.path.join(OTHELLO_HOME, "data/intervention_benchmark.pkl"), "rb"
) as input_file:
    dataset = pickle.load(input_file)

intv_data = [
    {
        "intervention_position": permit_reverse(_sample["pos_int"]),
        "intervention_from": _sample["ori_color"],
        "intervention_to": 2 - _sample["ori_color"],
        "completion": _sample["history"],
    }
    for _sample in dataset
]

_othello = get_othello(ood_perc=0.0, data_root=None, wthor=False, ood_num=1)
char_dataset = CharDataset(_othello)


probes = {}
probe_dir = os.path.join(OTHELLO_HOME, "mech_int/probes/flipped")
for layer in range(8):
    probes[layer] = torch.load(
        os.path.join(probe_dir, f"resid_{layer}_flipped.pth")
    )
    probes[layer].requires_grad = False


othello_gpt = load_hooked_model("synthetic")


def patch_heads(
    modified_heads,
    hook,
    probe_layer,
    from_player,
    to_player,
    row,
    col,
):
    add_vector = probes[probe_layer][..., row, col, 0]
    modified_heads[:, -1, :] = modified_heads[:, -1, :] + (
        -1 * (add_vector / add_vector.norm())
    )
    return modified_heads


false_positives = []
false_negatives = []
false_positives_null_intv = []
false_negatives_null_intv = []

for _sample in tqdm(intv_data):
    board_state = OthelloBoardState()
    completion = _sample["completion"]
    board_state.update(completion[:-1])
    flipped = board_state.umpire_return_flipped(completion[-1])

    partial_game = torch.tensor(
        [char_dataset.stoi[s] for s in completion], dtype=torch.long
    ).to("cuda")

    pre_intv_valids = [
        permit_reverse(_) for _ in board_state.get_valid_moves()
    ]
    pre_intv_pred = othello_gpt(partial_game[None, :])

    pre_intv_pred = pre_intv_pred[0, -1, 1:]
    padding = torch.zeros(2).cuda()

    pre_intv_pred = torch.softmax(pre_intv_pred, dim=0)
    pre_intv_pred = torch.cat(
        [
            pre_intv_pred[:27],
            padding,
            pre_intv_pred[27:33],
            padding,
            pre_intv_pred[33:],
        ],
        dim=0,
    )

    move = random.choice(flipped)
    r, c = move
    board_state.state[r, c] = _sample["intervention_to"] - 1

    post_intv_valids = [
        permit_reverse(_) for _ in board_state.get_valid_moves()
    ]

    hook_fns = []
    for _layer in range(8):
        hook_fns.append(
            partial(
                patch_heads,
                probe_layer=_layer,
                from_player=_sample["intervention_from"],
                to_player=_sample["intervention_to"],
                row=r,
                col=c,
            )
        )

    orig_logits, blank_cache = othello_gpt.run_with_cache(
        partial_game, return_type="logits"
    )
    orig_topk_preds = orig_logits[0, -1].topk(k=60)
    (patched_logits, modified_cache,) = run_with_cache_and_hooks(
        othello_gpt,
        fwd_hooks=[
            ("blocks.0.hook_attn_out", hook_fns[0]),
            ("blocks.1.hook_attn_out", hook_fns[1]),
            ("blocks.2.hook_attn_out", hook_fns[2]),
            ("blocks.3.hook_attn_out", hook_fns[3]),
            ("blocks.4.hook_attn_out", hook_fns[4]),
            ("blocks.5.hook_attn_out", hook_fns[5]),
            ("blocks.6.hook_attn_out", hook_fns[6]),
            ("blocks.7.hook_attn_out", hook_fns[7]),
        ],
        partial_game,
    )

    modified_topk_preds = patched_logits[0, -1].topk(k=60)

    orig_preds = [
        to_board_label(ITOS[x.item()]).lower()
        for x in orig_topk_preds.indices
        if x != 0
    ]
    modified_preds = [
        to_board_label(ITOS[x.item()]).lower()
        for x in modified_topk_preds.indices
        if x != 0
    ]

    modified_preds_cutoff = modified_preds[: len(post_intv_valids)]
    orig_preds_cutoff = orig_preds[: len(post_intv_valids)]
    _new_valid_gold = sorted(
        [x for x in post_intv_valids if x not in pre_intv_valids]
    )
    _new_valid_pred = sorted(
        [
            x
            for x in modified_preds_cutoff
            if x in post_intv_valids and x not in pre_intv_valids
        ]
    )

    _false_pos = [
        x for x in modified_preds_cutoff if x not in post_intv_valids
    ]
    _false_negs = [
        x for x in post_intv_valids if x not in modified_preds_cutoff
    ]
    false_positives.append(len(_false_pos))
    false_negatives.append(len(_false_negs))

    _false_pos_orig = [
        x for x in orig_preds_cutoff if x not in post_intv_valids
    ]
    _false_negs_orig = [
        x for x in post_intv_valids if x not in orig_preds_cutoff
    ]
    false_positives_null_intv.append(len(_false_pos_orig))
    false_negatives_null_intv.append(len(_false_negs_orig))

errs = [
    false_positives[idx] + false_negatives[idx]
    for idx in range(len(false_positives))
]
print("errs:", np.mean(errs))
null_intv_errs = [
    false_positives_null_intv[idx] + false_negatives_null_intv[idx]
    for idx in range(len(false_positives_null_intv))
]
print("null intv errs:", np.mean(null_intv_errs))
