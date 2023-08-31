"""
Script for training/evaluating probes.
"""

import json
import random
import os
import numpy as np
import torch

import einops
from fancy_einsum import einsum
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm

from mech_int.tl_othello_utils import (
    load_hooked_model,
    state_stack_to_one_hot_threeway,
    build_state_stack,
)
from constants import OTHELLO_HOME

random.seed(42)


# TODO
DATA_DIR = os.path.join(OTHELLO_HOME, "data")


def train(config):
    """Train probe model."""
    print("Training config:")
    print(json.dumps(config, indent=4))
    othello_gpt = load_hooked_model("synthetic")

    lr = config["lr"]
    wd = config["wd"]
    rows = config["rows"]
    cols = config["cols"]
    valid_every = config["valid_every"]
    batch_size = config["batch_size"]
    pos_start = config["pos_start"]
    pos_end = othello_gpt.cfg.n_ctx - config["pos_end"]
    num_epochs = config["num_epochs"]
    valid_size = config["valid_size"]
    valid_patience = config["valid_patience"]
    output_dir = config["output_dir"]
    assert os.path.isdir(output_dir)

    board_seqs_int = torch.load(
        os.path.join(
            DATA_DIR,
            "board_seqs_int_train.pth",
        )
    )
    board_seqs_string = torch.load(
        os.path.join(
            DATA_DIR,
            "board_seqs_string_train.pth",
        )
    )

    board_seqs_int_valid = torch.load(
        os.path.join(
            DATA_DIR,
            "board_seqs_int_valid.pth",
        )
    )
    board_seqs_string_valid = torch.load(
        os.path.join(
            DATA_DIR,
            "board_seqs_string_valid.pth",
        )
    )

    valid_indices = torch.arange(valid_size)
    valid_games_int = board_seqs_int_valid[valid_indices]
    valid_games_str = board_seqs_string_valid[valid_indices]
    valid_state_stack = build_state_stack(valid_games_str)
    train_size = board_seqs_int.shape[0]

    modes = 1
    options = 3

    for layer in tqdm(range(8)):
        print(f"Training layer {layer}!")
        done_training = False
        probe_name = f"resid_{layer}_linear"
        lowest_val_loss = 999999

        probe_model = (
            torch.randn(
                modes,
                othello_gpt.cfg.d_model,
                rows,
                cols,
                options,
                requires_grad=False,
                device="cuda",
            )
            / np.sqrt(othello_gpt.cfg.d_model)
        )
        probe_model.requires_grad = True
        optimiser = torch.optim.AdamW(
            [probe_model], lr=lr, betas=(0.9, 0.99), weight_decay=wd
        )

        torch.manual_seed(42)

        train_seen = 0
        for epoch in range(num_epochs):
            if done_training:
                print(f"Training seen: {train_seen}")
                break

            full_train_indices = torch.randperm(train_size)
            for idx in tqdm(range(0, train_size, batch_size)):
                if done_training:
                    print(f"Training seen: {train_seen}")
                    break
                train_seen += batch_size
                indices = full_train_indices[idx : idx + batch_size]
                games_int = board_seqs_int[indices]
                games_str = board_seqs_string[indices]
                state_stack = build_state_stack(games_str)

                state_stack = state_stack[:, pos_start:pos_end, :, :]

                state_stack_one_hot = state_stack_to_one_hot_threeway(
                    state_stack
                ).cuda()
                with torch.inference_mode():
                    _, cache = othello_gpt.run_with_cache(
                        games_int.cuda()[:, :-1], return_type=None
                    )

                    resid_post = cache["resid_post", layer][
                        :, pos_start:pos_end
                    ]

                probe_out = einsum(
                    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                    resid_post.clone(),
                    probe_model,
                )

                # [modes, batch, pos, 8, 8, options]
                probe_log_probs = probe_out.log_softmax(-1)

                # [mode, pos, 8, 8]
                probe_correct_log_probs = (
                    einops.reduce(
                        probe_log_probs * state_stack_one_hot,
                        "modes batch pos rows cols options -> modes pos rows cols",
                        "mean",
                    )
                    * options
                )

                train_loss = -probe_correct_log_probs[0, :].mean(0).sum()
                train_loss.backward()

                optimiser.step()
                optimiser.zero_grad()

                if idx % valid_every == 0:
                    val_losses = []
                    val_accuracies = []
                    for val_batch_idx in range(0, valid_size, batch_size):
                        _valid_indices = valid_indices[
                            val_batch_idx : val_batch_idx + batch_size
                        ]
                        _valid_games_int = valid_games_int[_valid_indices]
                        _valid_state_stack = valid_state_stack[_valid_indices]
                        _valid_state_stack = _valid_state_stack[
                            :, pos_start:pos_end, ...
                        ]
                        _valid_stack_one_hot = state_stack_to_one_hot_threeway(
                            _valid_state_stack
                        ).cuda()

                        _val_logits, _val_cache = othello_gpt.run_with_cache(
                            _valid_games_int.cuda()[:, :-1],
                            return_type="logits",
                        )
                        val_resid_post = _val_cache["resid_post", layer][
                            :, pos_start:pos_end
                        ]
                        _val_probe_out = einsum(
                            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                            val_resid_post.clone(),
                            probe_model,
                        )

                        _val_probe_log_probs = _val_probe_out.log_softmax(-1)
                        val_probe_correct_log_probs = (
                            einops.reduce(
                                _val_probe_log_probs * _valid_stack_one_hot,
                                "modes batch pos rows cols options -> modes pos rows cols",
                                "mean",
                            )
                            * options
                        )
                        val_loss = (
                            -val_probe_correct_log_probs[0, :].mean(0).sum()
                        ).item()
                        val_losses.append(val_loss * _valid_indices.shape[0])

                        val_preds = _val_probe_out.argmax(-1)
                        val_gold = _valid_stack_one_hot.argmax(-1)

                        val_results = val_preds == val_gold
                        val_accuracy = (
                            val_results.sum() / val_results.numel()
                        ).item()
                        val_accuracies.append(
                            val_accuracy * _valid_indices.shape[0]
                        )

                    validation_loss = sum(val_losses) / valid_size
                    validation_accuracy = sum(val_accuracies) / valid_size
                    print(f"  Validation Accuracy: {validation_accuracy}")
                    print(f"  Validation Loss: {validation_loss}")
                    if validation_loss < lowest_val_loss:
                        print(f"  New lowest valid loss! {validation_loss}")
                        curr_patience = 0
                        torch.save(
                            probe_model, f"{output_dir}/{probe_name}.pth"
                        )

                        lowest_val_loss = validation_loss

                    else:
                        curr_patience += 1
                        print(
                            f"  Did not beat previous best ({lowest_val_loss})"
                        )
                        print(f"  Current patience: {curr_patience}")
                        if curr_patience >= valid_patience:
                            print("  Ran out of patience! Stopping training.")
                            done_training = True


def evaluate(probe_dir):
    """
    Evaluate probe model.
    """
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

    test_size = 1000
    board_seqs_int = board_seqs_int[-test_size:]
    board_seqs_string = board_seqs_string[-test_size:]

    games_int = board_seqs_int
    games_str = board_seqs_string
    all_indices = torch.arange(test_size)
    batch_size = 128
    orig_state_stack = build_state_stack(games_str)

    pos_start = 0
    pos_end = othello_gpt.cfg.n_ctx - 0

    for layer in range(8):
        linear_probe = torch.load(os.path.join(probe_dir, f"resid_{layer}_linear.pth"))
        for test_layer in range(layer + 1):
            if layer != test_layer:
                continue
            # print(f"Layer {test_layer}.")
            accs = []
            per_timestep_num_correct = torch.zeros((59, 8, 8))
            all_preds = []
            all_groundtruths = []
            for idx in range(0, test_size, batch_size):
                indices = all_indices[idx : idx + batch_size]
                _games_int = games_int[indices]

                # state_stack = orig_state_stack[:, pos_start:pos_end, :, :]
                state_stack = orig_state_stack[
                    indices, pos_start:pos_end, :, :
                ]
                state_stack_one_hot = state_stack_to_one_hot_threeway(
                    state_stack
                ).cuda()

                logits, cache = othello_gpt.run_with_cache(
                    _games_int.cuda()[:, :-1], return_type="logits"
                )
                resid_post = cache["resid_post", test_layer][
                    :, pos_start:pos_end
                ]
                probe_out = einsum(
                    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
                    resid_post.clone(),
                    linear_probe,
                )

                # [256, 51, 8, 8]
                preds = probe_out.argmax(-1)
                groundtruth = state_stack_one_hot.argmax(-1)
                test_results = preds == groundtruth
                test_acc = (test_results.sum() / test_results.numel()).item()
                per_timestep_num_correct += test_results[0].sum(0).cpu()
                all_preds.append(preds)
                all_groundtruths.append(groundtruth)
                accs.append(test_acc * indices.shape[0])

            if test_layer == layer:
                _all_preds = torch.cat(all_preds, dim=1)
                _all_gt = torch.cat(all_groundtruths, dim=1)
                f1_score = multiclass_f1_score(
                    _all_preds.view(-1), _all_gt.view(-1), num_classes=3
                )
                print(f"F1_score: {f1_score}")


if __name__ == "__main__":
    train_config = {
        "model": "othello_gpt",
        "lr": 1e-2,
        "wd": 0.01,
        "rows": 8,
        "cols": 8,
        "valid_every": 200,
        "batch_size": 128,
        "pos_start": 0,
        "pos_end": 0,
        "num_epochs": 1,
        "valid_size": 512,
        "valid_patience": 10,
        "output_dir": "probes/linear",
    }
    assert train_config["model"] in ["othello_gpt"]

    train(train_config)
    evaluate(train_config["output_dir"])
