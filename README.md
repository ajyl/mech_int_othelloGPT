# Mechanistic Interpretations of OthelloGPT

This repository provides the experiments in Emergent Linear Representations in World Models of Self-Supervised Sequence Models.


## Abstract

> How do sequence models represent their decision-making process? Prior work suggests that Othello-playing neural network learned nonlinear models of the board state (Li et al., 2023a). In this work, we provide evidence of a closely related linear representation of the board. In particular, we show that probing for “my colour” vs. “opponent’s colour” may be a simple yet powerful way to interpret the model’s internal state. This precise understanding of the internal representations allows us to control the model’s behaviour with simple vector arithmetic. Linear representations enable significant interpretability progress, which we demonstrate with further exploration of how the world model is computed.

## Installation

```
conda env create -f environment.yml
conda activate mech_int_othello
```

Download relevent data:

* [OthelloGPT](https://drive.google.com/file/d/1Soc-OCGfuIk6P617hS6qqrAy_bI7KQHs/view?usp=drive_link): We only analyze the synthetic model. Save this checkpoint to the root directory of this repo.
* [sequence data](https://drive.google.com/drive/folders/1SdBk94m3QJW0w4sfx6MQlh7BtyQlYeRl?usp=sharing): Game sequence data used for training probes and the rest of our experiments. Keep this data in `./data`.


## Relevant Files

This repo uses bits and pieces from [Othello World](https://github.com/likenneth/othello_world).
All of the experiments in our paper can be found in the `mech_int` directory.

* `board_probe.py` contains training + evaluation scripts for our linear probes. See `train()` and `evaluate()`.
* `train_flipped.py` contains training + evaluation scripts for our `Flipped` probes. See `train()` and `evaluate()`.
* `intervene.py`, `intervene_blank.py`, `intervene_flipped.py` contain our intervention experiments. 
* `tl_othello_utils.py` contains various utility functions.
* `./figures/` contains various notebooks that were used to create our figures.
* `./probes/` contains all of our probes.


