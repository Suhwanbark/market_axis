#!/usr/bin/env python3

import torch

from news_large_embedding_control import last_token_pool


def test_last_token_pool_left_padding() -> None:
    states = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    pooled = last_token_pool(states, mask)
    assert torch.equal(pooled, states[:, -1])


def test_last_token_pool_right_padding() -> None:
    states = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    pooled = last_token_pool(states, mask)
    assert torch.equal(pooled[0], states[0, 1])
    assert torch.equal(pooled[1], states[1, 2])


if __name__ == "__main__":
    test_last_token_pool_left_padding()
    test_last_token_pool_right_padding()
    print("LARGE EMBEDDING CONTROL TESTS PASS")
