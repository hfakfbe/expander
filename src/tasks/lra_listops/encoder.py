from __future__ import annotations


TOKEN_TO_ID = {
    "<pad>": 0,
    "(": 1,
    ")": 2,
    "[MAX": 3,
    "[MIN": 4,
    "[MED": 5,
    "[SM": 6,
    "]": 7,
    "0": 8,
    "1": 9,
    "2": 10,
    "3": 11,
    "4": 12,
    "5": 13,
    "6": 14,
    "7": 15,
    "8": 16,
    "9": 17,
}


def encode_input(value) -> list[int]:
    return [TOKEN_TO_ID[str(item)] for item in value]

