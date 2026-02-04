# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the
# Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List

import tiktoken  # make sure `pip install tiktoken` is done

logger = getLogger()


class Tokenizer:
    """Tokenizing and encoding/decoding text using tiktoken BPE."""

    def __init__(
        self,
        encoding_name: str,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
    ):
        """
        Initialize the tokenizer with a tiktoken encoding.

        Args:
            encoding_name (str): Name of the tiktoken encoding to load
                                 (e.g. "gpt2", "cl100k_base", "o200k_base").
            bos_token (str): String used as BOS when bos=True in encode().
            eos_token (str): String used as EOS when eos=True in encode().
            pad_token (str): String to use as PAD (if needed elsewhere).
        """
        assert isinstance(encoding_name, str)
        self.encoding_name = encoding_name

        # Load a built-in tiktoken encoding by name.
        self.enc = tiktoken.get_encoding(encoding_name)
        logger.info(f"Loaded tiktoken encoding '{encoding_name}'")

        # Total vocab size (mergeable tokens + special tokens).
        self.n_words: int = self.enc.n_vocab

        # Helper to try to map a string token → a single token id.
        # If the string is not in the vocab as a single token, we return -1
        # and the BOS/EOS/PAD logic will simply skip it.
        def get_single_id(token_str: str) -> int:
            ids = self.enc.encode(token_str, allowed_special="all")
            if len(ids) == 1:
                return ids[0]
            else:
                logger.warning(
                    f"Token '{token_str}' does not map to a single id "
                    f"(got {ids}); it will not be used as a dedicated special id."
                )
                return -1

        # BOS / EOS / PAD token IDs (best-effort guesses based on string forms).
        # You may want to customize these to match your training setup.
        self.bos_id: int = get_single_id(bos_token)
        self.eos_id: int = get_single_id(eos_token)
        self.pad_id: int = get_single_id(pad_token)

        logger.info(
            f"Vocab size: {self.n_words} "
            f"- BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs using tiktoken.

        Args:
            s (str): Input string.
            bos (bool): Whether to prepend the BOS token (if available).
            eos (bool): Whether to append the EOS token (if available).

        Returns:
            List[int]: List of token IDs.
        """
        assert isinstance(s, str)

        # allowed_special="all" so we don't error if s already contains special tokens
        t = self.enc.encode(s, allowed_special="all")

        if bos and self.bos_id >= 0:
            t = [self.bos_id] + t
        if eos and self.eos_id >= 0:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): Token IDs.

        Returns:
            str: Decoded string.
        """
        return self.enc.decode(t)
