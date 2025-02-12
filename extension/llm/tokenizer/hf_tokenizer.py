import json
import os
import re
from typing import Dict, List, Optional


class HFTokenizer:
    def __init__(self):
        self.special_token_encoder: Dict[str, int] = {}
        self.special_token_decoder: Dict[int, str] = {}
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.n_words: int = 0
        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.initialized: bool = False
        self.pre_tokenizer_config = None

    def load(self, path: str) -> bool:
        if os.path.isdir(path):
            model_json = os.path.join(path, "tokenizer.json")
            model_config_json = os.path.join(path, "tokenizer_config.json")
        else:
            model_json = path
            model_config_json = ""

        if not os.path.exists(model_json):
            print(f"no tokenizer.json found in {path}")
            return False

        try:
            with open(model_json, "r") as file:
                parsed_json = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing json file: {e}")
            return False

        # Parse special tokens
        try:
            special_tokens = parsed_json["added_tokens"]
            for token_info in special_tokens:
                token = token_info["content"]
                token_id = token_info["id"]
                if token in self.special_token_encoder:
                    print(f"duplicate special token: {token}")
                    return False
                if token_id in self.special_token_decoder:
                    print(f"duplicate special token id: {token_id}")
                    return False
                self.special_token_encoder[token] = token_id
                self.special_token_decoder[token_id] = token
        except KeyError as e:
            print(f"Could not parse special tokens: {e}")
            return False

        # Parse standard tokens
        try:
            vocab = parsed_json["model"]["vocab"]
            for token, token_id in vocab.items():
                if token_id not in self.special_token_decoder:
                    if token in self.encoder:
                        print(f"duplicate token: {token}")
                        return False
                    if token_id in self.decoder:
                        print(f"duplicate token id: {token_id}")
                        return False
                    self.encoder[token] = token_id
                    self.decoder[token_id] = token
        except KeyError as e:
            print(f"Could not parse tokens: {e}")
            return False

        self.n_words = len(self.encoder) + len(self.special_token_encoder)

        # Parse tokenizer config if available
        if model_config_json and os.path.exists(model_config_json):
            try:
                with open(model_config_json, "r") as file:
                    config_json = json.load(file)
                bos_token = config_json["bos_token"]
                eos_token = config_json["eos_token"]
                if bos_token not in self.special_token_encoder:
                    print(f"BOS token {bos_token} not in special tokens")
                    return False
                if eos_token not in self.special_token_encoder:
                    print(f"EOS token {eos_token} not in special tokens")
                    return False
                self.bos_id = self.special_token_encoder[bos_token]
                self.eos_id = self.special_token_encoder[eos_token]
            except KeyError as e:
                print(f"Could not parse eos/bos from tokenizer config: {e}")
                return False
        else:
            # Guess BOS and EOS tokens
            bos_candidates = []
            eos_candidates = []
            for token in self.special_token_encoder:
                if "bos" in token or "begin" in token:
                    bos_candidates.append(token)
                if "eos" in token or "end" in token:
                    eos_candidates.append(token)
            if len(bos_candidates) == 1:
                self.bos_id = self.special_token_encoder[bos_candidates[0]]
            if len(eos_candidates) == 1:
                self.eos_id = self.special_token_encoder[eos_candidates[0]]
            if self.bos_id is not None and self.eos_id is None:
                self.eos_id = self.bos_id
            elif self.eos_id is not None and self.bos_id is None:
                self.bos_id = self.eos_id

        # Parse pre-tokenizer configuration
        try:
            self.pre_tokenizer_config = parsed_json.get("pre_tokenizer", {})
        except KeyError as e:
            print(f"Could not parse pre_tokenizer: {e}")
            return False

        self.initialized = True
        return True

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        breakpoint()
        if not self.initialized:
            raise ValueError("Tokenizer not initialized")
        tokens = []
        for piece in self._pretokenize(text):
            if piece in self.encoder:
                tokens.append(self.encoder[piece])
            else:
                # Handle unknown tokens (e.g., byte pair encoding)
                pass
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        if not self.initialized:
            raise ValueError("Tokenizer not initialized")
        text = ""
        for token in tokens:
            if token in self.decoder:
                text += self.decoder[token]
            elif token in self.special_token_decoder:
                text += self.special_token_decoder[token]
            else:
                # Handle unknown tokens
                pass
        return text

    def _pretokenize(self, text: str) -> List[str]:
        if not self.pre_tokenizer_config:
            return [text]  # Default to no pre-tokenization

        breakpoint()
        pre_tokenizer_type = self.pre_tokenizer_config.get("type", "")
        if pre_tokenizer_type == "Split":
            return self._split_pretokenize(text)
        elif pre_tokenizer_type == "Digits":
            return self._digits_pretokenize(text)
        elif pre_tokenizer_type == "ByteLevel":
            return self._byte_level_pretokenize(text)
        elif pre_tokenizer_type == "Sequence":
            return self._sequence_pretokenize(text)
        else:
            return [text]  # Unsupported pre-tokenizer type

    def _split_pretokenize(self, text: str) -> List[str]:
        pattern = self.pre_tokenizer_config.get("pattern", "")
        if not pattern:
            return [text]
        return re.split(f"({pattern})", text)

    def _digits_pretokenize(self, text: str) -> List[str]:
        individual_digits = self.pre_tokenizer_config.get("individual_digits", False)
        if individual_digits:
            return list(text)  # Split into individual characters
        else:
            return re.split(r"(\d+)", text)  # Split on digits

    def _byte_level_pretokenize(self, text: str) -> List[str]:
        add_prefix_space = self.pre_tokenizer_config.get("add_prefix_space", False)
        pattern = self.pre_tokenizer_config.get("pattern", "")
        if add_prefix_space and not text.startswith(" "):
            text = " " + text
        if not pattern:
            pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        return re.findall(pattern, text)

    def _sequence_pretokenize(self, text: str) -> List[str]:
        pretokenizers = self.pre_tokenizer_config.get("pretokenizers", [])
        pieces = [text]
        for pretokenizer_config in pretokenizers:
            new_pieces = []
            for piece in pieces:
                new_pieces.extend(self._pretokenize(piece))
            pieces = new_pieces
        return pieces
