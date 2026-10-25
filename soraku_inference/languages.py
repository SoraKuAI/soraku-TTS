"""Persisted language tokens shared by training and native inference."""

LANGUAGES = ("zh", "en", "ja", "auto")
LANGUAGE_TOKENS = tuple(f"<|lang_{language}|>" for language in LANGUAGES)


def configure_language_tokens(tokenizer, expected=None):
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(LANGUAGE_TOKENS)},
        replace_additional_special_tokens=False,
    )
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in LANGUAGE_TOKENS)
    if len(set(ids)) != 4 or any(
        tokenizer(token, add_special_tokens=False)["input_ids"] != [identifier]
        for token, identifier in zip(LANGUAGE_TOKENS, ids)
    ):
        raise ValueError("language markers must be four distinct atomic special tokens")
    if expected is not None and tuple(expected) != ids:
        raise ValueError(f"language token IDs changed: expected {expected}, got {ids}")
    return ids


def load_checkpoint_tokenizer(text_model, config, checkpoint):
    from pathlib import Path
    from transformers import AutoTokenizer

    saved = Path(checkpoint).parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(saved) if config.language_token_ids and saved.is_dir() else text_model,
        local_files_only=True,
    )
    if config.language_token_ids:
        configure_language_tokens(tokenizer, config.language_token_ids)
    return tokenizer


def prefix_language(ids, language, config):
    if not config.language_token_ids:
        return list(ids)
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language!r}")
    return [config.language_token_ids[LANGUAGES.index(language)], *ids]
