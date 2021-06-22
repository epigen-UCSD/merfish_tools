"""Global configuration settings for the pipeline."""

import json

config = {}


def get(key: str):
    return config[key]


def has(key: str) -> bool:
    return key in config


def load_from_file(config_file: str) -> None:
    global config
    with open(config_file) as conf:
        config = json.loads(conf.read())


def load(args):
    global config
    if args.config_file:
        load_from_file(args.config_file)
    for key, value in vars(args).items():
        if value is None:
            continue
        config[key] = value
