import random

from .schemas import IncomingMessage


def classify_bot(conversation: list[IncomingMessage], participant_index: int) -> float:
    return random.uniform(0, 1)
