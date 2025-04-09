# core/attacks/__init__.py
from .fgsm import FGSMAttack
from .pgd import PGDAttack
from .cw import CWAttack

ATTACK_MAP = {
    "fgsm": FGSMAttack,
    "pgd": PGDAttack,
    "cw": CWAttack
}

def get_attack(attack_name: str):
    if attack_name not in ATTACK_MAP:
        raise ValueError(f"Unsupported attack: {attack_name}. Available: {list(ATTACK_MAP.keys())}")
    return ATTACK_MAP[attack_name]