# kg_ai_papers/models/section.py
from dataclasses import dataclass

@dataclass
class Section:
    title: str
    level: str  # "1", "2", ...
    text: str
