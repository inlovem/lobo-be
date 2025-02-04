from enum import Enum


class QueryStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"