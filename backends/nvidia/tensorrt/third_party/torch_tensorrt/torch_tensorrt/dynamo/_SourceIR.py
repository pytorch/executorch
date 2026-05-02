from enum import Enum, auto


class SourceIR(Enum):
    NN = auto()
    ACC = auto()
    ATEN = auto()
    PRIM = auto()
    TORCHTRT_LOWERED = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        if self == SourceIR.NN:
            return "nn"
        elif self == SourceIR.ACC:
            return "acc"
        elif self == SourceIR.ATEN:
            return "aten"
        elif self == SourceIR.PRIM:
            return "prim"
        elif self == SourceIR.TORCHTRT_LOWERED:
            return "torchtrt_lowered"
        else:
            return "unknown_ir"
