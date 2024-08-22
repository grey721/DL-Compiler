from io import StringIO
from enum import Enum
import sys
import traceback


class Capturing(list):

    # def __enter__(self):
    #     pass
    
    # def __exit__(self, *args):
    #     pass

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class TransformRule(Enum):
    NOPE = 1
    GEN_NPU_OP_TIME_STEP = 2
    GEN_NPU_OP_GROUP = 3
    GEN_NPU_TENSOR_LIFE_CYCLE = 4
    NPU_MEMORY_SCHEDULE = 5
    NPU_MEMORY_ALLOCATION = 6
    CHECK_BLOCK_AND_TILE_SHAPE = 7
    NPU_TENSOR_LIFE_CYCLE_REPORT = 8
    NPU_MEMORY_ALLOCATION_2 = 9
    UPDATE_CONCAT_TENSOR_LIFE_CYCLE = 10

