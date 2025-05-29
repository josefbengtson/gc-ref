import os
import sys
from RoMa_functions.RoMa.roma.models import roma_outdoor, roma_indoor

DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1