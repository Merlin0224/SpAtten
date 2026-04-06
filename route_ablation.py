import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import triton

from spatten_bert_ultimate import (
    TRITON_META_DEFAULTS,
    triton_fused_spatten_ultimate,
    triton_fused_spatten_v_prune,
    triton_progressive_qk,
)


