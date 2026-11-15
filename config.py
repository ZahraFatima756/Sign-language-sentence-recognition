"""Central settings for the PSL sentence recogniser.

Everything that must match between the dataset and the live camera lives in the
"CAPTURE" block. Those values are copied from the author's app2_video.py, which
produced the dataset. Do not change them unless you also re-record the dataset.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
DATA_DIR = Path(os.environ.get("PSL_DATA_DIR", ROOT / "data" / "FinalDataset"))
CACHE_DIR = ROOT / "cache"
RUN_DIR = ROOT / "runs"
MODEL_DIR = RUN_DIR / "model"
REPORT_DIR = RUN_DIR / "reports"
EXAMPLE_DIR = RUN_DIR / "examples"
REAL_DIR = ROOT / "real_test"
SENTENCES_CSV = ROOT / "sentences.csv"

# ---------------------------------------------------------------- CAPTURE (must match app2_video.py)
CAPTURE_SIZE = 400       # every saved frame is 400 x 400, gray
WRITE_FPS = 10           # fps written into the .avi container
HAND_SIZE_MAX = 370      # the hand crop is resized to at most 370 px
PAD_VALUE = 1            # border gray value used by the author (background = 1)
THRESH_VALUE = 215       # binary threshold applied to the masked frame

# ---------------------------------------------------------------- model input
INK_THRESHOLD = 127      # saved frame: ink = 255, background = 1 -> binarise at 127
DILATE_KERNEL = 3        # thicken the 1-px skeleton lines before shrinking
MODEL_SIZE = 48          # CNN input is MODEL_SIZE x MODEL_SIZE
MOTION_THRESHOLD = 0.04 # mean per-frame change (0..1) that counts as movement; tuned on the dataset (see README)
MIN_CLIP_FRAMES = 3      # never trim a clip below this many frames
FRAME_STRIDE = 2         # 10 fps -> 5 fps for the sequence model

# ---------------------------------------------------------------- splits and stitching
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
VARIANTS_PER_SENTENCE = {"train": 300, "val": 40, "test": 40}
SPEED_RANGE = (0.8, 1.25)   # random time-warp per word
MAX_REST_FRAMES = 3         # idle frames added at the start and end of a sentence
BLEND_RANGE = (1, 3)        # blended frames between two words

# ---------------------------------------------------------------- training
EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 2e-3
PATIENCE = 6                # early stopping on validation WER
CONV_CHANNELS = (16, 32, 64, 96)
GRU_HIDDEN = 128

# ---------------------------------------------------------------- live
LIVE_MIN_FRAMES = 5         # refuse to predict from a recording shorter than this
SENTENCE_MATCH_MAX = 0.34   # max normalised sign-edit distance to accept a sentence
URDU_FONT_ENV = "PSL_URDU_FONT"  # path to an Urdu-capable .ttf (e.g. Noto Nastaliq Urdu)

# ---------------------------------------------------------------- folder name -> canonical sign
# Folder names are matched to sentence signs by their letters, ignoring the
# difference between Arabic ي/ك and Urdu ی/ک and the ی/ئ spelling variants.
# Add an explicit entry here only if automatic matching reports a problem.
SIGN_ALIASES: dict[str, str] = {}
