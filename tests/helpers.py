import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from src.data_generator import make_clean, make_messy  # noqa
