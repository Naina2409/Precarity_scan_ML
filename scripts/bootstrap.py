"""One-command setup: synthetic data -> full analysis + first model -> fixed demo accounts.
Usage: python scripts/bootstrap.py"""
import subprocess, sys
from pathlib import Path
import _common  # noqa
from demo_users import create_demo_users

here = Path(__file__).parent
for s in ("generate_data.py", "run_analysis.py"):
    subprocess.run([sys.executable, str(here / s)], check=True)
print("\nDemo accounts (fixed, for demonstration only):")
create_demo_users()
print("\nNext: streamlit run app.py   |   uvicorn api.main:app --reload")