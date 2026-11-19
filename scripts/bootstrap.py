"""One-command setup: synthetic data -> full analysis + first model -> demo accounts.
Usage: python scripts/bootstrap.py
Demo passwords are random and printed ONCE (nothing is hard-coded or committed)."""
import secrets, subprocess, sys
from pathlib import Path
import _common  # noqa
from src.auth import create_user
from src.config import USERS_FILE

here = Path(__file__).parent
for s in ("generate_data.py", "run_analysis.py"):
    subprocess.run([sys.executable, str(here / s)], check=True)
if not USERS_FILE.exists():
    print("\nDemo accounts (save these now, they are not stored in plain text):")
    for name, role in (("admin", "admin"), ("researcher", "researcher"), ("worker1", "worker")):
        pw = secrets.token_urlsafe(9) + "7"
        create_user(name, pw, role); print(f"  {role:10s} username={name}  password={pw}")
print("\nNext: streamlit run app.py   |   uvicorn api.main:app --reload")
