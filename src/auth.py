"""Authentication helpers: salted PBKDF2-HMAC-SHA256 password hashes, roles, simple lockout.
No Streamlit import here so it can be unit-tested and reused by the API."""
import hashlib, hmac, os, re, time
import pandas as pd
from .config import USERS_FILE

ITERATIONS = 200_000
MAX_FAILS, LOCK_SECONDS = 5, 300
ROLES = ("worker", "researcher", "admin")
_fails = {}   # username -> (count, last_time)  (in-memory; use a database/redis in production)


def hash_password(password, salt=None, iterations=ITERATIONS):
    salt = salt or os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${dk}"


def verify_password(password, stored):
    try:
        _, it, salt, dk = stored.split("$")
        cand = hash_password(password, salt, int(it)).split("$")[3]
        return hmac.compare_digest(cand, dk)
    except Exception:
        return False


def password_problems(pw):
    out = []
    if len(pw) < 8: out.append("at least 8 characters")
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw): out.append("letters and digits")
    return out


def _load(path):
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame(columns=["username", "password_hash", "role"])


def create_user(username, password, role="worker", path=USERS_FILE):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", username): raise ValueError("Username: 3-30 chars (a-z, 0-9, _ . -)")
    if role not in ROLES: raise ValueError("Unknown role")
    prob = password_problems(password)
    if prob: raise ValueError("Password needs " + " and ".join(prob))
    users = _load(path)
    if (users["username"] == username).any(): raise ValueError("Username already exists")
    users.loc[len(users)] = [username, hash_password(password), role]
    path.parent.mkdir(parents=True, exist_ok=True)
    users.to_csv(path, index=False)


def authenticate(username, password, path=USERS_FILE):
    """Return role on success, None on failure."""
    u = username.strip().lower()
    cnt, last = _fails.get(u, (0, 0))
    if cnt >= MAX_FAILS and time.time() - last < LOCK_SECONDS:
        return None
    users = _load(path)
    row = users[users["username"] == u]
    ok = (not row.empty) and verify_password(password, row.iloc[0]["password_hash"])
    if ok:
        _fails.pop(u, None)
        return row.iloc[0]["role"]
    _fails[u] = (cnt + 1, time.time())
    return None


def is_locked(username):
    cnt, last = _fails.get(username.strip().lower(), (0, 0))
    return cnt >= MAX_FAILS and time.time() - last < LOCK_SECONDS
