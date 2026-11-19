"""Create an account.  Usage: python scripts/create_user.py <username> <role: worker|researcher|admin> [password]
If no password is given you are prompted (hidden input)."""
import getpass, sys
import _common  # noqa
from src.auth import create_user

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(__doc__)
    pw = sys.argv[3] if len(sys.argv) > 3 else getpass.getpass("Password: ")
    create_user(sys.argv[1], pw, sys.argv[2]); print("User created.")
