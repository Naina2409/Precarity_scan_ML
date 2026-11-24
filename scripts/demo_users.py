"""Create fixed DEMO accounts (safe to run repeatedly; existing users are skipped).
For demonstration only. Do not use these passwords with real users."""
import _common  # noqa
from src.auth import create_user

DEMO = [("admin", "Admin@12345", "admin"),
        ("researcher", "Research@12345", "researcher"),
        ("worker1", "Worker@12345", "worker")]


def create_demo_users():
    for name, pw, role in DEMO:
        try:
            create_user(name, pw, role)
            print(f"created  {role:10s} {name} / {pw}")
        except ValueError as e:
            print(f"skipped  {name}: {e}")


if __name__ == "__main__":
    create_demo_users()