import tempfile, unittest
from pathlib import Path
import helpers  # noqa
from src import auth


class TestAuth(unittest.TestCase):
    def setUp(self):
        self.p = Path(tempfile.mkdtemp()) / "users.csv"; auth._fails.clear()

    def test_hash_is_salted_and_verifies(self):
        h1, h2 = auth.hash_password("abc12345"), auth.hash_password("abc12345")
        self.assertNotEqual(h1, h2)
        self.assertTrue(auth.verify_password("abc12345", h1)); self.assertFalse(auth.verify_password("wrong", h1))

    def test_create_and_login(self):
        auth.create_user("Asha", "secret123", "worker", self.p)
        self.assertEqual(auth.authenticate("asha", "secret123", self.p), "worker")
        self.assertIsNone(auth.authenticate("asha", "nope", self.p))
        self.assertNotIn("secret123", self.p.read_text())

    def test_validation(self):
        with self.assertRaises(ValueError): auth.create_user("a", "secret123", "worker", self.p)
        with self.assertRaises(ValueError): auth.create_user("valid", "short", "worker", self.p)
        auth.create_user("dup", "secret123", "worker", self.p)
        with self.assertRaises(ValueError): auth.create_user("dup", "secret123", "worker", self.p)

    def test_lockout(self):
        auth.create_user("lock", "secret123", "worker", self.p)
        for _ in range(auth.MAX_FAILS): auth.authenticate("lock", "bad", self.p)
        self.assertTrue(auth.is_locked("lock"))
        self.assertIsNone(auth.authenticate("lock", "secret123", self.p))


if __name__ == "__main__":
    unittest.main()
