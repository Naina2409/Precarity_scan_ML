import unittest
import helpers  # noqa
try:
    from fastapi.testclient import TestClient
    from api.main import app
    HAVE = True
except Exception:   # fastapi / httpx not installed
    HAVE = False


@unittest.skipUnless(HAVE, "fastapi and httpx are required")
class TestApi(unittest.TestCase):
    def test_options_and_validation(self):
        c = TestClient(app)
        self.assertEqual(c.get("/options").status_code, 200)
        self.assertEqual(c.post("/predict", json={"city": "Delhi"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
