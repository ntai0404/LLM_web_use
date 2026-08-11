import unittest
from pathlib import Path

from app import PROJECT_DIR, app, manager


class ManagementUiTests(unittest.IsolatedAsyncioTestCase):
    def test_dashboard_assets_and_routes_exist(self):
        html = Path(PROJECT_DIR, "web", "templates", "index.html").read_text(encoding="utf-8")
        css = Path(PROJECT_DIR, "web", "static", "app.css").read_text(encoding="utf-8")
        javascript = Path(PROJECT_DIR, "web", "static", "app.js").read_text(encoding="utf-8")
        paths = {route.path for route in app.routes}

        self.assertIn("LLM Web Router", html)
        self.assertIn(".provider-card", css)
        self.assertIn("/api/providers", javascript)
        self.assertIn('apiFetch("/api/generate"', javascript)
        self.assertIn("/", paths)
        self.assertIn("/static", paths)
        self.assertIn("/api/providers", paths)
        self.assertIn("/api/providers/{provider_name}/test", paths)
        self.assertIn("/api/providers/{provider_name}/keepalive", paths)

    async def test_provider_list_is_registry_backed_and_safe(self):
        providers = await manager.provider_list(refresh=False)

        self.assertEqual("gemini-web", providers[0]["id"])
        serialized = str(providers).lower()
        self.assertNotIn("cookie", serialized)
        self.assertNotIn("snlm0e", serialized)
        self.assertNotIn("local state", serialized)


if __name__ == "__main__":
    unittest.main()
