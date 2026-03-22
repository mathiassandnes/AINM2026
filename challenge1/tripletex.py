"""
Tripletex API client. Async wrapper around httpx with auth, error handling, and GET caching.
"""

import json as json_mod
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def _cache_key(path: str, params: dict | None) -> str:
    """Create a hashable cache key from path and params."""
    p = sorted((params or {}).items())
    return f"{path}|{json_mod.dumps(p, sort_keys=True)}"


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.client = httpx.AsyncClient(timeout=60.0)
        self.call_count = 0
        self.error_count = 0
        self._get_cache: dict[str, dict] = {}

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        self.call_count += 1

        # Retry on connection errors (cold start can cause timeouts)
        last_err = None
        for attempt in range(3):
            try:
                resp = await self.client.request(method, url, auth=self.auth, **kwargs)
                break
            except (httpx.ConnectTimeout, httpx.ConnectError) as e:
                last_err = e
                log.warning(f"[{self.call_count}] {method} {path} connect error (attempt {attempt+1}/3): {e}")
                import asyncio
                await asyncio.sleep(1 * (attempt + 1))
        else:
            raise last_err

        log.info(f"[{self.call_count}] {method} {path} -> {resp.status_code}")

        if resp.status_code >= 400:
            self.error_count += 1
            error_body = resp.text[:1000]
            log.warning(f"  Error body: {error_body}")
            return {"error": True, "status_code": resp.status_code, "message": error_body}

        if resp.status_code == 204 or not resp.text:
            return {}
        return resp.json()

    async def get(self, path: str, params: dict | None = None) -> dict:
        key = _cache_key(path, params)
        if key in self._get_cache:
            log.info(f"[cache] GET {path} (saved an API call)")
            return self._get_cache[key]
        result = await self._request("GET", path, params=params)
        if not result.get("error"):
            # Don't cache empty results — entities may be created later
            # and we need fresh lookups to find them
            values = result.get("values")
            if values is None or len(values) > 0:
                self._get_cache[key] = result
        return result

    async def post(self, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        return await self._request("POST", path, json=json, params=params)

    async def put(self, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        return await self._request("PUT", path, json=json, params=params)

    async def delete(self, path: str) -> dict:
        return await self._request("DELETE", path)

    async def get_all(self, path: str, params: dict | None = None) -> list[dict]:
        """GET with pagination, returns all values."""
        params = params or {}
        params.setdefault("count", 1000)
        params.setdefault("from", 0)
        result = await self.get(path, params=params)
        return result.get("values", [])

    async def close(self):
        await self.client.aclose()
