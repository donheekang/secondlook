import os
import sys
import httpx

api_base = os.environ.get("API_BASE")
token = os.environ.get("ADMIN_TOKEN")

if not api_base or not token:
    print("Missing API_BASE or ADMIN_TOKEN env vars", file=sys.stderr)
    sys.exit(2)

url = api_base.rstrip("/") + "/admin/symbols/update"
headers = {"Authorization": f"Bearer {token}"}

try:
    r = httpx.post(url, headers=headers, timeout=120)
    print("status:", r.status_code)
    print(r.text)
    r.raise_for_status()
except Exception as e:
    print("error:", repr(e), file=sys.stderr)
    sys.exit(1)
