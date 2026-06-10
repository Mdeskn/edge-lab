"""Environment-configured Uvicorn launcher for the dashboard backend."""
import os

import uvicorn


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    """Run the API using the documented dashboard bind host and port variables."""
    uvicorn.run(
        "dashboard.backend.main:app",
        host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.environ.get("DASHBOARD_PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=_env_bool("DASHBOARD_ACCESS_LOG", False),
    )


if __name__ == "__main__":
    main()
