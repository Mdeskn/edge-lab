"""Environment-configured Uvicorn launcher for the dashboard backend."""
import os

import uvicorn


def main() -> None:
    """Run the API using the documented dashboard host and port variables."""
    uvicorn.run(
        "dashboard.backend.main:app",
        host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.environ.get("DASHBOARD_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
