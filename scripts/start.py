"""Seed ephemeral storage and start the production web server."""

import os

import uvicorn

from scripts.seed import seed


def main() -> None:
    seed()
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()