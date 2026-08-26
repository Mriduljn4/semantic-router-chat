"""Run lightweight end-to-end regression checks against a running API."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen

from src.evaluation_cases import EVALUATION_CASES


def main() -> None:
    """Check routing, Tavily metadata, and required answer content for each case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    passed = 0

    for case in EVALUATION_CASES:
        request = Request(
            f"{args.url.rstrip('/')}/query",
            data=json.dumps({"query": case.query}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())

        answer = payload["answer"].lower()
        agent_ok = payload["routed_agent"] == case.expected_agent
        search_ok = ("web_search" in payload["tools_used"]) == case.expects_web_search
        quality_ok = all(check in answer for check in case.quality_checks)
        outcome = agent_ok and search_ok and quality_ok
        print(json.dumps({"query": case.query, "passed": outcome, "agent": payload["routed_agent"], "tools": payload["tools_used"]}))
        passed += outcome

    if passed != len(EVALUATION_CASES):
        raise SystemExit(f"{passed}/{len(EVALUATION_CASES)} evaluation cases passed")
    print(f"All {passed} evaluation cases passed.")


if __name__ == "__main__":
    main()