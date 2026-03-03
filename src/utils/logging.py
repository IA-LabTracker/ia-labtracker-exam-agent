from __future__ import annotations

import logging

# configure basic logging for any module that uses the standard logging API.
# uvicorn may add its own handlers, but basicConfig ensures there's at least
# one handler attached when running with `python src/main.py` or under tests.
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("exam_reconciler")
logger.setLevel(logging.INFO)
# propagate through the root logger so that uvicorn's handlers pick up our
# messages; without this you may see nothing in console when uvicorn is used.
logger.propagate = True
