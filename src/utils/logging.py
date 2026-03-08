from __future__ import annotations

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger("exam_reconciler")
logger.setLevel(logging.DEBUG)
logger.propagate = True
