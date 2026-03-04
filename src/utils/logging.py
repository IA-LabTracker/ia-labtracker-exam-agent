from __future__ import annotations

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("exam_reconciler")
logger.setLevel(logging.INFO)
logger.propagate = True
