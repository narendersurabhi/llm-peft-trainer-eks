from __future__ import annotations

import os


def world_info() -> dict[str, int | str]:
    return {
        "rank": int(os.getenv("RANK", "0")),
        "local_rank": int(os.getenv("LOCAL_RANK", "0")),
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
        "master_addr": os.getenv("MASTER_ADDR", "localhost"),
        "master_port": os.getenv("MASTER_PORT", "29500"),
    }
