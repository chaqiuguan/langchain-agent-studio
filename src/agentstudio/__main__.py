"""支持 `python -m agentstudio` 直接启动命令行界面。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
