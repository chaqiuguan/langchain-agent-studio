"""零成本演示入口：不需要 API Key、不需要 pip install，直接运行

    python demo.py

（等价于安装后执行 `python -m agentstudio.demo`）
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentstudio.demo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
