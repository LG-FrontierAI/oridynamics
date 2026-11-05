"""推理输出目录：已有路径保留，以时间戳区分新任务，创建时原子处理竞争。"""
from datetime import datetime
from pathlib import Path
import re
import socket
from uuid import uuid4


def timestamped_output(prefix: Path) -> Path:
    host = re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname())
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    return prefix.with_name(f'{prefix.name}_{host}_{stamp}_{uuid4().hex[:8]}')


def output_directory(requested: Path, *, create: bool = False) -> Path:
    """预览不写盘；实际创建若遇到并发占用则换名，绝不复用已有目录。"""
    requested = requested.absolute()
    candidate = requested
    for _ in range(100):
        if create:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                # 父路径不是目录时直接报错，不把权限或路径错误当作任务名冲突。
                if not candidate.exists() and not candidate.is_symlink():
                    raise
        elif not candidate.exists() and not candidate.is_symlink():
            return candidate
        candidate = timestamped_output(requested)
    raise FileExistsError(f'Could not allocate a unique output directory for {requested}')
