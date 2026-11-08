"""与安装后的oridynamics命令使用相同入口；旧无子命令形式仍兼容。"""
from oridynamics.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
