"""在运行实验前检查 PyCharm 解释器和核心依赖。"""
import importlib
import platform
import sys


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Interpreter:", sys.executable)
    print("Architecture:", platform.architecture()[0])
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("请在 PyCharm 中选择64位 Python 3.12解释器后重新创建 .venv。")
    if platform.architecture()[0] != "64bit":
        raise SystemExit("本项目需要64位 Python。")
    packages = ["numpy", "pandas", "sklearn", "sentence_transformers", "torch", "transformers"]
    for name in packages:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'unknown')}")
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("环境检查通过，可以运行 run_experiment.py。")


if __name__ == "__main__":
    main()
