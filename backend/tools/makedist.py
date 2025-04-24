import argparse
import os
from qpt.executor import CreateExecutableModule as CEM
from qpt.modules.cuda import CopyCUDAPackage
from qpt.smart_opt import set_default_pip_source
from qpt.kernel.qinterpreter import PYPI_PIP_SOURCE
from qpt.modules.package import CustomPackage, DEFAULT_DEPLOY_MODE



def main():
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LAUNCH_PATH = os.path.join(WORK_DIR, 'gui.py')
    SAVE_PATH = os.path.join(os.path.dirname(WORK_DIR), 'vsr_out')
    ICON_PATH = os.path.join(WORK_DIR, "design", "vsr.ico")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="打包程序")
    parser.add_argument(
        "--cuda",
        nargs="?",                # 可选参数值
        const="11.8",             # 如果只写 --cuda，默认值是 10.2
        default=None,             # 不写 --cuda，则为 None
        help="是否包含CUDA模块，可指定版本，如 --cuda 或 --cuda=11.8"
    )
    parser.add_argument(
        "--directml",
        nargs="?",                # 可选参数值
        const=True,               # 如果只写 --directml，默认为True
        default=None,             # 不写 --directml，则为 None
        help="是否使用DirectML加速，仅指定 --directml 即可启用"
    )

    args = parser.parse_args()

    sub_modules = []

    if args.cuda == "11.8":
        sub_modules.append(CustomPackage("torch==2.7.0 torchvision==0.22.0", deploy_mode=DEFAULT_DEPLOY_MODE, find_links=PYPI_PIP_SOURCE, opts="--index-url https://download.pytorch.org/whl/cu118 "))
    elif args.cuda == "12.6":
        sub_modules.append(CustomPackage("torch==2.7.0 torchvision==0.22.0", deploy_mode=DEFAULT_DEPLOY_MODE, find_links=PYPI_PIP_SOURCE, opts="--index-url https://download.pytorch.org/whl/cu126 "))
    elif args.cuda == "12.8":
        sub_modules.append(CustomPackage("torch==2.7.0 torchvision==0.22.0", deploy_mode=DEFAULT_DEPLOY_MODE, find_links=PYPI_PIP_SOURCE, opts="--index-url https://download.pytorch.org/whl/cu128 "))

    if args.directml:
        sub_modules.append(CustomPackage("torch_directml==0.2.5.dev240914", deploy_mode=DEFAULT_DEPLOY_MODE))

    if os.getenv("QPT_Action") == "True":
        set_default_pip_source(PYPI_PIP_SOURCE)

    module = CEM(
        work_dir=WORK_DIR,
        launcher_py_path=LAUNCH_PATH,
        save_path=SAVE_PATH,
        icon=ICON_PATH,
        hidden_terminal=False,
        requirements_file="./requirements.txt",
        sub_modules=sub_modules,
    )

    module.make()


if __name__ == '__main__':
    main()