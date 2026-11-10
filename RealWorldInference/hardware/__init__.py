import os
import importlib
from typing import Dict, Any, Type, Optional

import yaml

from .controller import Controller

controllers: Dict[str, Type[Controller]] = {}


def _load_single_controller(name: str) -> Optional[Type[Controller]]:
    """
    懒加载单个controller，只加载指定的controller，避免加载所有controller导致环境冲突

    Args:
        name: 控制器类型（目录名）

    Returns:
        Controller类，如果不存在则返回None
    """
    # 如果已经加载过，直接返回
    if name in controllers:
        return controllers[name]
    package_name = __name__  # 一般是 "hardware"

    # 只导入指定的controller模块
    module_name = f"{package_name}.{name}.controller"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only treat the requested controller package/module as missing.
        # Missing dependencies inside an existing controller should surface with
        # their real traceback instead of being reported as "Controller not found".
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return None
        raise

    # 递归收集所有 Controller 子类（含间接子类，例如 RMBenchController → RoboTwinController → Controller）
    def _all_subclasses(cls):
        result = set(cls.__subclasses__())
        for sub in list(result):
            result.update(_all_subclasses(sub))
        return result

    for cls in _all_subclasses(Controller):
        module = cls.__module__  # 例如 "hardware.ac_one.controller"
        if not module.startswith(package_name + "."):
            continue
        parts = module.split(".")
        # 期望格式为 hardware.<dir>.controller
        if len(parts) < 3:
            continue
        controller_type = parts[1]  # 子目录名，例如 "ac_one"
        if controller_type == name:
            controllers[controller_type] = cls
            return cls

    return None


def get_config(name: str) -> Dict[str, Any]:
    """
    获取配置，name 为控制器类型（目录名）
    """
    _set_default_sim_env_vars()
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, name, "config.yaml")
    with open(config_path, "r") as f:
        return _expand_env_vars(yaml.safe_load(f))


def _set_default_sim_env_vars():
    """Fill in the RoboCasa365 source-tree paths from SIM_ROOT.

    run_eval.sh exports these for its subprocesses; setting them here keeps
    get_controller()/get_config() usable from ad-hoc Python entrypoints.
    Existing values always win.
    """
    rwai_dir = os.path.dirname(os.path.dirname(__file__))
    sim_root = os.environ.get("SIM_ROOT") or os.path.abspath(
        os.path.join(rwai_dir, "..", "sim")
    )
    defaults = {
        "SIM_ROOT": sim_root,
        "ROBOCASA365_DIR": os.path.join(sim_root, "robocasa"),
        "ROBOCASA365_ROBOSUITE_DIR": os.path.join(sim_root, "robosuite-master"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _expand_env_vars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value


def get_preview_adapter(name: str, slave_config: Dict[str, Any]):
    """Load and instantiate the per-robot debug-preview adapter.

    Each robot directory that supports debug preview must provide
    `hardware/<name>/preview_adapter.py` exporting a top-level
    `get_preview_adapter(slave_config) -> BasePreviewAdapter`.

    Raises ModuleNotFoundError if the robot has no preview adapter, and
    AttributeError if the module exists but doesn't expose the factory.
    Both are intentional: debug preview must fail loudly when not wired
    up rather than silently rendering garbage.
    """
    package_name = __name__
    module_name = f"{package_name}.{name}.preview_adapter"
    module = importlib.import_module(module_name)
    if not hasattr(module, "get_preview_adapter"):
        raise AttributeError(
            f"{module_name} must define a top-level `get_preview_adapter(config)` factory"
        )
    return module.get_preview_adapter(slave_config)


def get_controller(name: str, in_collect: bool = False, **overrides) -> Controller:
    """
    获取控制器实例，name 为控制器类型（目录名）

    使用懒加载策略，只加载指定的controller，避免加载所有controller导致环境冲突。
    overrides 中的键值会覆盖 hardware config 中的同名字段（如 debug_sim=True）。
    """
    # 使用懒加载，只加载指定的controller
    controller_cls = _load_single_controller(name)
    if controller_cls is None:
        raise ValueError(f"Controller {name} not found")

    config = get_config(name)
    config.update(overrides)
    return controller_cls(config=config, in_collect=in_collect)
