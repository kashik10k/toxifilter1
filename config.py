import yaml, pathlib
from dataclasses import dataclass

@dataclass
class Paths:
    toxic_model_dir: str
    toxic_pt: str
    sexual_model_dir: str
    sexual_pt: str

@dataclass
class Thresholds:
    toxic: float
    sexual: float
    max_length: int

@dataclass
class RuntimeCfg:
    trigger_keys: list[str]
    mask: str

@dataclass
class EscalationCfg:
    suggest: int
    warn: int
    enforce: int
    decay_keystrokes: int

@dataclass
class LogCfg:
    file: str
    level: str

@dataclass
class AppCfg:
    paths: Paths
    thr: Thresholds
    runtime: RuntimeCfg
    esc: EscalationCfg
    log: LogCfg

def load_config(path: str = "config.yaml") -> AppCfg:
    cfg = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    return AppCfg(
        paths=Paths(cfg["toxic_model_dir"], cfg["toxic_pt"], cfg["sexual_model_dir"], cfg["sexual_pt"]),
        thr=Thresholds(cfg["threshold_toxic"], cfg["threshold_sexual"], cfg["max_length"]),
        runtime=RuntimeCfg(cfg["trigger_keys"], cfg["mask"]),
        esc=EscalationCfg(cfg["suggest_threshold"], cfg["warn_threshold"], cfg["enforce_threshold"], cfg["decay_after_keystrokes"]),
        log=LogCfg(cfg["log_file"], cfg["log_level"]),
    )
