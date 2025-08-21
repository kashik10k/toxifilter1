from config import load_config
from sentence_model import SentenceModel
from sentence_model_com import MultiDetector

cfg = load_config()

toxic = SentenceModel(
    model_dir=cfg.paths.toxic_model_dir,
    pt_path=cfg.paths.toxic_pt,
    threshold=cfg.thr.toxic,
    max_length=cfg.thr.max_length,
)
sexual = SentenceModel(
    model_dir=cfg.paths.sexual_model_dir,
    pt_path=cfg.paths.sexual_pt,
    threshold=cfg.thr.sexual,
    max_length=cfg.thr.max_length,
)
multi = MultiDetector(
    toxic_model_dir=cfg.paths.toxic_model_dir,
    sexual_model_dir=cfg.paths.sexual_model_dir,
    toxic_pt_path=cfg.paths.toxic_pt,
    sexual_pt_path=cfg.paths.sexual_pt,
    threshold=min(cfg.thr.toxic, cfg.thr.sexual),
)

samples = [
    "I hate you, idiot.",
    "18+ hot v!deos watch now",
    "Have a nice day",
]
for s in samples:
    print(s, "->", multi.predict(s))
