from types import SimpleNamespace


configs = SimpleNamespace(**{})

# cuda or cpu
configs.device = "cuda"

# project directories
configs.dataset_dir = "../../datasets"
configs.results_dir = "../../results"
configs.ckpt_dir = "../../checkpoints"
configs.runs_dir = "../runs"
configs.runs_tracker = "../runs/tracker.json"
configs.runs_stats_save_dir = "../runs/runs_stats_saves"