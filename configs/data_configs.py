from types import SimpleNamespace


cfg = SimpleNamespace(**{})
cfg.loading = {
    "tvsumm": {
    	"data_file": "../datasets/tvsumm/test_preprocessing.pt"
    }
}
cfg.video_dirs = {
    "tvsumm": "../datasets/tvsumm/videos",
    "summe": "../datasets/summe/videos",
}