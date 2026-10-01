"""training/wandb_logger.py, with a fake `wandb` module standing in for the real one — no network,
no real W&B run, matching this project's CPU-only test convention. WandbLogger.__new__ bypasses
__init__'s real `wandb.init()` call, the same bare-object pattern test_training_mechanics.py uses
for Trainer."""
from training.wandb_logger import WandbLogger


class _FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.files_added = []

    def add_file(self, path):
        self.files_added.append(path)


class _FakeWandb:
    def __init__(self):
        self.logged = []  # (artifact, aliases) pairs

    def Artifact(self, name, type, metadata=None):
        return _FakeArtifact(name, type, metadata or {})

    def log_artifact(self, artifact, aliases=None):
        self.logged.append((artifact, aliases))


def _fake_logger(enabled: bool = True) -> WandbLogger:
    logger = WandbLogger.__new__(WandbLogger)
    logger.enabled = enabled
    logger._wandb = _FakeWandb()
    return logger


def test_log_artifact_attaches_metadata_aliases_and_the_file(tmp_path):
    logger = _fake_logger()
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")

    logger.log_artifact(str(ckpt), name="myrun-best", type_="model",
                        metadata={"epoch": 3, "val_macro_dice": 0.6}, aliases=["best"])

    (artifact, aliases), = logger._wandb.logged
    assert artifact.name == "myrun-best" and artifact.type == "model"
    assert artifact.metadata == {"epoch": 3, "val_macro_dice": 0.6}
    assert artifact.files_added == [str(ckpt)]
    assert aliases == ["best"]


def test_log_artifact_defaults_metadata_and_aliases_to_empty():
    logger = _fake_logger()
    logger.log_artifact("some/path.pt", name="n", type_="model")
    (artifact, aliases), = logger._wandb.logged
    assert artifact.metadata == {}
    assert aliases is None


def test_log_artifact_is_a_noop_when_disabled():
    logger = _fake_logger(enabled=False)
    logger.log_artifact("some/path.pt", name="n", type_="model")
    assert logger._wandb.logged == []
