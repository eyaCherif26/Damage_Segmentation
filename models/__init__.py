"""Importing this package registers all built-in models (the @register_model decorators in
smp_model.py / segformer_model.py only run on import)."""
from models import torchvision_encoders as _torchvision_encoders  # noqa: F401  (registers the tv- encoder)
from models import segformer_model as _segformer_model  # noqa: F401
from models import smp_model as _smp_model  # noqa: F401
from models.registry import build_model, registered_models  # noqa: F401
