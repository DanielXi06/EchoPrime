__all__ = ["EchoPrime", "EchoPrimeTextEncoder", "EchoPrimeBinaryClassifier"]


def __getattr__(name):
    if name in {"EchoPrime", "EchoPrimeTextEncoder"}:
        from .model import EchoPrime, EchoPrimeTextEncoder

        return {"EchoPrime": EchoPrime, "EchoPrimeTextEncoder": EchoPrimeTextEncoder}[name]
    if name == "EchoPrimeBinaryClassifier":
        from .video_classifier import EchoPrimeBinaryClassifier

        return EchoPrimeBinaryClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
