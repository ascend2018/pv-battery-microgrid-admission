"""Storage-only failures; none imply a scientific experiment outcome."""


class StorageError(Exception):
    """Base error for compact storage."""


class ValidationError(StorageError, ValueError):
    """Malformed input, schema, or immutable stored representation."""


class QuotaExceeded(StorageError):
    """A reservation would exceed a regional/global or file limit."""


class RetentionError(StorageError):
    """A lifecycle change would lose an active or registered model."""


class IntegrityError(ValidationError):
    """Stored bytes or references violate their committed identity."""


class StateError(StorageError):
    """An operation is invalid in the writer's current state."""
