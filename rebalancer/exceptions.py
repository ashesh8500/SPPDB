class InsufficientDataError(Exception):
    """Raised when there's not enough historical data for analysis."""
    pass

class InvalidWeightError(Exception):
    """Raised when portfolio weights are invalid."""
    pass
