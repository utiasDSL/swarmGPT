"""Exception collection for swarm_gpt."""


class LLMException(Exception):
    """Raised on LLM errors."""


class LLMFormatError(LLMException):
    """Raised when the LLM format is not as expected."""


class LLMPlanError(LLMException):
    """Raised when the LLM plan violates the constraints."""


class LLMResponseProcessingError(LLMException):
    """Raised when the LLM response cannot be processed."""
