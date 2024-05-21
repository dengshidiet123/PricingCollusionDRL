class AlgorithmNotIntegratedError(Exception):
    """When algorithm is not integrated with the problem."""


class EnvironmentDefinitionError(Exception):
    """Error due wrong environment definition for stable-baseline3, rllib."""
