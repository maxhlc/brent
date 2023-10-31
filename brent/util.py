# Standard imports
import itertools


def generate_parameter_permutations(parameters):
    # Extract keys and values
    keys, values = zip(*parameters.items())

    # Return generated permutations
    return [dict(zip(keys, v)) for v in itertools.product(*values)]
