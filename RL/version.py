version_info = (0, 1, 2)
# format:
# ('RL_major', 'RL_minor', 'RL_patch')

def get_version():
    "Returns the version as a human-format string."
    return '%d.%d.%d' % version_info

__version__ = get_version()
