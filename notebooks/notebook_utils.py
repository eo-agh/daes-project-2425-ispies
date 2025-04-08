import os

GITIGNORE = ".gitignore"


def set_root_directory():
    """Set the project root directory in the notebook."""
    dir_list = os.listdir(".")
    if GITIGNORE in dir_list:
        return
    os.chdir("..")
    set_root_directory()
