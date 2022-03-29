import os
import setuptools
from pathlib import Path
from packaging import version


# --- Compute the version
NAME = "triton-python-model"
VERSION = os.environ.get("TRITON_PYTHON_MODEL_VERSION")
assert VERSION is not None, "Set env variable TRITON_PYTHON_MODEL_VERSION as the package version"
if VERSION is not None:
    assert VERSION[0] == "v", "Env variable TRITON_PYTHON_MODEL_VERSION should start with 'v' and use vX.Y.Z syntax"
    VERSION = VERSION[1:]
try:
    version.Version(VERSION)  # assert if regex fails
except version.InvalidVersion as err:
    print(
        str(err) + f"\nVersion must match the regex: {version.VERSION_PATTERN}")


def set_version_in_file(version="HEAD"):
    ini_file_path = Path(__file__).parent / \
        "triton_python_model" / "__init__.py"
    ini_file_lines = list(open(ini_file_path))
    with open(ini_file_path, "w") as f:
        for line in ini_file_lines:
            if line.startswith("__version__"):
                f.write("__version__ = \"{}\"\n".format(version))
            else:
                f.write(line)


# -- Auto-update the build version in the library
set_version_in_file(VERSION)

# --- Cache readme into a string
README = ""
if Path("README.md").exists():
    with open("README.md", "r", encoding="utf-8") as fh:
        README = fh.read()

# -- Extract dependencies
REQS = [line.strip() for line in open("requirements.txt")]
INSTALL_PACKAGES = [line for line in REQS if not line.startswith("#")]

# -- Build the whl file
setuptools.setup(
    name=NAME,
    version=VERSION,
    description="A python package to streamline python model serving in Visual Data Processing (VDP) project",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Instill AI",
    author_email="support@instill.tech",
    license="Apache 2.0",
    install_requires=INSTALL_PACKAGES,
    url="https://github.com/instill-ai/trition-python-model",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
)

# --- Revert the version in the source file
set_version_in_file("HEAD")
