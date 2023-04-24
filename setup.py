from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="crickets",
    version='1.0',
    packages=find_packages(),
    entry_points={"console_scripts": "crickets = crickets:main"},
    author="E. Fazzari",
    description="Crickets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edofazza/Crickets/",
    setup_requires=[
        "pytest",
    ],	
    install_requires=[
        "pytest-shutil",
        "scipy",
        "numpy",
        "matplotlib",
        "pathlib",
	    "pandas",
        "ruamel.yaml",
	    "sklearn",
        "pyyaml",
        "opencv-python",
        "tensorflow",
        "deap",
        "sleap",
        "moviepy",
        "seaborn",
    ],
)
