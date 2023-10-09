from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="cece",
    version="0.1.01",
    description="Conceptual Edits as Counterfactual Explanations",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Giorgos Filandrianos",
    author_email="georgefilandr@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy >= 1.20.1", "networkx >= 2.5"],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.8.8",
)