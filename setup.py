import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="pytorch_tokenizers"),
    package_dir={"": "pytorch_tokenizers"},
)
