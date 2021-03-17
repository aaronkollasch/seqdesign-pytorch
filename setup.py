import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="seqdesign_pt",
    version="1.0.0",
    author="Aaron Kollasch",
    author_email="awkollasch@gmail.com",
    description="Protein design and variant prediction using autoregressive generative models",
    license="MIT",
    keywords="autoregressive protein sequence design deep learning generative",
    url="https://github.com/aaronkollasch/seqdesign-pytorch",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    scripts=[
        "bin/calc_logprobs_seqs_fr",
        "bin/generate_sample_seqs_fr",
        "bin/run_autoregressive_fr",
    ],
)
