import setuptools

setuptools.setup(
    name="DRF",
    version="0.1",
    author="Weibin Chen, So Takao",
    description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)