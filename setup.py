import setuptools



setuptools.setup(
    name="us_workshop",
    version="0.1.0",
    author="lab4us",
    author_email="support@lab4us.eu",
    description="us_workshop",
    long_description="us_workshop",
    long_description_content_type="text/markdown",
    url="https://lab4us.eu",
    packages=setuptools.find_packages(exclude=[]),
    classifiers=[
        "Development Status :: 1 - Planning",

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    install_requires=[
        "numpy >= 1.22.3",
        "scipy >= 1.8.0",
        "matplotlib >= 3.5.1",
        "h5py >= 3.8.0"
    ],
    python_requires='>=3.8'
)