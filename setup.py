import setuptools

setuptools.setup(
    name="syndisk",
    version="0.0.1",
    author="Richard Teague",
    author_email="richard.d.teague@cfa.harvard.edu",
    description="Tools to simulate observations of protoplanetary disks.",
    url="https://github.com/richteague/syndisk",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
        "gofish",
      ]
)
