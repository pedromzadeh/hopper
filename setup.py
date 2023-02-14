from setuptools import setup

setup(
    name="CCAM",
    version="1.0",
    author="Pedrom Zadeh",
    description="Characterizing collisions with acceleration maps",
    packages=[
        "box",
        "cell",
        "polarity",
        "potential",
        "simulator",
        "substrate",
        "helper_functions",
        "visuals",
        "analysis",
    ],
)
