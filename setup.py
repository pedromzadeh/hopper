from setuptools import setup

setup(
    name="hopper",
    version="1.0",
    author="Pedrom Zadeh",
    description="Modeling single cell dynamics in confinement",
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
