from setuptools import setup

setup(name="Uncertain",
      packages=["trees"],
      package_data={"trees": ["icons/*.svg"]},
      classifiers=["Example :: Invalid"],
      # Declare trees package to contain widgets for the "Trees" category
      entry_points={"orange.widgets": "Uncertainty = trees"},
      )