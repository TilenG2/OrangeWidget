from setuptools import setup

setup(name="Uncertain",
      packages=["uncertain"],
      package_data={"uncertain": ["icons/*.svg"]},
      classifiers=["Example :: Invalid"],
      # Declare uncertain package to contain widgets for the "Uncertainty" category
      entry_points={"orange.widgets": "Uncertainty = uncertain"},
      )