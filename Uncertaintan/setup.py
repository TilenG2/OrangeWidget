from setuptools import setup

setup(name="Uncertaintan",
      packages=["trees"],
      package_data={"trees": ["icons/*.svg"]},
      classifiers=["Example :: Invalid"],
      # Declare trees package to contain widgets for the "Trees" category
      entry_points={"orange.widgets": "Trees = trees"},
      )