from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="para_eff_pt",  
    version="0.0",     
    description="Parameter Efficient Pretraining",  
    packages=find_packages(), 
    install_requires=required,  

)