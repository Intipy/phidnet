import setuptools



with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()



setuptools.setup(
    
    name="phidnet",
    version="0.1.0",
    author="Intipy",
    author_email="jios6790@gmail.com",
    description="Phidnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Intipy/phidnet.git",
    packages=setuptools.find_packages(),
    
    install_requires = [
        "numpy",
        "matplotlib"
    ],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    python_requires='>=3.6',
    
)
