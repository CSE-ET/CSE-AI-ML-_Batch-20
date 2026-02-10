from setuptools import setup, find_packages

setup(
    name="realtime-facial-recognition",
    version="1.0.0",
    author="Senior CV Engineer",
    description="Real-time facial recognition using Haar Cascade and optimized LBPH",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-contrib-python>=4.9.0.80",
        "numpy>=1.26.0",
        "scipy>=1.11.1",
        "pyyaml>=6.0.1",
        "scikit-learn>=1.3.0",
        "pillow>=12.0.0",
        "tqdm>=4.66.1",
        "matplotlib>=3.9.2",
    ],
    python_requires=">=3.10",
)