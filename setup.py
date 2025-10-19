from setuptools import setup, find_packages

setup(
    name="realtime-facial-recognition",
    version="1.0.0",
    author="Senior CV Engineer",
    description="Real-time facial recognition using Haar Cascade and optimized LBPH",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python==4.9.0.80",
        "numpy==1.24.3",
        "scipy==1.11.1",
        "pyyaml==6.0.1",
    ],
    python_requires=">=3.10",
)