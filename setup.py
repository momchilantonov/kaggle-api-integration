from setuptools import setup, find_packages

setup(
    name="kaggle_api_integration",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'kaggle==1.5.16',
        'pandas==2.1.4',
        'numpy==1.24.3',
        'python-dotenv==1.0.0',
        'requests==2.31.0',
        'tqdm==4.66.1',
        'python-slugify==8.0.1',
        'certifi==2023.11.17',
        'urllib3==2.1.0',
        'six==1.16.0',
    ],
    extras_require={
        'test': [
            'pytest==7.4.3',
            'pytest-cov==4.1.0',
            'pytest-mock==3.12.0',
        ],
    }
)
