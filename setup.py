from setuptools import setup, find_packages
import sys

this_directory = sys.path[0]
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

requirements = [
    'spacy>=2.2.1',
    'scispacy>=0.2.4',
    'numpy>=1.17.4',
    'wmd>=1.3.2',
    'pandas>=0.25.3',
    'scikit-learn>=0.21.3',
    're>=2.2.1',
    'json>=2.0.9']

author = (
    "Yu Bai,"
    "Yunchen Yang"
)

setup(
    name='scireader',
    python_requires=">=3.6",
    version='0.0.1',
    author=author,
    author_email='yubailibra@yahoo.com',
    url='https://github.com/yubailibra/scireader',
    packages=find_packages(),
    install_requires=requirements,
    license="MIT license",
    description='ScispaCy-based natural language processing pipeline for scientific literature mining',
    keywords='scireader',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    long_description_content_type="text/markdown",
    long_description=long_description
)