from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

requirements = ['tensorboard', 'scikit-learn', 'seqeval', 'psutil', 'sacrebleu', 'rouge-score',
                'tensorflow_datasets', 'pytorch-lightning==0.8.5', 'matplotlib', 'git-python==1.0.3',
                'streamlit', 'elasticsearch', 'pandas', 'conllu', 'spacy<3.0.0', 'inflect', 'transformers',
                'wandb']

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='kogito',
    version='0.0.1',
    description='A Python NLP Commonsense Reasoning library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mismayil/comet-atomic-2020',
    classifiers=[],
    author='EPFL NLP Lab',
    author_email='mahammad.ismayilzada@epfl.ch',
    license='Apache License 2.0',
    keywords='natural-language-processing nlp natural-language-understanding commonsense-reasoning',
    packages=find_packages(exclude=['scripts']),
    install_requires=requirements,
    python_requires='>=3.6',

    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [],
        'test': ['pytest'],
    },

    package_data={},
    include_package_data=True,
    data_files=[],
    entry_points={},
)