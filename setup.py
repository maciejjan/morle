from setuptools import setup, find_packages

with open('README.md') as fp:
    README = fp.read()

# TODO check whether the required shell commands are available:
# - hfst-fst2strings, hfst-xfst
# - sort

setup(
    name='morle',
    version='0.9.0',
    author='Maciej Sumalvico',
    author_email='macjan@o2.pl',
    description='A MORphemeLEss MORphology LEarner.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/mmj/morle',
    license='MIT',
    packages=find_packages('src', exclude=['tests', 'tests.*']),
    package_dir={'' : 'src'},
    test_suite='tests',
    install_requires=[
        'hfst >= 3.14.0',
        'keras >= 2.2.0',
        'networkx >= 1.10, < 2.0',
        'numpy >= 1.15.2',
        'pyyaml >= 3.13',
        'scipy >= 1.1.0',
        'tqdm >= 4.11.2'
    ],
    package_data={
        'morle': ['config-default.ini'],
    },
    entry_points={
        'console_scripts': [
            'morle = morle.main:main'
        ]
    },
)
