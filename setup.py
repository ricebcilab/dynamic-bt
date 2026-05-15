from setuptools import setup

setup(
    name='dynamic-bt',
    version='0.0.0',
    packages=['dynamic_bt', 'dynamic_bt.criteria', 'dynamic_bt.skills'],
    package_dir={
        'dynamic_bt': 'dynamic_bt',
        'dynamic_bt.criteria': 'criteria',
        'dynamic_bt.skills': 'skills',
    },
    install_requires=[
        'numpy',
        'pyyaml',
        'scipy',
    ],
)
