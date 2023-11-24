"""
Setup
"""
import setuptools

setuptools.setup(name='actorcritic',
                 version='0.0.1',
                 description='Advantage Actor-Critic (A2C) Agent',
                 author='alpine-chamois',
                 url='https://github.com/alpine-chamois/actor-critic/',
                 license="MIT",
                 python_requires=">=3.10",
                 install_requires=['torch~=2.0.1',
                                   'stable_baselines3[extra]~=2.2.1',
                                   'pygame~=2.1.3',
                                   'gymnasium~=0.29.1'],
                 packages=setuptools.find_packages())
