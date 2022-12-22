from setuptools import setup,find_packages

requirements=[r.strip() for r in open("requirements.txt").readlines()]

setup(
   name='speechlmscore',
   version='1.0.0',
   description='a speech evaluation metric',
   author='Soumi Maiti',
   author_email='smaiti@andrew.cmu.edu',
   url="https://github.com/soumimaiti/speechlmsocre_tool",
   packages=find_packages(),
   install_requires=requirements,
)