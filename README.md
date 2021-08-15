# OnlineLDS
Source code for the AAAI 2019 paper "On-Line Learning of Linear Dynamical Systems: Exponential Forgetting in Kalman Filters" (https://arxiv.org/abs/1809.05870)

3rd version

What's done in this version
1.All the classes, their methods and arguments are named according to styleguide requirements. Method was named predict.

2.Created folder LDS. Each folder inside contains some similar classes and their corresponding unit tests. Consists of folders: ds(Dynamical System), ts(Time Series), h_m(Hankel), matlab_options(Matlab class options), filters, online_lds(cost_ftl,gradient_ftl,print_verbose).
Folder filters contains filtering_abc_class, filtering_siso, wave_filtering_siso_abs programs and folder real_filters with kalman_filtering_siso, wave_filtering_siso, wave_filtering_siso_ftl programs.
Folder online_lds is created only for the wave_filtering_siso_ftl program. Three function from previous onlinelds.py were moved here.
Inside the programs all the classes are named according to traditional style.

3.Setuptools.py and MANIFEST.in are outside the folder LDSnew.


4.Each class has a unit test. They're pretty simple though.   Each test is pretty basic and right now just checks a single attribute or argument. 
Can't check the test for wave_filtering_siso_ftl gives error:  minpack2.dcsrch() 1st argument (stp) can't be converted to double
To run any of the tests we need to be in the folder LDSnew and type in the terminal something like:
python -m AddClasses.ds.test_dynamical_system 

5.Added very brief docstrings, some of them are pretty obvious. Need to know more about returns of methods.
Also I can't understand G.t_t in the function predict of KalmanFilteringSISO class. G is a numpy array, so it doesn't have argument t_t. And it's not used anyhow in the main program.


Setuptools.py:
from setuptools import setup, find_packages

setup(
    name='LDS',
    version='1.0',
    author='Jakub Mareček, Kapa Kudaibergenov',
    author_email='jakub.marecek@gmail.com, kudkapa59@gmail.com',
    packages=find_packages(),
)

MANIFEST.in:
include LDSnew/OARIMA_code_data/*
include LDSnew/OARIMA_code_data/code/*
include LDSnew/OARIMA_code_data/data/*
include LDSnew/*
include LDSnew/slides/*
include LDSnew/outputs/*
