# Spring 2025 - CS6601 - Assignment 0

## Instructor: Dr. Thad Starner

## Deadline: <font color = 'red'>Sunday, Jan 12th, 3:59 pm EST</font>

#### Released: Monday, Jan 6th, 7:59 am EST

This assignment is designed to help you get comfortable with your local python environment, Git, introduce you to jupyter notebooks, provide a refresher on the Python language, and introduce you to Gradescope. After following the setup process in this README, you will follow the instructions in `notebook.ipynb` to make your first graded submission! Let's get started!

If you have not setup your Python environment yet,

#### STOP!

Complete the environment setup by following the instructions found at https://github.gatech.edu/omscs6601/env_setup. Then return here.

A video demonstration of the instructions below can be found [here](https://youtu.be/53tSkbbwy0k).

### Table of Contents
- [Get repository](#repo)
- [Packages](#pkg)
- [Jupyter](#jupyter)
- [Summary](#summary)

<a name="repo"/></a>

## Setup a local repository

Georgia Tech has paid for Enterprise Github and you have the ability to build and store your code in private repositories. Your repositories can be found at https://github.gatech.edu/[YOUR_STUDENT_GT_ID], where `[YOUR_STUDENT_GT_ID]` should be replaced with your Georgia Tech username (i.e. gburdell3). Please make sure that any repository you create is **private** to avoid OSI violations.

[Georgia Tech Github](https://github.gatech.edu/)</br>
[Help on GitHub](https://docs.github.com/en/get-started/quickstart)</br>

First things first, let's pull this repository to your local machine:

```
git clone https://github.gatech.edu/omscs6601/assignment_0.git
```

**NOTE: You will be provided a private repository for each of the rest of the assignments, more details will be provided on Edstem on where to find them. Please make sure to clone from the provided personal private repository for the assignments as it is not guaranteed that the public repositories are the most up to date versions.**

<a name="fork-instructions"/></a>

## Instructions to create a private forked repository for assignments

If you find that you need to work off of a copy of a public repository (like this one), you can follow the instructions below to create a personal copy of the repository. Remember that the repository you create needs to be **private** so that you do not accidentally violate OSI policies. The instructions below are modified to reflect URLs for this Assignment 0 repository.

1. Login to github.gatech.edu and create a private repo with a name of your choosing. We will call ours `cs6601_a0`, and for demonstration purposes, our GT ID will be `gburdell3`. Double check that the repo is private, otherwise you may violate the OSI policy.

2. Clone the class repository for Assignment 0 by using the command in a local file directory of your choice.

```
git clone --bare https://github.gatech.edu/omscs6601/assignment_0.git
```

3. Change your directory into the newly cloned repository and then mirror this to the private repo you created on Github.

```
cd assignment_0.git
git push --mirror https://github.gatech.edu/gburdell3/cs6601_a0
cd ..
```

4. You can now delete the `assignment_0.git` directory from your local files if you wish (you will no longer need it).

5. Now that your private repository on Github matches the Assignment 0 repository, you can clone the private repo onto your local system.

```
git clone https://github.gatech.edu/gburdell3/cs6601_a0
```

6. Next you can change your local directory to point to this newly cloned repo and add a remote branch upstream pointing to the original Assignment 0 repository it was cloned from.

```
cd cs6601_a0
git remote add upstream https://github.gatech.edu/omscs6601/assignment_0.git
```

You check if the remote branch has been added using ``git remote -v``

7. Now you can use your repository like so:

```
git pull upstream master # the original repo 
git push origin master # your repo 
```

If you do not specify the remote, it will default to the origin (your repo)

8. If you are scared of pushing to upstream you can disable pushing to upstream using

```
git remote set-url --push upstream PUSH_DISABLED
```

<!-- > **Note:** If you are on Windows, students in the past have commonly reported an error during package installation that resembles the error in this [Github post](https://github.com/pytorch/pytorch/issues/34798). To fix this issue, head over to the [PyTorch site](https://pytorch.org) and follow the instructions to install torch manually in `ai_env`. If this does not work, you may also instead try running `conda install -c ankurankan pgmpy=0.1.10`. After trying one of the previous suggestions and getting a successful install, try `pip install -r requirements.txt` again. -->


<a name="pkg"/></a>

## Packages

![Python Logo](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)

We will be using multiple python packages throughout this class. Here are some of them:

* **jupyter** - interactive notebook (you will learn more about them soon)
* **numpy** - a package for scientific computing (multi-dimensional array manipulation)
* **matplotlib** - a plotting library
* **networkx** - a package for manipulating networks/graphs
* **pandas** - a package for data analysis
* **pgmpy** - library for probabilistic graphical models 

<!-- You can see the complete list of packages and required versions in [./requirements.txt](./requirements.txt). -->

Making sure that you are in an active Conda environment (i.e. `(ai_env)` as seen in the setup), you can install all the packages for an assignment by using the command ``pip install -r requirements.txt``, if the assignment comes with a `requirements.txt` file.

Please navigate to your cloned Assignment 0 directory (i.e. `cs6601_a0`), activate your environment (`conda activate ai_env`), and run `pip install -r requirements.txt`.

Once installed, you can run `pip freeze` to see the list of all of the packages installed in your `ai_env` environment.

<!-- > **Note:** If you are on Windows, students in the past have commonly reported an error during package installation that resembles the error in this [Github post](https://github.com/pytorch/pytorch/issues/34798). To fix this issue, head over to the [PyTorch site](https://pytorch.org) and follow the instructions to install torch manually in `ai_env`. If this does not work, you may also instead try running `conda install -c ankurankan pgmpy=0.1.10`. After trying one of the previous suggestions and getting a successful install, try `pip install -r requirements.txt` again. -->

<a name="jupyter"/></a>

## Jupyter

Sometimes the assignment repositories will come with a Jupyter Notebook which will contain the instructions for completing the assignment and the code cells that you need to fill out. To open Jupyter Notebook, navigate to the file directory containing the `notebook.ipynb` file, make sure your conda environment is running and that the libraries have already been installed, and then run the below command:

    (ai_env) $ jupyter notebook

This should automatically open the `notebook.ipynb` as a Jupyter Notebook. If it doesn't automatically open, you can access the Jupyter Notebook at [http://localhost:8888](http://localhost:8888/) in your browser.

<a name="summary"/></a>

## Assignment 0

Your first assignment in this course is to setup your Python environment following the above instructions and then to follow the tutorial in the provided Jupyter Notebook. Upon completion of the tutorial, you will upload two Python files (`first_submission.py` and `priority_queue.py`) to [Gradescope](https://www.gradescope.com). This submission is worth **2 points of your final grade**. More details and instructions can be found in the provided Jupyter notebook. Good luck!

**Note:** It is EXTREMELY IMPORTANT that you submit this assignment before the deadline as completion of the assignment is necessary for us to release the remaining course assignments to you. Please submit the first task at a minimum (it takes minimal effort) as soon as you have everything set up.

