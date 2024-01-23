# CPOM Sea Ice Chain 

Sea ice altimetry chain **currently in development**

Developed using a fork of the generic CLEV2ER algorithm framework <https://github.com/cpomsoft/clev2er>, documented at **<https://cpomsoft.github.io/clev2er/>**

# Installation

cd to a directory where you want the software installed. A directory called **cpom_seaice** will be created in this directory.
Run the command:

with https:

```git clone https://github.com/CPOM-Altimetry/cpom_seaice.git```

or with ssh:

```git clone git@github.com:CPOM-Altimetry/cpom_seaice.git```

or with the GitHub CLI:

```gh repo clone CPOM-Altimetry/cpom_seaice```

## Shell Environment Setup

The following shell environment variables need to be set to support framework
operations. 

In a bash shell this might be done by adding export lines to your $HOME/.bashrc file.  

- Set the *CLEV2ER_BASE_DIR* environment variable to the root of the cpom sea ice package (ie path ending with ...../cpom_seaice)
- Add $CLEV2ER_BASE_DIR/src to *PYTHONPATH*.   
- Add ${CLEV2ER_BASE_DIR}/src/clev2er/tools to the *PATH*.   
- Set the shell's *ulimit -n* to allow enough file descriptors to be available for
    multi-processing.

An example environment setup is shown below (the path in the first line should be
adapted for your specific directory path):

```script
export CLEV2ER_BASE_DIR=/Users/someuser/software/cpom_seaice
export PYTHONPATH=$PYTHONPATH:$CLEV2ER_BASE_DIR/src
export PATH=${CLEV2ER_BASE_DIR}/src/clev2er/tools:${PATH}
# for multi-processing/shared mem support set ulimit
# to make sure you have enough file descriptors available
ulimit -n 8192
```

## Python Requirement

python v3.10 must be installed or available before proceeding.
A recommended minimal method of installation of python 3.10 is using miniconda as 
follows (other appropriate methods may also be used):

For miniconda installation, select the **python 3.10** installer for your operating 
system from:

https://docs.conda.io/en/latest/miniconda.html

For example, for Linux, download the installer and install 
a minimal python 3.10 installation using:

```script
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
chmod +x Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
./Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no] yes
```
You may need to start a new shell to refresh your environment before
checking that python 3.10 is in your path.

Check that python v3.10 is now available, by typing:

```
python -V
```

## Virtual Environment and Package Requirements

This project uses *poetry* (a dependency manager, see: https://python-poetry.org/) to manage 
package dependencies and virtual envs.

First, you need to install *poetry* on your system using instructions from
https://python-poetry.org/docs/#installation. Normally this just requires running:

`curl -sSL https://install.python-poetry.org | python3 -`

You should also then ensure that poetry is in your path, such that the command

`poetry --version`

returns the poetry version number. You may need to modify your 
PATH variable in order to achieve this.

### Install Required Python packages using Poetry

Run the following command to install python dependencies for this project
(for info, it uses settings in pyproject.toml to know what to install)

```
cd $CLEV2ER_BASE_DIR
poetry install
```


### Load the Virtual Environment

Now you are all setup to go. Whenever you want to run any CLEV2ER chains you 
must first load the virtual environment using the `poetry shell` command.

```
cd $CLEV2ER_BASE_DIR
poetry shell
```

Note that if you have the wrong version of python (not v3.10) in your path you will see
errors. You can tell poetry which version of python to use in this case using:

```
poetry env use $(which python3.10)
poetry shell
```

You should now be setup to run processing chains, etc.

## Run a simple chain test example

The following command will run a simple example test chain which dynamically loads
2 template algorithms and runs them on a set of CryoSat L1b files in a test data directory. 
The algorithms do not perform any actual processing as they are just template examples.
Make sure you have the virtual environment already loaded using `poetry shell` before
running this command.

`run_chain.py -n testchain -d $CLEV2ER_BASE_DIR/testdata/cs2/l1bfiles`

There should be no errors.

Note that the algorithms that are dynamically run are located in 
$CLEV2ER_BASE_DIR/src/clev2er/algorithms/testchain/alg_template1.py, alg_template2.py

The list of algorithms (and their order) for *testchain* are defined in 
$CLEV2ER_BASE_DIR/config/algorithm_lists/testchain/testchain_A002.xml

Algorithm configuration settings are defined in
$CLEV2ER_BASE_DIR/config/main_config.xml and
$CLEV2ER_BASE_DIR/config/chain_configs/testchain/testchain_A002.xml

To find all the command line options for *run_chain.py*, type:

`python run_chain.py -h`

For further info, please see `clev2er.tools`

## Developer Requirements

This section details additional installation requirements for developers who will develop/adapt 
new chains or algorithms.

### Install pre-commit hooks

pre-commit hooks are static code analysis scripts which are run (and must be passed) before
each git commit. For this project they include pylint, ruff, mypy, black, isort, pdocs.

To install pre-commit hooks, do the following: (note that the second line is not necessary if 
you have already loaded the virtual environment using `poetry shell`)

```
cd $CLEV2ER_BASE_DIR
poetry shell
pre-commit install
pre-commit run --all-files
```

Now, whenever you make changes to your code, it is recommended to run the following
in your current code directory.  

```pre-commit run --all-files```

This will check that your code passes all static code
tests prior to running git commit. Note that these same tests are also run when
you do a new commit, ie using `git commit -a -m "commit message"`. If the tests fail
you must correct the errors before proceeding, and then rerun the pre-commit and/or git commit.


