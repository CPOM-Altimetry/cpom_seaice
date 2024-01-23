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

- Set the *CLEV2ER_BASE_DIR* environment variable to the root of the cpom sea ice package.  
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


