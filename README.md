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

