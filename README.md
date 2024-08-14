# anyburl-experiments
Script to generate data and scripts for anyburl experiments.

### Setup

    cd anyburl-experiments
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements
    pip install git+https://github.com/symbolic-kg/PyClause.git
    python -m prepare

The script will create lots of folders and files containing data, configuration files, and scripts (see below).

    cp learn.py ranking100.py metrics100.py hetionet/final/graph_on_all

### Installing PyClause on Sonoma with Mac Intel

Installing PyClause on MacOs Sonoma on a Mac Intel machibe usually fails because Apple decided to remove support to `openmp` from their c/c++ compiler. To work around this, we need to install an open source compiler that supports this library.

In order to complete the installation follow these steps:

1. Install the command line tool by running the following command on your termibal: `xcode-select --install`
2. Install `brew` with following command: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Honebrew/install/HEAD/install.sh)"`
3. Install `llvm` and `openmp` with the following command: `brew install llvm libomp` -- notice that we have also to `export CPPFLAGS="-I/ust/local/opt/libomp/include"` and `export LDFLAGS="-L/usr/local/opt/libomp/lib"` and ` export CC=/usr/local/opt/llvm/bin/clang`
4. Upgrade `pip` and `setuptools`: `pip3 install --upgrade pip` and `pip3 install --upgrade setuptools`
5. Run `pip install -e .`

You might still encounter an error (clang not recognising the parameter `-march=native`). If so, try the following commands:

1. `export CPPFLAGS="-I/usr/local/opt/libomo/include -Xclang -fooenmp"`
2. `export LDFLAGS="-L/usr/local/opt/libomp/lib"`
3. `export SYSTEM_VERSION_COMPAT=1`

And finally: `pip install -e .`

the above allowed me to succesfully install PyClause system-wide on my machine.

### Run Experiments

    cd hetionet/final/graph_on_all
    ./1_learn.sh
    ./2_apply.sh
    ./3_eval.sh
    ./4_explain.sh

I couldn't maje this ./4_explain.sh to work: the documentation suggested to try what's in `2_alt__apply_explain.sh` which seems to work. Hit ratio and MRR are all 0s (but I hadn't run `learn.sh` for 1_000 as suggested).

With PyClause:

    cd hetionet/final/graph_on_all
    cd data
    grep treats hetionet-v1.0-test.tsv> hetionet-v1.0-test-treats.tsv
    cd ..
    python learn.py
    pyhton ranking100.py
    python metrics100.py 


### Next Steps
Actually understand the results and check them!
