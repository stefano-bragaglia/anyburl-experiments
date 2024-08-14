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
