""" File description.
"""
import hashlib
import json
import os
import re
import stat
from os import makedirs
from os.path import dirname
from os.path import exists
from os.path import join

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

tqdm.pandas()

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

TOP_K = 100
WALL_TIME = 1_000

PROGRAM_23_1 = 'AnyBURL-23-1.jar'
PROGRAM_22 = 'AnyBURL-22.jar'

METAGRAPH = 'hetionet/raw/hetionet-v1.0-metagraph.json'
EDGES = 'hetionet/raw/hetionet-v1.0-edges.sif.gz'
GRAPH = 'hetionet/interim/hetionet-v1.0-graph.tsv'

TRAIN_ALL = 'hetionet/final/graph_on_all/data/hetionet-v1.0-train.tsv'
VALID_ALL = 'hetionet/final/graph_on_all/data/hetionet-v1.0-valid.tsv'
TEST_ALL = 'hetionet/final/graph_on_all/data/hetionet-v1.0-test.tsv'

CFG_LEARN_ALL = 'hetionet/final/graph_on_all/config-learn.properties'
CFG_APPLY_ALL = 'hetionet/final/graph_on_all/config-apply.properties'
CFG_APPLY_EXPL_ALL = 'hetionet/final/graph_on_all/config-apply-explain.properties'
CFG_EVAL_ALL = 'hetionet/final/graph_on_all/config-eval.properties'
RULES_ALL = 'hetionet/final/graph_on_all/rules'
PREDS_ALL = 'hetionet/final/graph_on_all/preds'
EXPLAIN_ALL = 'hetionet/final/graph_on_all/explanations.txt'

SCR_LEARN_ALL = 'hetionet/final/graph_on_all/1_learn.sh'
SCR_APPLY_ALL = 'hetionet/final/graph_on_all/2_apply.sh'
SCR_APPLY_EXPL_ALL = 'hetionet/final/graph_on_all/2_alt__apply_explain.sh'
SCR_EVAL_ALL = 'hetionet/final/graph_on_all/3_eval.sh'
SCR_EXPLAIN_ALL = 'hetionet/final/graph_on_all/4_explain.sh'
EXPLAIN_DIR_ALL = 'hetionet/final/graph_on_all/explanations'

TRAIN_1ST = 'hetionet/final/graph_on_train/data/hetionet-v1.0-train.tsv'
VALID_1ST = 'hetionet/final/graph_on_train/data/hetionet-v1.0-valid.tsv'
TEST_1ST = 'hetionet/final/graph_on_train/data/hetionet-v1.0-test.tsv'

CFG_LEARN_1ST = 'hetionet/final/graph_on_train/config-learn.properties'
CFG_APPLY_1ST = 'hetionet/final/graph_on_train/config-apply.properties'
CFG_APPLY_EXPL_1ST = 'hetionet/final/graph_on_train/config-apply-explain.properties'
CFG_EVAL_1ST = 'hetionet/final/graph_on_train/config-eval.properties'
RULES_1ST = 'hetionet/final/graph_on_train/rules'
PREDS_1ST = 'hetionet/final/graph_on_train/predictions'
EXPLAIN_1ST = 'hetionet/final/graph_on_train/explanations.txt'

SCR_LEARN_1ST = 'hetionet/final/graph_on_train/1_learn.sh'
SCR_APPLY_1ST = 'hetionet/final/graph_on_train/2_apply.sh'
SCR_APPLY_EXPL_1ST = 'hetionet/final/graph_on_train/2_alt__apply_explain.sh'
SCR_EVAL_1ST = 'hetionet/final/graph_on_train/3_eval.sh'
SCR_EXPLAIN_1ST = 'hetionet/final/graph_on_train/4_explain.sh'
EXPLAIN_DIR_1ST = 'hetionet/final/graph_on_train/explanations'

DIGESTS = {
    PROGRAM_23_1:
        'e9ef2253f9900e41040a6823ec7b705fb1c557384fc4f7925177c60719e35d424d8e115aaa5e7532c8b12e94fbe81002a8c20cbd9c8ab29bc75980b308793452',
    PROGRAM_22:
        '777619a13ec8e35ecc13d1272408aac879708a0d4ce799daad0f3172d84beb6fe814549e5f9099354aecb498eadf822517748113671f33c1bb0d944588508784',
    METAGRAPH:
        '6a47903a5be866b40557071a11162347ffabd05c5698ba008aeb4943a1cc55a3187d5f9e67a626d4bec3725f6bb6befbf5d969a6230cdc89f1c896058b572b04',
    EDGES:
        'b9c5654481ed5f21660fa80450048ce73720d517228e6338cd880407bb4c857ad02e276059811932231539fe7705cde9820024634e72409cd9f2c98c68e8fca1',
    GRAPH:
        'd7e8cd41aa96fe68ec8b8da17e17a346f655cf4786ba7f77d40e0916d60032dc4b354abb26c56b5d8a85faa9c06b02632c4e8db38cb777b04d26baed4966643a',
    TRAIN_ALL:
        '3c7a70bee4f6ba140a566b0714ba1483e23de05de8cb5be35ca1c55479d4e4c1e10910c0a79ece9e312b5d5d00c431b0203f19fa18c4050a1c78198c454b9868',
    VALID_ALL:
        'ce1d6c42b3ef51fde2cf7a4f30efaea7660eaa2f680a695f6f20a7f8fd419d01b2669d08f291f5eb08f6ebca137bbe758e55a6c1d7f6ccc1f72dcea2a4db5ae0',
    TEST_ALL:
        'd8f44fa50a37488053c144d3a43caf5633ae91b698002b158a79053cb351b45bee4bc1a033dfbdf91211b347791cf72c76df383ac72a6177668963bfc28232c7',
    TRAIN_1ST:
        '3c7a70bee4f6ba140a566b0714ba1483e23de05de8cb5be35ca1c55479d4e4c1e10910c0a79ece9e312b5d5d00c431b0203f19fa18c4050a1c78198c454b9868',
    VALID_1ST:
        'f6cbed6925cb22ba570cf97e4ced8304fd4d29197d89bb486b1afc4a41a492f00fd497ca3c447a4efaaf762174505fae9926c020a795d6f7cc6d917b30aeae53',
    TEST_1ST:
        'c6aa0ab8e2739af6bec34dfb963a2e882fc3a644d44b6aeba0fb18b503c246cfc0a0744edce884fd9951faf2c53eaa1cd5be290db48fc1c430aeb7a420413ea6',
}

URLS = {
    PROGRAM_23_1:
        'https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-23-1.jar',
    PROGRAM_22:
        'https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-22.jar',
    METAGRAPH:
        'https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-metagraph.json',
    EDGES:
        'https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz',
}


def get_digest(filename: str, chunk_size: int = 8192) -> str:
    with open(filename, "rb") as file:
        file_hash = hashlib.blake2b()
        while chunk := file.read(chunk_size):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def download(root: str, filename: str, url: str, digests: dict[str, str], force: bool = False,
             chunk_size: int = 8192) -> None:
    fullname = join(root, filename)
    if force or not exists(fullname):
        with requests.get(url, allow_redirects=True) as r:
            r.raise_for_status()

            makedirs(dirname(fullname), exist_ok=True)
            with open(fullname, 'wb') as file:
                for chunk in tqdm(r.iter_content(chunk_size), desc=f"Downloading '{url}'"):
                    file.write(chunk)

    hex = get_digest(fullname, chunk_size)
    assert digests.get(filename) == hex, f"The downloaded file '{filename}' is corrupted (digest: {hex})"


def save_tsv(root: str, filename: str, df: pd.DataFrame) -> None:
    fullname = join(root, filename)
    makedirs(dirname(fullname), exist_ok=True)
    chunks = np.array_split(df.index, 100)  # split into 100 chunks
    for chunk, subset in enumerate(tqdm(chunks, desc=f"Saving to '{filename}'")):
        if chunk == 0:  # first row
            df.loc[subset].to_csv(fullname, mode='w', sep='\t', header=False, index=False)
        else:
            df.loc[subset].to_csv(fullname, mode='a', sep='\t', header=False, index=False)


def create_hetionet_graph(
        root: str, filename: str,
        metagraph: str, edges: str,
        digests: dict[str, str],
        force: bool = False,
        chunk_size: int = 8192
) -> None:
    fullname = join(root, filename)
    metagraph = join(root, metagraph)
    edges = join(root, edges)
    if force or not exists(fullname):
        with open(metagraph, 'r') as file:
            data = json.load(file)

        types = {l: c.replace(' ', '_').lower() for l, c in data['kind_to_abbrev'].items() if not c.islower()}
        norm_types = lambda x: f"{types[x.split('::', maxsplit=1)[0]]}__{x.split('::', maxsplit=1)[1].lower()}"

        rels = {c: l.replace(' ', '_').lower() for l, c in data['kind_to_abbrev'].items() if c.islower()}
        non_lower = re.compile('[^a-z]')
        norm_rels = lambda x: rels[non_lower.sub('', x)]

        df = pd.read_csv(edges, sep='\t')
        tqdm.pandas(desc="Preparing 'sources'")
        df.insert(0, 'source', df.pop('source').progress_apply(lambda x: norm_types(x)))
        tqdm.pandas(desc="Preparing 'relations'")
        df.insert(1, 'relation', df.pop('metaedge').progress_apply(lambda x: norm_rels(x)))
        tqdm.pandas(desc="Preparing 'targets'")
        df.insert(2, 'target', df.pop('target').progress_apply(lambda x: norm_types(x)))
        tqdm.pandas(desc=None)
        df = df.sort_values(['relation', 'source', 'target'])

        save_tsv(root, filename, df)

    hex = get_digest(fullname, chunk_size)
    assert digests.get(filename) == hex, f"The graph file '{filename}' is corrupted (digest: {hex})"


def split_hetionet_graph(
        root: str, train_filename: str, valid_filename: str, test_filename: str,
        graph: str, relation: str, frac: float,
        duplicate: bool,
        digests: dict[str, str],
        seed: int = 42,
        force: bool = False,
        chunk_size: int = 8192
):
    assert 0 < frac < 0.5
    train_fullname = join(root, train_filename)
    valid_fullname = join(root, valid_filename)
    test_fullname = join(root, test_filename)
    if force or not exists(train_fullname) or not exists(valid_fullname) or not exists(test_fullname):
        df = pd.read_csv(graph, names=['source', 'relation', 'target'], sep='\t')
        is_relation = df.relation == relation
        graph = df.loc[~is_relation]
        train = df.loc[is_relation]

        valid = train.sample(frac=frac, random_state=seed)
        train = pd.merge(train, valid, indicator='pick', how='outer').query('pick=="left_only"').drop('pick', axis=1)

        test = train.sample(n=len(valid), random_state=seed)
        train = pd.merge(train, test, indicator='pick', how='outer').query('pick=="left_only"').drop('pick', axis=1)

        order = ['relation', 'source', 'target']
        train = pd.concat([graph, train]).sort_values(order)
        save_tsv(root, train_filename, train)

        if duplicate:
            valid = pd.concat([graph, valid]).sort_values(order)
        save_tsv(root, valid_filename, valid)

        if duplicate:
            test = pd.concat([graph, test]).sort_values(order)
        save_tsv(root, test_filename, test)

    hex = get_digest(train_fullname, chunk_size)
    assert digests.get(train_filename) == hex, f"The graph file '{train_filename}' is corrupted (digest: {hex})"
    hex = get_digest(valid_fullname, chunk_size)
    assert digests.get(valid_filename) == hex, f"The graph file '{valid_filename}' is corrupted (digest: {hex})"
    hex = get_digest(test_fullname, chunk_size)
    assert digests.get(test_filename) == hex, f"The graph file '{test_filename}' is corrupted (digest: {hex})"


def get_snapshots(wall_time: int) -> list[int]:
    result = [10]
    while result[-1] < wall_time:
        result.append(result[-1] * (5 if str(result[-1])[0] == '1' else 2))

    return result


def create_hetionet_config_learn(
        root: str, filename: str,
        train: str, rules: str,
        relation: str, snapshots: list[int],
):
    lines = [
        f"PATH_TRAINING    = {join(root, train)}\n",
        f"PATH_OUTPUT      = {join(root, rules)}\n",
        f"SINGLE_RELATIONS = {relation}\n",
        f"SNAPSHOTS_AT     = {','.join(str(s) for s in snapshots)}\n",
        f"WORKER_THREADS   = {min(os.cpu_count() - 1, round(0.9 * os.cpu_count()))}\n",
    ]
    fullname = join(root, filename)
    makedirs(dirname(fullname), exist_ok=True)
    with open(fullname, 'w') as file:
        file.writelines(lines)


def create_hetionet_config_apply(
        root: str, filename: str,
        train: str, valid: str, test: str,
        rules: str, preds: str, expl: str | None,
        top_k: int, snapshots: list[int],
):
    lines = [
        f"PATH_TRAINING    = {join(root, train)}\n",
        f"PATH_VALID       = {join(root, valid)}\n",
        f"PATH_TEST        = {join(root, test)}\n",
        f"PATH_RULES       = {join(root, rules)}-{snapshots[-1]}\n",
        f"PATH_OUTPUT      = {join(root, preds)}-{snapshots[-1]}\n",
    ]
    if expl is not None:
        lines += [f"PATH_EXPLANATION = {join(root, expl)}\n"]
    lines += [
        f"TOP_K_OUTPUT     = {top_k}\n",
        f"WORKER_THREADS   = {min(os.cpu_count() - 1, round(0.9 * os.cpu_count()))}\n",
    ]
    fullname = join(root, filename)
    makedirs(dirname(fullname), exist_ok=True)
    with open(fullname, 'w') as file:
        file.writelines(lines)


def create_hetionet_config_eval(
        root: str, filename: str,
        train: str, valid: str, test: str,
        preds: str, top_k: int, snapshots: list[int],
):
    lines = [
        f"PATH_TRAINING    = {join(root, train)}\n",
        f"PATH_VALID       = {join(root, valid)}\n",
        f"PATH_TEST        = {join(root, test)}\n",
        f"PATH_PREDICTIONS = {join(root, preds)}-{snapshots[-1]}\n",
        f"TOP_K            = {top_k}\n",
    ]
    fullname = join(root, filename)
    makedirs(dirname(fullname), exist_ok=True)
    with open(fullname, 'w') as file:
        file.writelines(lines)


def create_hetionet_script(
        root: str, filename: str,
        program: str, function: str, config: str
):
    assert function in ['Learn', 'Apply', 'Eval']
    lines = [
        f"#!/usr/bin/env bash\n",
        f"java -Xmx12G -cp {join(root, program)} de.unima.ki.anyburl.{function} {join(root, config)}\n",
    ]
    fullname = join(root, filename)
    makedirs(dirname(fullname), exist_ok=True)
    with open(fullname, 'w') as file:
        file.writelines(lines)

    st = os.stat(fullname)
    os.chmod(fullname, st.st_mode | stat.S_IEXEC)


def create_hetionet_script_explain(
        root: str, filename: str,
        program: str,
        explain: str, data: str,
):
    lines = [
        f"#!/usr/bin/env bash\n",
        f"java -Xmx3G -cp {join(root, program)} de.unima.ki.anyburl.Explain {join(root, explain)} {join(root, data)}\n",
    ]
    makedirs(join(root, explain), exist_ok=True)
    with open(join(root, explain, 'target.txt'), 'w') as file:
        file.writelines([
            f"c__db00290\ttreats\td__doid:4159\n",
        ])
    fullname = join(root, filename)
    makedirs(dirname(fullname), exist_ok=True)
    with open(fullname, 'w') as file:
        file.writelines(lines)

    st = os.stat(fullname)
    os.chmod(fullname, st.st_mode | stat.S_IEXEC)


def main(root: str) -> None:
    """ Main method.
    """
    snapshots = get_snapshots(WALL_TIME)

    for filename, url in URLS.items():
        download(root, filename, url, DIGESTS)

    create_hetionet_graph(root, GRAPH, METAGRAPH, EDGES, DIGESTS)

    split_hetionet_graph(root, TRAIN_ALL, VALID_ALL, TEST_ALL, GRAPH, 'treats', 1 / 30, True, DIGESTS)

    create_hetionet_config_learn(
        root, CFG_LEARN_ALL, TRAIN_ALL, RULES_ALL, 'treats', snapshots)
    create_hetionet_config_apply(
        root, CFG_APPLY_ALL, TRAIN_ALL, VALID_ALL, TEST_ALL, RULES_ALL, PREDS_ALL, None, TOP_K, snapshots)
    create_hetionet_config_apply(
        root, CFG_APPLY_EXPL_ALL, TRAIN_ALL, VALID_ALL, TEST_ALL, RULES_ALL, PREDS_ALL, EXPLAIN_ALL, TOP_K, snapshots)
    create_hetionet_config_eval(
        root, CFG_EVAL_ALL, TRAIN_ALL, VALID_ALL, TEST_ALL, PREDS_ALL, TOP_K, snapshots)

    create_hetionet_script(root, SCR_LEARN_ALL, PROGRAM_23_1, 'Learn', CFG_LEARN_ALL)
    create_hetionet_script(root, SCR_APPLY_ALL, PROGRAM_23_1, 'Apply', CFG_APPLY_ALL)
    create_hetionet_script(root, SCR_APPLY_EXPL_ALL, PROGRAM_22, 'Apply', CFG_APPLY_EXPL_ALL)
    create_hetionet_script(root, SCR_EVAL_ALL, PROGRAM_23_1, 'Eval', CFG_EVAL_ALL)
    create_hetionet_script_explain(root, SCR_EXPLAIN_ALL, PROGRAM_22, EXPLAIN_DIR_ALL, dirname(TRAIN_ALL))

    split_hetionet_graph(root, TRAIN_1ST, VALID_1ST, TEST_1ST, GRAPH, 'treats', 1 / 30, False, DIGESTS)

    create_hetionet_config_learn(
        root, CFG_LEARN_1ST, TRAIN_1ST, RULES_1ST, 'treats', snapshots)
    create_hetionet_config_apply(
        root, CFG_APPLY_1ST, TRAIN_1ST, VALID_1ST, TEST_1ST, RULES_1ST, PREDS_1ST, None, TOP_K, snapshots)
    create_hetionet_config_apply(
        root, CFG_APPLY_EXPL_1ST, TRAIN_1ST, VALID_1ST, TEST_1ST, RULES_1ST, PREDS_1ST, EXPLAIN_1ST, TOP_K, snapshots)
    create_hetionet_config_eval(
        root, CFG_EVAL_1ST, TRAIN_1ST, VALID_1ST, TEST_1ST, PREDS_1ST, TOP_K, snapshots)

    create_hetionet_script(root, SCR_LEARN_1ST, PROGRAM_23_1, 'Learn', CFG_LEARN_1ST)
    create_hetionet_script(root, SCR_APPLY_1ST, PROGRAM_23_1, 'Apply', CFG_APPLY_1ST)
    create_hetionet_script(root, SCR_APPLY_EXPL_1ST, PROGRAM_22, 'Apply', CFG_APPLY_EXPL_1ST)
    create_hetionet_script(root, SCR_EVAL_1ST, PROGRAM_23_1, 'Eval', CFG_EVAL_1ST)
    create_hetionet_script_explain(root, SCR_EXPLAIN_1ST, PROGRAM_22, EXPLAIN_DIR_1ST, dirname(TRAIN_1ST))


if __name__ == '__main__':
    main(dirname(__file__))
    print('Done.')
