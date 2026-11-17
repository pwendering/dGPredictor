#!/usr/bin/env python3
"""
dGPredictor — Predict Gibbs free energy of a reaction from the command line.

Usage:
    # Single reaction
    python predict_dG.py "C16688 + C00001 <=> C00095 + C00092"

    # Single reaction with novel metabolite
    python predict_dG.py "C01745 + C00004 <=> N00001 + C00003 + C00001" \
        --novel '{"N00001": "Oc1cccc(c1)/C=C/c2ccccc2"}'

    # Batch from JSON reaction file
    python predict_dG.py --file kegg_rxns.json -o predictions.json

    # Filter to specific reactions
    python predict_dG.py --file kegg_rxns.json -o predictions.json \
        --rxns R00253 R00256 R00010
"""

import argparse
import json
import re
import sys
import os
import time

import numpy as np
import joblib
from rdkit import Chem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

SKIP_COMPOUNDS = {'C00080', 'C00282'}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_model():
    path = os.path.join(MODEL_DIR, 'M12_model_BR.pkl')
    if not os.path.exists(path):
        eprint(f"Error: model file not found at {path}")
        eprint("Run 'python model_gen.py' first (takes ~8 minutes).")
        sys.exit(1)
    return joblib.load(open(path, 'rb'))


def load_molsig(radius):
    files = {1: 'decompose_vector_ac.json',
             2: 'decompose_vector_ac_r2_py3_indent_modified_manual.json'}
    path = os.path.join(DATA_DIR, files[radius])
    t0 = time.time()
    data = json.load(open(path))
    eprint(f"  Loaded radius-{radius} signatures: {len(data)} compounds "
           f"({time.time()-t0:.1f}s)")
    return data


def load_group_names(radius):
    files = {1: 'group_names_r1.txt',
             2: 'group_names_r2_py3_modified_manual.txt'}
    path = os.path.join(DATA_DIR, files[radius])
    names = open(path).read().splitlines()
    eprint(f"  Loaded radius-{radius} group names: {len(names)}")
    return names


def count_substructures(radius, molecule):
    m = molecule
    smi_count = {}
    for i in range(m.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, i)
        atoms = set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())
        if len(atoms) == 0:
            atoms = {i}
        smi = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms),
                                        bondsToUse=env, canonical=True)
        smi_count[smi] = smi_count.get(smi, 0) + 1
    return smi_count


def decompose_novel(smiles_dict, radius):
    result = {}
    for cid, smiles in smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            eprint(f"  Warning: could not parse SMILES for {cid}")
            continue
        mol = Chem.RemoveHs(mol)
        result[cid] = count_substructures(radius, mol)
    return result


def parse_reaction_side(s):
    if s.strip() == "null":
        return {}
    bag = {}
    for member in re.split(r'\s+\+\s+', s.strip()):
        tokens = member.split(None, 1)
        if not tokens:
            continue
        if len(tokens) == 1:
            amount, key = 1.0, member
        else:
            amount, key = float(tokens[0]), tokens[1]
        bag[key] = bag.get(key, 0) + amount
    return bag


def parse_reaction(formula, arrow='<=>'):
    tokens = formula.split(arrow)
    if len(tokens) != 2:
        eprint(f"Error: reaction must contain exactly one '{arrow}'")
        return None
    rxn = {}
    for cid, cnt in parse_reaction_side(tokens[0].strip()).items():
        rxn[cid] = rxn.get(cid, 0) - cnt
    for cid, cnt in parse_reaction_side(tokens[1].strip()).items():
        rxn[cid] = rxn.get(cid, 0) + cnt
    return rxn


def build_rule_vector(rxn_dict, molsig_r1, molsig_r2,
                      group_idx_r1, group_idx_r2,
                      novel_r1=None, novel_r2=None):
    if novel_r1:
        molsig_r1 = {**molsig_r1, **novel_r1}
    if novel_r2:
        molsig_r2 = {**molsig_r2, **novel_r2}

    v1 = np.zeros(len(group_idx_r1), dtype=float)
    v2 = np.zeros(len(group_idx_r2), dtype=float)

    for met, stoic in rxn_dict.items():
        if met in SKIP_COMPOUNDS:
            continue

        sig1 = molsig_r1.get(met)
        sig2 = molsig_r2.get(met)
        if sig1 is None or sig2 is None:
            return None

        for group, count in sig1.items():
            idx = group_idx_r1.get(group)
            if idx is not None:
                v1[idx] += count * stoic
        for group, count in sig2.items():
            idx = group_idx_r2.get(group)
            if idx is not None:
                v2[idx] += count * stoic

    return np.concatenate([v1, np.zeros(44), v2, np.zeros(44)])


def main():
    parser = argparse.ArgumentParser(
        description='Predict Gibbs free energy of a reaction')
    parser.add_argument('reaction', type=str, nargs='?', default=None,
                        help='Reaction equation using KEGG IDs')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='JSON file with {rxn_id: equation} dict')
    parser.add_argument('--rxns', type=str, nargs='*', default=None,
                        help='Only predict these reaction IDs (from --file)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for batch predictions')
    parser.add_argument('--novel', type=str, default=None,
                        help='JSON dict of {id: SMILES} for novel metabolites')
    parser.add_argument('--pH', type=float, default=7.0)
    parser.add_argument('--I', type=float, default=0.25,
                        help='Ionic strength [M]')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.reaction is None and args.file is None:
        eprint("Provide a reaction or use --file for batch mode.")
        parser.print_help()
        sys.exit(1)

    batch_mode = args.file is not None

    eprint("dGPredictor")
    eprint("=" * 50)
    t_start = time.time()

    eprint("Loading data...")
    model = load_model()
    molsig_r1 = load_molsig(1)
    molsig_r2 = load_molsig(2)
    group_r1 = load_group_names(1)
    group_r2 = load_group_names(2)
    group_idx_r1 = {g: i for i, g in enumerate(group_r1)}
    group_idx_r2 = {g: i for i, g in enumerate(group_r2)}
    eprint(f"  Data loaded ({time.time()-t_start:.1f}s)")

    novel_r1 = None
    novel_r2 = None
    if args.novel:
        novel_smiles = json.loads(args.novel)
        t0 = time.time()
        novel_r1 = decompose_novel(novel_smiles, 1)
        novel_r2 = decompose_novel(novel_smiles, 2)
        eprint(f"  Decomposed {len(novel_smiles)} novel compound(s) "
               f"({time.time()-t0:.1f}s)")

    if batch_mode:
        with open(args.file) as f:
            reactions = json.load(f)

        eprint(f"Loaded {len(reactions)} reactions from {args.file}")

        if args.rxns:
            rxn_ids = [r.upper() for r in args.rxns]
            reactions = {k: v for k, v in reactions.items()
                         if k.upper() in rxn_ids}
            eprint(f"  Filtered to {len(reactions)} specified reactions")

        # Phase 1: Build all feature vectors
        eprint("Phase 1: Building feature vectors...")
        t_phase = time.time()
        results = {}
        X_vectors = []
        valid_ids = []

        for rid, eq in reactions.items():
            rxn_dict = parse_reaction(eq)
            if rxn_dict is None:
                results[rid] = {"error": "cannot parse equation"}
                continue
            X = build_rule_vector(rxn_dict, molsig_r1, molsig_r2,
                                  group_idx_r1, group_idx_r2,
                                  novel_r1, novel_r2)
            if X is None:
                results[rid] = {"error": "missing compound in signature database"}
                continue
            X_vectors.append(X)
            valid_ids.append(rid)

        eprint(f"  {len(X_vectors)} valid vectors built "
               f"({time.time()-t_phase:.1f}s)")

        # Phase 2: Batch predict
        n_valid = len(X_vectors)
        if n_valid > 0:
            eprint("Phase 2: Batch predicting...")
            t_phase = time.time()
            X_batch = np.array(X_vectors)
            del X_vectors

            CHUNK = 200
            means = np.empty(n_valid)
            stds = np.empty(n_valid)
            for start in range(0, n_valid, CHUNK):
                end = min(start + CHUNK, n_valid)
                m, s = model.predict(X_batch[start:end], return_std=True)
                means[start:end] = m.ravel()
                stds[start:end] = s.ravel()

            for i, rid in enumerate(valid_ids):
                ci = 1.96 * stds[i] / np.sqrt(4001)
                results[rid] = {
                    "dG": round(float(means[i]), 2),
                    "std": round(float(stds[i]), 2),
                    "CI95": round(float(ci), 2),
                    "equation": reactions[rid],
                }

            eprint(f"  Predicted {n_valid} reactions in "
                   f"{time.time()-t_phase:.1f}s")

        eprint(f"Skipped: {len(reactions) - n_valid}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            eprint(f"Saved to {args.output}")

        eprint(f"\nTotal time: {time.time()-t_start:.1f}s")

    else:
        if args.verbose:
            eprint(f"Reaction: {args.reaction}")
            eprint(f"pH = {args.pH}, I = {args.I} M")

        rxn_dict = parse_reaction(args.reaction)
        if rxn_dict is None:
            eprint("Cannot parse reaction equation.")
            sys.exit(1)

        if args.verbose:
            eprint(f"Parsed: {rxn_dict}")

        X = build_rule_vector(rxn_dict, molsig_r1, molsig_r2,
                              group_idx_r1, group_idx_r2,
                              novel_r1, novel_r2)
        if X is None:
            eprint("Cannot predict: one or more compounds missing from database.")
            sys.exit(1)

        ymean, ystd = model.predict(X.reshape(1, -1), return_std=True)
        ci = 1.96 * ystd[0] / np.sqrt(4001)

        eprint(f"\nTotal time: {time.time()-t_start:.1f}s\n")

        print(f"Reaction:  {args.reaction}")
        print(f"dG°'  = {ymean[0]:>8.2f}  kJ/mol")
        print(f"σ     = {ystd[0]:>8.2f}  kJ/mol")
        print(f"95% CI= {ci:>8.2f}  kJ/mol")


if __name__ == '__main__':
    main()