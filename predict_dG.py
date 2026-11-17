import json
import re
import sys
import os
import time
import math

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_smiles():
    db = pd.read_csv(os.path.join(DATA_DIR, 'cache_compounds_20160818.csv'),
                     index_col='compound_id')
    db_smiles = db['smiles_pH7'].to_dict()
    return db_smiles


def load_molsig_rad1():
    molecular_signature_r1 = json.load(
        open(os.path.join(DATA_DIR, 'decompose_vector_ac.json')))
    return molecular_signature_r1


def load_molsig_rad2():
    molecular_signature_r2 = json.load(
        open(os.path.join(DATA_DIR,
             'decompose_vector_ac_r2_py3_indent_modified_manual.json')))
    return molecular_signature_r2


def load_model():
    filename = os.path.join(MODEL_DIR, 'M12_model_BR.pkl')
    if not os.path.exists(filename):
        eprint("Error: model file not found at", filename)
        eprint("Run 'python model_gen.py' first.")
        sys.exit(1)
    loaded_model = joblib.load(open(filename, 'rb'))
    return loaded_model


def load_compound_cache():
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'CC'))
    from compound_cacher import CompoundCacher
    ccache = CompoundCacher()
    return ccache


def count_substructures(radius, molecule):
    m = molecule
    smi_count = dict()
    atomList = [atom for atom in m.GetAtoms()]

    for i in range(len(atomList)):
        env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, i)
        atoms = set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())

        if len(atoms) == 0:
            atoms = {i}

        smi = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms),
                                        bondsToUse=env, canonical=True)

        if smi in smi_count:
            smi_count[smi] = smi_count[smi] + 1
        else:
            smi_count[smi] = 1
    return smi_count


def decompse_novel_mets_rad1(novel_smiles, radius=1):
    decompose_vector = dict()

    for cid, smiles_pH7 in novel_smiles.items():
        mol = Chem.MolFromSmiles(smiles_pH7)
        mol = Chem.RemoveHs(mol)
        smi_count = count_substructures(radius, mol)
        decompose_vector[cid] = smi_count
    return decompose_vector


def decompse_novel_mets_rad2(novel_smiles, radius=2):
    decompose_vector = dict()

    for cid, smiles_pH7 in novel_smiles.items():
        mol = Chem.MolFromSmiles(smiles_pH7)
        mol = Chem.RemoveHs(mol)
        smi_count = count_substructures(radius, mol)
        decompose_vector[cid] = smi_count
    return decompose_vector


def parse_reaction_formula_side(s):
    if s.strip() == "null":
        return {}

    compound_bag = {}
    for member in re.split('\s+\+\s+', s):
        tokens = member.split(None, 1)
        if len(tokens) == 0:
            continue
        if len(tokens) == 1:
            amount = 1
            key = member
        else:
            amount = float(tokens[0])
            key = tokens[1]

        compound_bag[key] = compound_bag.get(key, 0) + amount

    return compound_bag


def parse_formula(formula, arrow='<=>'):
    tokens = formula.split(arrow)
    if len(tokens) < 2:
        return None
    if len(tokens) > 2:
        return None

    left = tokens[0].strip()
    right = tokens[1].strip()

    sparse_reaction = {}
    for cid, count in parse_reaction_formula_side(left).items():
        sparse_reaction[cid] = sparse_reaction.get(cid, 0) - count

    for cid, count in parse_reaction_formula_side(right).items():
        sparse_reaction[cid] = sparse_reaction.get(cid, 0) + count

    return sparse_reaction


def get_rule(rxn_dict, molsig1, molsig2, novel_decomposed1, novel_decomposed2):
    if novel_decomposed1 != None:
        for cid in novel_decomposed1:
            molsig1[cid] = novel_decomposed1[cid]
    if novel_decomposed2 != None:
        for cid in novel_decomposed2:
            molsig2[cid] = novel_decomposed2[cid]

    molsigna_df1 = pd.DataFrame.from_dict(molsig1).fillna(0)
    all_mets1 = molsigna_df1.columns.tolist()
    all_mets1.append("C00080")
    all_mets1.append("C00282")

    molsigna_df2 = pd.DataFrame.from_dict(molsig2).fillna(0)
    all_mets2 = molsigna_df2.columns.tolist()
    all_mets2.append("C00080")
    all_mets2.append("C00282")

    moieties_r1 = open(os.path.join(DATA_DIR, 'group_names_r1.txt'))
    moieties_r2 = open(os.path.join(DATA_DIR, 'group_names_r2_py3_modified_manual.txt'))
    moie_r1 = moieties_r1.read().splitlines()
    moie_r2 = moieties_r2.read().splitlines()

    molsigna_df1 = molsigna_df1.reindex(moie_r1)
    molsigna_df2 = molsigna_df2.reindex(moie_r2)

    rule_df1 = pd.DataFrame(index=molsigna_df1.index)
    rule_df2 = pd.DataFrame(index=molsigna_df2.index)

    rule_df1['change'] = 0
    for met, stoic in rxn_dict.items():
        if met == "C00080" or met == "C00282":
            continue
        if met not in molsigna_df1.columns:
            return None, None, None
        rule_df1['change'] += molsigna_df1[met] * stoic

    rule_df2['change'] = 0
    for met, stoic in rxn_dict.items():
        if met == "C00080" or met == "C00282":
            continue
        if met not in molsigna_df2.columns:
            return None, None, None
        rule_df2['change'] += molsigna_df2[met] * stoic

    rule_vec1 = rule_df1.to_numpy().T
    rule_vec2 = rule_df2.to_numpy().T

    m1, n1 = rule_vec1.shape
    m2, n2 = rule_vec2.shape

    zeros1 = np.zeros((m1, 44))
    zeros2 = np.zeros((m2, 44))
    X1 = np.concatenate((rule_vec1, zeros1), 1)
    X2 = np.concatenate((rule_vec2, zeros2), 1)

    rule_comb = np.concatenate((X1, X2), 1)

    return rule_comb, rule_df1, rule_df2


def get_ddG0(rxn_dict, pH, I, novel_mets):
    ccache = load_compound_cache()
    T = 298.15
    ddG0_forward = 0
    for compound_id, coeff in rxn_dict.items():
        if novel_mets != None and compound_id in novel_mets:
            comp = novel_mets[compound_id]
        else:
            comp = ccache.get_compound(compound_id)
        ddG0_forward += coeff * comp.transform_pH7(pH, I, T)

    return ddG0_forward


def get_dG0(rxn_dict, rid, pH, I, loaded_model, molsig_r1, molsig_r2,
            novel_decomposed_r1, novel_decomposed_r2, novel_mets):

    rule_comb, rule_df1, rule_df2 = get_rule(
        rxn_dict, molsig_r1, molsig_r2, novel_decomposed_r1, novel_decomposed_r2)

    if rule_comb is None:
        return None, None, None, None

    X = rule_comb

    ymean, ystd = loaded_model.predict(X, return_std=True)

    conf_int = (1.96 * ystd[0]) / math.sqrt(4001)

    return ymean[0] + get_ddG0(rxn_dict, pH, I, novel_mets), conf_int, rule_df1, rule_df2


def parse_novel_molecule(add_info):
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'CC'))
    from compound import Compound
    result = {}
    for cid, InChI in add_info.items():
        c = Compound.from_inchi('Test', cid, InChI)
        result[cid] = c
    return result


def parse_novel_smiles(result):
    novel_smiles = {}
    for cid, c in result.items():
        smiles = c.smiles_pH7
        novel_smiles[cid] = smiles
    return novel_smiles


def main():
    import argparse
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
    loaded_model = load_model()
    molsig_r1 = load_molsig_rad1()
    molsig_r2 = load_molsig_rad2()

    novel_mets = None
    novel_smiles = None
    novel_decomposed_r1 = None
    novel_decomposed_r2 = None
    if args.novel:
        try:
            novel_smiles = json.loads(args.novel)
            novel_mets = parse_novel_molecule(novel_smiles)
            novel_smiles = parse_novel_smiles(novel_mets)
            novel_decomposed_r1 = decompse_novel_mets_rad1(novel_smiles)
            novel_decomposed_r2 = decompse_novel_mets_rad2(novel_smiles)
        except Exception as e:
            eprint("Error processing novel compounds:", str(e))
            sys.exit(1)

    eprint(f"  Data loaded ({time.time()-t_start:.1f}s)")

    if batch_mode:
        with open(args.file) as f:
            reactions = json.load(f)

        eprint(f"Loaded {len(reactions)} reactions from {args.file}")

        if args.rxns:
            rxn_ids = [r.upper() for r in args.rxns]
            reactions = {k: v for k, v in reactions.items()
                         if k.upper() in rxn_ids}
            eprint(f"  Filtered to {len(reactions)} specified reactions")

        eprint("Building feature vectors...")
        results = {}
        X_vectors = []
        valid_ids = []

        for rid, eq in reactions.items():
            rxn_dict = parse_formula(eq)
            if rxn_dict is None:
                results[rid] = {"error": "cannot parse equation"}
                continue
            rule_comb, _, _ = get_rule(
                rxn_dict, molsig_r1, molsig_r2,
                novel_decomposed_r1, novel_decomposed_r2)
            if rule_comb is None:
                results[rid] = {"error": "missing compound in signature database"}
                continue
            X_vectors.append(rule_comb)
            valid_ids.append(rid)

        eprint(f"  {len(X_vectors)} valid vectors built")

        n_valid = len(X_vectors)
        if n_valid > 0:
            eprint("Batch predicting...")
            X_batch = np.array([v.flatten() for v in X_vectors])
            del X_vectors

            CHUNK = 200
            means = np.empty(n_valid)
            stds = np.empty(n_valid)
            for start in range(0, n_valid, CHUNK):
                end = min(start + CHUNK, n_valid)
                m, s = loaded_model.predict(X_batch[start:end], return_std=True)
                means[start:end] = m.ravel()
                stds[start:end] = s.ravel()

            for i, rid in enumerate(valid_ids):
                ci = 1.96 * stds[i] / math.sqrt(4001)
                results[rid] = {
                    "dG": round(float(means[i]), 2),
                    "std": round(float(stds[i]), 2),
                    "CI95": round(float(ci), 2),
                    "equation": reactions[rid],
                }

            eprint(f"  Predicted {n_valid} reactions")

        eprint(f"Skipped: {len(reactions) - n_valid}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            eprint(f"Saved to {args.output}")

        eprint(f"Total time: {time.time()-t_start:.1f}s")

    else:
        if args.verbose:
            eprint(f"Reaction: {args.reaction}")

        rxn_dict = parse_formula(args.reaction)
        if rxn_dict is None:
            eprint("Cannot parse reaction equation.")
            sys.exit(1)

        if args.verbose:
            eprint(f"Parsed: {rxn_dict}")

        mu, conf_int, _, _ = get_dG0(
            rxn_dict, 'R00000', args.pH, args.I, loaded_model,
            molsig_r1, molsig_r2,
            novel_decomposed_r1, novel_decomposed_r2, novel_mets)

        if mu is None:
            eprint("Cannot predict: missing compound in signature database.")
            sys.exit(1)

        eprint(f"Total time: {time.time()-t_start:.1f}s\n")

        print(f"Reaction:  {args.reaction}")
        print(f"dG°'  = {mu:>8.2f}  kJ/mol")
        print(f"95% CI= {conf_int:>8.2f}  kJ/mol")


if __name__ == '__main__':
    main()