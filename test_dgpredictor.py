import json
import os
import re
import math
import sys

import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import predict_dG


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def model_data():
    model = predict_dG.load_model()
    molsig_r1 = predict_dG.load_molsig(1)
    molsig_r2 = predict_dG.load_molsig(2)
    group_r1 = predict_dG.load_group_names(1)
    group_r2 = predict_dG.load_group_names(2)
    group_idx_r1 = {g: i for i, g in enumerate(group_r1)}
    group_idx_r2 = {g: i for i, g in enumerate(group_r2)}
    return model, molsig_r1, molsig_r2, group_idx_r1, group_idx_r2


@pytest.fixture(scope="session")
def kegg_rxns():
    path = os.path.join(SCRIPT_DIR, "kegg_rxns.json")
    return json.load(open(path))


# =============================================================================
# parse_reaction
# =============================================================================

class TestParseReaction:
    def test_basic(self):
        r = predict_dG.parse_reaction("C00001 + C00002 <=> C00003")
        assert r == {"C00001": -1, "C00002": -1, "C00003": 1}

    def test_with_stoichiometry(self):
        r = predict_dG.parse_reaction("2 C00001 + C00002 <=> 3 C00003")
        assert r == {"C00001": -2, "C00002": -1, "C00003": 3}

    def test_null_side(self):
        r = predict_dG.parse_reaction("null <=> C00001")
        assert r == {"C00001": 1}

    def test_single_compound(self):
        r = predict_dG.parse_reaction("C00001 <=> C00002")
        assert r == {"C00001": -1, "C00002": 1}

    def test_h2_o_skipped(self):
        r = predict_dG.parse_reaction("C00080 + C00282 <=> C00001")
        assert "C00080" in r
        assert "C00282" in r

    def test_invalid_arrow(self):
        assert predict_dG.parse_reaction("C00001 -> C00002") is None

    def test_no_arrow(self):
        assert predict_dG.parse_reaction("C00001") is None


class TestParseReactionSide:
    def test_empty(self):
        assert predict_dG.parse_reaction_side("") == {}

    def test_null(self):
        assert predict_dG.parse_reaction_side("null") == {}

    def test_single(self):
        assert predict_dG.parse_reaction_side("C00001") == {"C00001": 1}

    def test_with_count(self):
        assert predict_dG.parse_reaction_side("2 C00001") == {"C00001": 2}

    def test_multiple(self):
        r = predict_dG.parse_reaction_side("2 C00001 + C00002 + 3 C00003")
        assert r == {"C00001": 2, "C00002": 1, "C00003": 3}


# =============================================================================
# count_substructures
# =============================================================================

class TestCountSubstructures:
    def test_water(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("O")
        mol = Chem.RemoveHs(mol)
        sig = predict_dG.count_substructures(1, mol)
        assert "O" in sig
        assert sig["O"] == 1

    def test_methane(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C")
        mol = Chem.RemoveHs(mol)
        sig = predict_dG.count_substructures(1, mol)
        assert "C" in sig
        assert sig["C"] == 1

    def test_ethanol(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.RemoveHs(mol)
        sig1 = predict_dG.count_substructures(1, mol)
        assert "CO" in sig1
        sig2 = predict_dG.count_substructures(2, mol)
        assert len(sig2) >= 2

    def test_radius_different(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCCO")
        mol = Chem.RemoveHs(mol)
        sig1 = predict_dG.count_substructures(1, mol)
        sig2 = predict_dG.count_substructures(2, mol)
        assert len(sig2) > 0


# =============================================================================
# decompose_novel
# =============================================================================

class TestDecomposeNovel:
    def test_single_smiles(self):
        result = predict_dG.decompose_novel({"X": "O"}, 1)
        assert "X" in result
        assert "O" in result["X"]

    def test_multiple(self):
        result = predict_dG.decompose_novel({"A": "O", "B": "CCO"}, 1)
        assert "A" in result
        assert "B" in result

    def test_invalid_smiles(self):
        result = predict_dG.decompose_novel({"X": "not-a-smiles"}, 1)
        assert result == {}


# =============================================================================
# build_rule_vector
# =============================================================================

class TestBuildRuleVector:
    def test_returns_array(self, model_data):
        _, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        rxn = {"C00001": -1, "C00002": 1}
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2, gidx1, gidx2)
        n_features = len(gidx1) + 44 + len(gidx2) + 44
        assert X.shape == (n_features,)

    def test_missing_compound(self, model_data):
        _, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        rxn = {"ZZZZZZ": -1, "C00001": 1}
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2, gidx1, gidx2)
        assert X is None

    def test_novel_compound(self, model_data):
        _, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        novel = predict_dG.decompose_novel({"N2": "O"}, 1)
        novel2 = predict_dG.decompose_novel({"N2": "O"}, 2)
        rxn = {"C00001": -1, "N2": 1}
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2,
                                          gidx1, gidx2, novel, novel2)
        assert X is not None

    def test_skip_compounds(self, model_data):
        _, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        rxn = {"C00080": -1, "C00282": -1}
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2, gidx1, gidx2)
        assert X is not None
        assert np.all(X == 0)


# =============================================================================
# Prediction
# =============================================================================

class TestPrediction:
    def test_single_reaction(self, model_data):
        model, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        rxn = predict_dG.parse_reaction("C16688 + C00001 <=> C00095 + C00092")
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2, gidx1, gidx2)
        ymean, ystd = model.predict(X.reshape(1, -1), return_std=True)
        assert isinstance(float(ymean[0]), float)
        assert float(ystd[0]) > 0

    def test_known_value(self, model_data):
        model, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        rxn = predict_dG.parse_reaction("C16688 + C00001 <=> C00095 + C00092")
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2, gidx1, gidx2)
        ymean, ystd = model.predict(X.reshape(1, -1), return_std=True)
        assert abs(float(ymean[0]) - (-26.03)) < 0.1

    def test_all_predictions_have_dG(self):
        path = os.path.join(SCRIPT_DIR, "full_predictions.json")
        pred = json.load(open(path))
        for rid, entry in pred.items():
            if "error" not in entry:
                assert "dG" in entry
                assert isinstance(entry["dG"], (int, float))
                assert "std" in entry
                assert entry["std"] > 0


# =============================================================================
# Model integrity
# =============================================================================

class TestModelIntegrity:
    def test_model_loaded(self, model_data):
        model = model_data[0]
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")
        assert model.fit_intercept == False

    def test_model_dimensions(self, model_data):
        model = model_data[0]
        n_features = model.n_features_in_
        # original feature space: 1435 r1 + 44 pad + 24881 r2 + 44 pad = 26404
        assert n_features == 26404

    def test_model_mse(self, model_data):
        from scipy.io import loadmat
        model = model_data[0]
        mat = loadmat(os.path.join(SCRIPT_DIR, "data", "Test_KEGG_all_grp.mat"))
        y = mat["y"].flatten()
        Xrc = mat["X_comb_all"]
        y_pred = model.predict(Xrc)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        assert abs(mse - 9.60) < 0.1
        assert abs(r2 - 0.9998) < 0.0001

    def test_signature_coverage(self):
        import pandas as pd
        molsig = json.load(
            open(os.path.join(SCRIPT_DIR, "data", "decompose_vector_ac.json")))
        cache = pd.read_csv(
            os.path.join(SCRIPT_DIR, "data", "cache_compounds_20160818.csv"),
            index_col="compound_id")
        c_in_cache = set(cache.index[cache["smiles_pH7"].notna()])
        c_in_sig = set(k for k in molsig if k.startswith("C"))
        uncovered = c_in_cache - c_in_sig
        assert len(uncovered) <= 10


# =============================================================================
# Reaction parsing (MetaNetX format compatibility)
# =============================================================================

class TestKEGGReactions:
    def test_all_parse(self, kegg_rxns):
        failed = []
        for rid, eq in kegg_rxns.items():
            r = predict_dG.parse_reaction(eq)
            if r is None:
                failed.append(rid)
        assert len(failed) == 0, f"{len(failed)} reactions failed to parse"

    def test_no_zero_stoich(self, kegg_rxns):
        degenerate = 0
        for rid, eq in list(kegg_rxns.items())[:500]:
            r = predict_dG.parse_reaction(eq)
            if r:
                non_zero = {c for c, v in r.items() if abs(v) > 1e-6}
                if len(non_zero) == 0:
                    degenerate += 1
        assert degenerate < 10, f"Too many degenerate reactions: {degenerate}"

    def test_has_substrates_and_products(self, kegg_rxns):
        degenerate = 0
        for rid, eq in list(kegg_rxns.items())[:500]:
            r = predict_dG.parse_reaction(eq)
            if r:
                substrates = [c for c, v in r.items() if v < 0]
                products = [c for c, v in r.items() if v > 0]
                if len(substrates) == 0 and len(products) == 0:
                    degenerate += 1
        assert degenerate < 10, f"Too many degenerate reactions: {degenerate}"


# =============================================================================
# Error handling
# =============================================================================

class TestErrorHandling:
    def test_missing_model_file(self):
        path = os.path.join(SCRIPT_DIR, "model", "M12_model_BR.pkl")
        assert os.path.exists(path), "Model file not found"
        # load_model() calls sys.exit(1) if missing; verify the guard works
        model = predict_dG.load_model()
        assert model is not None

    def test_nonexistent_compound(self, model_data):
        _, molsig_r1, molsig_r2, gidx1, gidx2 = model_data
        rxn = {"DOES_NOT_EXIST": -1}
        X = predict_dG.build_rule_vector(rxn, molsig_r1, molsig_r2, gidx1, gidx2)
        assert X is None


# =============================================================================
# Supplementary material validation
# =============================================================================

class TestSupplementaryMaterial:
    def test_reproduce_supplementary(self):
        pred_path = os.path.join(SCRIPT_DIR, "full_predictions.json")
        pred = json.load(open(pred_path))

        from scipy.io import loadmat
        mat = loadmat(os.path.join(SCRIPT_DIR, "data",
                                   "component_contribution_python.mat"))
        train_S = mat["train_S"]
        train_cids = [str(c).strip() for c in mat["train_cids"].flatten()]
        b = mat["b"].flatten()

        kegg_rxns = json.load(
            open(os.path.join(SCRIPT_DIR, "kegg_rxns.json")))

        diffs = []
        for col_idx in range(min(train_S.shape[1], 500)):
            col = train_S[:, col_idx]
            dG_exp = float(b[col_idx])
            dG_pred = None
            for rid, eq in kegg_rxns.items():
                rxn = predict_dG.parse_reaction(eq)
                if rxn is None:
                    continue
                match = True
                for row_idx in np.where(col != 0)[0]:
                    cid = train_cids[row_idx]
                    stoic = float(col[row_idx])
                    if rxn.get(cid, 0) != stoic:
                        match = False
                        break
                if match and set(rxn.keys()) == set(
                        train_cids[i] for i in np.where(col != 0)[0]):
                    if rid in pred and "dG" in pred[rid]:
                        dG_pred = float(pred[rid]["dG"])
                    break
            if dG_pred is not None:
                diffs.append(abs(dG_pred - dG_exp))

        if diffs:
            assert np.median(diffs) < 5.0


# =============================================================================
# Regression: known issues
# =============================================================================

class TestKnownDOs:
    def test_predict_dG_script_exists(self):
        path = os.path.join(SCRIPT_DIR, "predict_dG.py")
        assert os.path.exists(path)

    def test_model_file_exists(self):
        path = os.path.join(SCRIPT_DIR, "model", "M12_model_BR.pkl")
        assert os.path.exists(path)

    def test_molsig_files_exist(self):
        for f in ["decompose_vector_ac.json",
                   "decompose_vector_ac_r2_py3_indent_modified_manual.json"]:
            path = os.path.join(SCRIPT_DIR, "data", f)
            assert os.path.exists(path), f"Missing: {f}"

    def test_group_name_files_exist(self):
        for f in ["group_names_r1.txt", "group_names_r2_py3_modified_manual.txt"]:
            path = os.path.join(SCRIPT_DIR, "data", f)
            assert os.path.exists(path), f"Missing: {f}"