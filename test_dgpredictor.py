import json
import os
import sys

import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import predict_dG


@pytest.fixture(scope="session")
def signatures():
    r1 = predict_dG.load_molsig_rad1()
    r2 = predict_dG.load_molsig_rad2()
    return r1, r2


@pytest.fixture(scope="session")
def model_and_sigs(signatures):
    model = predict_dG.load_model()
    r1, r2 = signatures
    return model, r1, r2


@pytest.fixture(scope="session")
def kegg_rxns():
    return json.load(open(os.path.join(SCRIPT_DIR, "data", "kegg", "kegg_rxns.json")))


# =============================================================================
# Parse reaction (original: parse_formula / parse_reaction_formula_side)
# =============================================================================

class TestParseFormula:
    def test_basic(self):
        r = predict_dG.parse_formula("C00001 + C00002 <=> C00003")
        assert r == {"C00001": -1, "C00002": -1, "C00003": 1}

    def test_with_stoichiometry(self):
        r = predict_dG.parse_formula("2 C00001 + C00002 <=> 3 C00003")
        assert r == {"C00001": -2, "C00002": -1, "C00003": 3}

    def test_null_side(self):
        r = predict_dG.parse_formula("null <=> C00001")
        assert r == {"C00001": 1}

    def test_single_compound(self):
        r = predict_dG.parse_formula("C00001 <=> C00002")
        assert r == {"C00001": -1, "C00002": 1}

    def test_keeps_h2o(self):
        r = predict_dG.parse_formula("C00080 + C00282 <=> C00001")
        assert "C00080" in r
        assert "C00282" in r

    def test_invalid_arrow(self):
        assert predict_dG.parse_formula("C00001 -> C00002") is None

    def test_no_arrow(self):
        assert predict_dG.parse_formula("C00001") is None


class TestParseReactionFormulaSide:
    def test_empty(self):
        assert predict_dG.parse_reaction_formula_side("") == {}

    def test_null(self):
        assert predict_dG.parse_reaction_formula_side("null") == {}

    def test_single(self):
        assert predict_dG.parse_reaction_formula_side("C00001") == {"C00001": 1}

    def test_with_count(self):
        assert predict_dG.parse_reaction_formula_side("2 C00001") == {"C00001": 2}

    def test_multiple(self):
        r = predict_dG.parse_reaction_formula_side("2 C00001 + C00002 + 3 C00003")
        assert r == {"C00001": 2, "C00002": 1, "C00003": 3}


# =============================================================================
# count_substructures (exact copy from original)
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
# get_rule (exact copy from original, returns (rule_comb, df1, df2))
# =============================================================================

class TestGetRule:
    def test_returns_tuple(self, signatures):
        r1, r2 = signatures
        rxn = {"C00001": -1, "C00002": 1}
        rule_comb, df1, df2 = predict_dG.get_rule(rxn, r1, r2, None, None)
        assert rule_comb is not None
        assert df1 is not None
        assert df2 is not None

    def test_missing_compound(self, signatures):
        r1, r2 = signatures
        rxn = {"ZZZZZZ": -1, "C00001": 1}
        rule_comb, df1, df2 = predict_dG.get_rule(rxn, r1, r2, None, None)
        assert rule_comb is None

    def test_skip_compounds(self, signatures):
        r1, r2 = signatures
        rxn = {"C00080": -1, "C00282": -1}
        rule_comb, df1, df2 = predict_dG.get_rule(rxn, r1, r2, None, None)
        assert rule_comb is not None


# =============================================================================
# Prediction via get_dG0 (original prediction entry point)
# =============================================================================

class TestPrediction:
    def test_known_value(self, model_and_sigs):
        model, r1, r2 = model_and_sigs
        rxn = predict_dG.parse_formula("C16688 + C00001 <=> C00095 + C00092")
        mu, ci, _, _ = predict_dG.get_dG0(
            rxn, 'test', 7.0, 0.25, model, r1, r2, None, None, None)
        assert mu is not None
        assert abs(mu - (-26.03)) < 0.1

    def test_single_reaction(self, model_and_sigs):
        model, r1, r2 = model_and_sigs
        rxn = predict_dG.parse_formula("C16688 + C00001 <=> C00095 + C00092")
        mu, ci, _, _ = predict_dG.get_dG0(
            rxn, 'test', 7.0, 0.25, model, r1, r2, None, None, None)
        assert isinstance(mu, float)
        assert isinstance(ci, float)

    def test_all_predictions_have_dG(self):
        path = os.path.join(SCRIPT_DIR, "predictions", "kegg", "full_predictions.json")
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
    def test_model_loaded(self, model_and_sigs):
        model = model_and_sigs[0]
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")
        assert model.fit_intercept == False

    def test_model_dimensions(self, model_and_sigs):
        model = model_and_sigs[0]
        assert model.n_features_in_ == 26404

    def test_model_mse(self, model_and_sigs):
        from scipy.io import loadmat
        from sklearn.metrics import mean_squared_error, r2_score
        model = model_and_sigs[0]
        mat = loadmat(os.path.join(SCRIPT_DIR, "data", "Test_KEGG_all_grp.mat"))
        y = mat["y"].flatten()
        Xrc = mat["X_comb_all"]
        y_pred = model.predict(Xrc)
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
# KEGG reactions parsing
# =============================================================================

class TestKEGGReactions:
    def test_all_parse(self, kegg_rxns):
        failed = []
        for rid, eq in kegg_rxns.items():
            if predict_dG.parse_formula(eq) is None:
                failed.append(rid)
        assert len(failed) == 0, f"{len(failed)} reactions failed to parse"

    def test_no_zero_stoich(self, kegg_rxns):
        degenerate = 0
        for rid, eq in list(kegg_rxns.items())[:500]:
            r = predict_dG.parse_formula(eq)
            if r:
                non_zero = {c for c, v in r.items() if abs(v) > 1e-6}
                if len(non_zero) == 0:
                    degenerate += 1
        assert degenerate < 10

    def test_has_substrates_and_products(self, kegg_rxns):
        degenerate = 0
        for rid, eq in list(kegg_rxns.items())[:500]:
            r = predict_dG.parse_formula(eq)
            if r:
                subs = [c for c, v in r.items() if v < 0]
                pros = [c for c, v in r.items() if v > 0]
                if len(subs) == 0 and len(pros) == 0:
                    degenerate += 1
        assert degenerate < 10


# =============================================================================
# Error handling
# =============================================================================

class TestErrorHandling:
    def test_missing_model_file(self):
        path = os.path.join(SCRIPT_DIR, "model", "M12_model_BR.pkl")
        assert os.path.exists(path)
        model = predict_dG.load_model()
        assert model is not None

    def test_nonexistent_compound(self, signatures):
        r1, r2 = signatures
        rxn = {"DOES_NOT_EXIST": -1}
        rc, _, _ = predict_dG.get_rule(rxn, r1, r2, None, None)
        assert rc is None


# =============================================================================
# Prediction consistency against training data
# =============================================================================

class TestPredictionConsistency:
    def test_reproduce_training(self):
        from scipy.io import loadmat
        pred_path = os.path.join(SCRIPT_DIR, "predictions", "kegg", "full_predictions.json")
        pred = json.load(open(pred_path))

        mat = loadmat(os.path.join(SCRIPT_DIR, "data",
                                   "component_contribution_python.mat"))
        train_S = mat["train_S"]
        train_cids = [str(c).strip() for c in mat["train_cids"].flatten()]
        b = mat["b"].flatten()

        kegg_rxns = json.load(
            open(os.path.join(SCRIPT_DIR, "data", "kegg", "kegg_rxns.json")))

        diffs = []
        for col_idx in range(min(train_S.shape[1], 500)):
            col = train_S[:, col_idx]
            dG_exp = float(b[col_idx])
            dG_pred = None
            for rid, eq in kegg_rxns.items():
                rxn = predict_dG.parse_formula(eq)
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
# File existence checks
# =============================================================================

class TestFilesExist:
    def test_predict_dG_script(self):
        assert os.path.exists(os.path.join(SCRIPT_DIR, "predict_dG.py"))

    def test_model_file(self):
        assert os.path.exists(os.path.join(SCRIPT_DIR, "model", "M12_model_BR.pkl"))

    def test_molsig_files(self):
        for f in ["decompose_vector_ac.json",
                   "decompose_vector_ac_r2_py3_indent_modified_manual.json"]:
            assert os.path.exists(os.path.join(SCRIPT_DIR, "data", f))

    def test_group_name_files(self):
        for f in ["group_names_r1.txt", "group_names_r2_py3_modified_manual.txt"]:
            assert os.path.exists(os.path.join(SCRIPT_DIR, "data", f))