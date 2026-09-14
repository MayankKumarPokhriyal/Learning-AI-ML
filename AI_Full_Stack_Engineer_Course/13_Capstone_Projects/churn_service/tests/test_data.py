import pandas as pd

from churn_service.data import clean, make_splits, parse_total_charges, split_features_target, validate_raw


def test_valid_frame_passes_with_blank_total_charges_warning(raw_frame):
    report = validate_raw(raw_frame)
    assert report.ok, report.errors
    assert any("blank" in w for w in report.warnings)


def test_clean_turns_blank_total_charges_into_zero_for_new_customers(raw_frame):
    cleaned = clean(raw_frame)
    assert cleaned["TotalCharges"].dtype == "float64"
    assert cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"].tolist() == [0.0]
    assert cleaned["TotalCharges"].notna().all()


def test_parse_total_charges_handles_text_numbers_and_none():
    parsed = parse_total_charges(pd.Series([" ", "12.5", None, "abc"]))
    assert parsed.isna().tolist() == [True, False, True, True]
    assert parsed.iloc[1] == 12.5


def test_blank_total_charges_with_positive_tenure_is_an_error(raw_frame):
    bad = raw_frame.copy()
    bad.loc[1, "TotalCharges"] = " "
    assert any("tenure > 0" in e for e in validate_raw(bad).errors)


def test_schema_errors_are_reported(raw_frame):
    assert "missing columns" in validate_raw(raw_frame.drop(columns=["Contract"])).errors[0]
    unknown = raw_frame.copy()
    unknown.loc[0, "Contract"] = "Three year"
    assert any("Contract: unexpected values" in e for e in validate_raw(unknown).errors)
    duplicated = pd.concat([raw_frame, raw_frame.head(1)], ignore_index=True)
    assert any("duplicate" in e for e in validate_raw(duplicated).errors)
    out_of_range = raw_frame.copy()
    out_of_range.loc[0, "tenure"] = -3
    assert any("tenure" in e for e in validate_raw(out_of_range).errors)


def test_internet_add_ons_must_match_internet_service(raw_frame):
    bad = raw_frame.copy()
    bad.loc[3, "StreamingTV"] = "Yes"  # customer has no internet but streams TV
    assert any("StreamingTV" in e for e in validate_raw(bad).errors)


def test_splits_are_disjoint_stratified_and_sized(raw_frame):
    rows = []
    for i in range(400):
        row = raw_frame.iloc[i % 4].copy()
        row["customerID"] = f"id-{i}"
        row["Churn"] = "Yes" if i % 4 == 0 else "No"
        rows.append(row)
    df = clean(pd.DataFrame(rows).reset_index(drop=True))
    splits = make_splits(df, seed=0)
    ids = [set(s["customerID"]) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert [len(s) for s in splits.values()] == [240, 80, 80]
    for frame in splits.values():
        assert abs(frame["Churn"].eq("Yes").mean() - 0.25) < 0.02
    X, y = split_features_target(splits["train"])
    assert "customerID" not in X.columns and "Churn" not in X.columns and set(y.unique()) == {0, 1}
