import pandas as pd
import numpy as np
from data_workbench.cleaning import DataCleaner


def test_swap_dates_are_fixed():
    df = pd.DataFrame({"Order Date": ["20/01/2020"], "Ship Date": ["19/01/2020"]})

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    cleaner = DataCleaner()
    out = cleaner.fix_order_ship_dates(df)

    assert out.iloc[0]["Time Till Shipping"] == pd.Timedelta(days=1)


def test_reformat_order_date_fixed():

    df = pd.DataFrame({"Order Date": ["01/11/2020"], "Ship Date": ["12/01/2020"]})

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    cleaner = DataCleaner()
    out = cleaner.fix_order_ship_dates(df)

    assert out.iloc[0]["Time Till Shipping"] == pd.Timedelta(days=1)

    pass


def test_reformat_ship_date_fixed():
    df = pd.DataFrame({"Order Date": ["01/01/2020"], "Ship Date": ["01/02/2020"]})

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    cleaner = DataCleaner()
    out = cleaner.fix_order_ship_dates(df)

    assert out.iloc[0]["Time Till Shipping"] == pd.Timedelta(days=1)


def test_reformat_dates_fixed():
    df = pd.DataFrame({"Order Date": ["12/01/2020"], "Ship Date": ["12/04/2020"]})

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    cleaner = DataCleaner()
    out = cleaner.fix_order_ship_dates(df)

    assert out.iloc[0]["Time Till Shipping"] == pd.Timedelta(days=3)

    pass


def test_no_date_change_required():
    df = pd.DataFrame({"Order Date": ["01/01/2020"], "Ship Date": ["15/01/2020"]})

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    cleaner = DataCleaner()
    out = cleaner.fix_order_ship_dates(df)

    assert out.iloc[0]["Time Till Shipping"] == pd.Timedelta(days=14)

    pass


def test_dataframe_summary():
    df = pd.DataFrame(
        {
            "Order ID": [
                "CA-2017-152156",
                "CA-2017-152157",
                "CA-2017-152158",
                "",
                None,
            ],
            "Quantity": [5, 2, None, 7, np.nan],
            "Price": [19.99, np.nan, 15.49, 42.00, 7.95],
            "Order Date": [
                "18/11/2017",
                pd.NaT,
                "20/11/2017",
                "",
                "   ",
            ],
            "Ship Date": [
                "20/11/2017",
                "22/11/2017",
                None,
                pd.NaT,
                "25/11/2017",
            ],
            "Customer Name": [
                "John Smith",
                "Jane Doe",
                "Alice Jones",
                "",
                "Bob Martin",
            ],
            "Comments": [None, "Delivered early", "   ", "", "Ok"],
            "Category": pd.Series(
                ["Furniture", "", "Office Supplies", None, "Technology"],
                dtype="category",
            ),
        }
    )
    df.insert(0, "Row ID", df.index + 1)

    expected = {
        "Order ID": [4, 5],
        "Quantity": [3, 5],
        "Price": [2],
        "Order Date": [2, 4, 5],
        "Ship Date": [3, 4],
        "Customer Name": [4],
        "Comments": [1, 3, 4],
        "Category": [2, 4],
    }
    cleaner = DataCleaner()
    assert expected == cleaner.summarise_missing(df)


def test_all_orders_consistent_returns_true_and_prints_message(capsys):
    cleaner = DataCleaner()

    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "B"],
            "Customer": ["Alice", "Alice", "Bob", "Bob"],
            "City": ["London", "London", "Hull", "Hull"],
        }
    )

    result = cleaner.check_order_consistency(
        df,
        key_col="Order ID",
        check_cols=["Customer", "City"],
    )

    assert result is True

    captured = capsys.readouterr()
    assert "All orders are consistent across the specified columns." in captured.out


def test_single_inconsistent_column_detected_and_returned(capsys):
    cleaner = DataCleaner()

    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "B"],
            "Customer": ["Alice", "Alice", "Bob", "Bob"],
            "City": ["London", "Paris", "Hull", "Hull"],  # A is inconsistent
        }
    )

    result = cleaner.check_order_consistency(
        df,
        key_col="Order ID",
        check_cols=["Customer", "City"],
    )

    assert result == {"A"}

    captured = capsys.readouterr()
    assert "Found 1 inconsistent orders:" in captured.out
    assert "Order A — inconsistent in: City" in captured.out


def test_multiple_inconsistent_columns_for_same_order(capsys):
    cleaner = DataCleaner()

    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "B"],
            "Customer": ["Alice", "Alicia", "Bob", "Bob"],  # inconsistent
            "City": ["London", "Paris", "Hull", "Hull"],  # inconsistent
        }
    )

    result = cleaner.check_order_consistency(
        df,
        key_col="Order ID",
        check_cols=["Customer", "City"],
    )

    assert result == {"A"}

    captured = capsys.readouterr()

    # Find the line that contains the inconsistent column list
    line = next(
        l for l in captured.out.splitlines() if "Order A — inconsistent in:" in l
    )

    assert "Customer" in line
    assert "City" in line


def test_multiple_inconsistent_orders_are_all_returned(capsys):
    cleaner = DataCleaner()

    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "B", "C"],
            "Customer": [
                "Alice",
                "Alice",
                "Bob",
                "Robert",
                "Charlie",
            ],  # B inconsistent
            "City": ["London", "Paris", "Hull", "Hull", "Leeds"],  # A inconsistent
        }
    )

    result = cleaner.check_order_consistency(
        df,
        key_col="Order ID",
        check_cols=["Customer", "City"],
    )

    assert result == {"A", "B"}

    captured = capsys.readouterr()
    assert "Found 2 inconsistent orders:" in captured.out
    assert "Order A — inconsistent in:" in captured.out
    assert "Order B — inconsistent in:" in captured.out


def test_nan_values_are_ignored_for_consistency_check():
    cleaner = DataCleaner()

    df = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B", "B"],
            "Customer": ["Alice", "Alice", "Bob", "Bob"],
            "City": ["London", pd.NA, "Hull", "York"],  # A consistent, B inconsistent
        }
    )

    result = cleaner.check_order_consistency(
        df,
        key_col="Order ID",
        check_cols=["Customer", "City"],
    )

    assert result == {"B"}
