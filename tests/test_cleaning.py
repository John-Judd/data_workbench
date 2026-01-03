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
