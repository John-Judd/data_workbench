import pandas as pd
from enum import Enum
import numpy as np


class DataCleaner:

    def __init__(self):
        self.order = "Order Date"
        self.ship = "Ship Date"
        self.time_col_name = "Time Till Shipping"
        self.day_threshold = 15

    def fix_order_ship_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Will attempt to fix erroneous shipping and order times.
        These can be input incorrectly by a used, using the day month
        in the wrong order and putting ship date in order and order date
        in ship. This function converts the values and checks if the converted
        values fall in the threshold. Of course it's possible an item could
        take 200 days to deliver, but if it is possible to convert the
        dates to give a valid lower date then it is taken. The function
        take a positive approach and identifies these outliers as errors
        in the data rather than accepting them or ignoring them.

        Args:
            df (pd.DataFrame): A dataframe with order and ship dates

        Returns:
            pd.DataFrame: the data frame where the ship and order dates have
            been augmented to fit in the thrshold if there is a possibility
            they have been entered erroneously.
        """

        # Calculate the time between order and shipping
        df["Time Till Shipping"] = df[self.ship] - df[self.order]

        # Now we only want to deal with rows with issues
        # NEgative numbers and ones with days over the threshold
        fix_mask = (
            (df[self.time_col_name] < pd.Timedelta(0))
            | (df[self.time_col_name] > pd.Timedelta(days=self.day_threshold))
        ) & df[self.time_col_name].notna()

        df.loc[fix_mask] = df.loc[fix_mask].apply(self._best_time_fix, axis=1)

        return df

    def _best_time_fix(self, row):
        """Row logic used in fix_order_ship_dates

        Args:
            row: row which possibly has erroneous data for order and ship date

        Returns:
            returns the altered row with fixed order and ship date
        """

        row = row.copy()

        class DateFixType(Enum):
            NONE = 0
            BOTH = 1
            SHIP = 2
            ORDER = 3
            SWAP = 4

        order_date = row[self.order]
        ship_date = row[self.ship]

        # Covert the dates to different formats
        order_reformat_valid = True
        try:
            reformated_order_date = self._swap_day_month(order_date)
        except ValueError:
            order_reformat_valid = False

        ship_reformat_valid = True
        try:
            reformated_ship_date = self._swap_day_month(ship_date)
        except ValueError:
            ship_reformat_valid = False

        # Preset with negative values so they are not picked
        both_reformat_time = ship_reformat_time = order_reformat_time = pd.Timedelta(
            days=-1
        )

        # No changes
        no_change_time = ship_date - order_date

        # Swap the dates
        swap_time = order_date - ship_date

        # Order reformat
        if order_reformat_valid:
            order_reformat_time = ship_date - reformated_order_date

        # Ship reformat
        if ship_reformat_valid:
            ship_reformat_time = reformated_ship_date - order_date

        # Both reformat
        if order_reformat_valid and ship_reformat_valid:
            both_reformat_time = reformated_ship_date - reformated_order_date

        # Set this to a very high value so it overwritten
        best_pick = pd.Timedelta(days=99999)

        zero_days = pd.Timedelta(days=0)

        # Now we pick the lowest which is not negative
        if no_change_time >= zero_days and no_change_time < best_pick:
            best_pick = no_change_time
            fix = DateFixType.NONE

        if swap_time >= zero_days and swap_time < best_pick and best_pick:
            best_pick = swap_time
            fix = DateFixType.SWAP

        if order_reformat_time >= zero_days and order_reformat_time < best_pick:
            best_pick = order_reformat_time
            fix = DateFixType.ORDER

        if ship_reformat_time >= zero_days and ship_reformat_time < best_pick:
            best_pick = ship_reformat_time
            fix = DateFixType.SHIP

        if both_reformat_time >= zero_days and both_reformat_time < best_pick:
            best_pick = both_reformat_time
            fix = DateFixType.BOTH

        match fix:
            case DateFixType.SWAP:
                row[self.order], row[self.ship] = row[self.ship], row[self.order]
            case DateFixType.ORDER:
                row[self.order] = reformated_order_date
            case DateFixType.SHIP:
                row[self.ship] = reformated_ship_date
            case DateFixType.BOTH:
                row[self.order] = reformated_order_date
                row[self.ship] = reformated_ship_date
            case _:
                print("Unknown state")

        row[self.time_col_name] = row[self.ship] - row[self.order]

        return row

    def _swap_day_month(self, date):
        """Swaps the day and month, not is the day is over 12 it
        is not a valid month and calling this will trigger
        the exception ValueError

        Args:
            date: date to attempt to swap day and month on

        Returns:
            Row with day and month flipped if possible
        """
        return pd.Timestamp(year=date.year, month=date.day, day=date.month)

    def normalise_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """This will switch all blank space, nan, nat any data
        which represents missing data to numpy nan.

        Args:
            df (pd.DataFrame): Dataframe to normalise missing data on

        Returns:
            pd.DataFrame: normalised dataframe
        """
        df = df.copy()
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df = df.where(pd.notna(df), np.nan)

        return df

    def summarise_missing(self, df: pd.DataFrame, id_col: str = "Row ID") -> dict:
        """Handy tool to find all the missing data in a dataframe providing a
        list of the rows for each instance of a missing cell in a column.

        Args:
            df (pd.DataFrame): _description_
            id_col (str, optional): _description_. Defaults to "Row ID".

        Raises:
            KeyError: If the identifier col is not in the dataframe

        Returns:
            dict: A dictionary with the key being the missing column name
            with points to a list of the Row ID which identifies them
        """

        if id_col not in df.columns:
            raise KeyError(f"ID column '{id_col}' not found in DataFrame")

        # Convert blank or whitespace into NaN
        df_clean = self.normalise_missing(df)

        result = {}

        for col in df_clean.columns:
            if col == id_col:
                continue  # Don't process the ID column itself

            # Boolean mask of missing values
            missing_mask = df_clean[col].isna()

            if missing_mask.any():
                # Extract row IDs for missing rows
                missing_ids = df_clean.loc[missing_mask, id_col].tolist()
                result[col] = missing_ids

        return result

    def check_order_consistency(
        self, df: pd.DataFrame, key_col: str, check_cols: list[str]
    ) -> set:
        """Rows which are in the same order, much of the data can be
        repeated, this allows us to fill missing data from other rows in the
        same order. Before this can be done we need to be sure the data is
        consistent across these same order rows as there may be errors.
        This function prints out the orders which are not consistent then
        this can be inspected by the used to make change changes required.
        It can also be used to confirm that the items across rows are
        consistent.

        Args:
            df (pd.DataFrame): complete data frame
            key_col (str): col which links all rows in an order together
            check_cols (list[str]): columns which should be same in all rows

        Returns:
            set: Set of orders which are inconsistent
        """
        # Count distinct values per order per column
        counts = df.groupby(key_col, dropna=False)[check_cols].nunique(dropna=True)

        # Identify inconsistent orders (any column > 1 unique value)
        inconsistent_mask = ~counts.le(1).all(axis=1)
        inconsistent_orders = counts.index[inconsistent_mask]

        if inconsistent_orders.empty:
            print("All orders are consistent across the specified columns.")
            return True

        print(f"Found {len(inconsistent_orders)} inconsistent orders:\n")

        inconsistent_set = set()
        for order_id in inconsistent_orders:
            inconsistent_set.add(order_id)
            # Columns that differ
            bad_cols = counts.columns[counts.loc[order_id] > 1].tolist()
            print(f"Order {order_id} — inconsistent in: {', '.join(bad_cols)}\n")

            # Print full rows for this order
            display_df = df[df[key_col] == order_id]
            print(display_df.to_string(index=False))
            print("-" * 80)

        return inconsistent_set

    def fill_blank_relative(
        self,
        df: pd.DataFrame,
        blank_col_name: str,
        relative_col_names: list[str],
    ) -> pd.DataFrame:
        """This function can be used to fill missing data when we
        know that there is a 1 to one relationship between the column
        we are trying to fill and other data. For example If we are
        missing an city in an address, we know the postal code and
        state / country will be relative.

        It is important to only use columns which have this relationship
        which has to be confirmed first. The function check_order_consistency
        can be used for this.

        Args:
            df (pd.DataFrame): The data frame containing the data
            blank_col_name (str): The column which have missing data
            relative_col_names (list[str]): The columns which are relative
            to the column missing data.

        Returns:
            pd.DataFrame: Returns the updated dataframe
        """

        blank_mask = df[blank_col_name].isna()
        rel_present_mask = df[relative_col_names].notna().all(axis=1)

        to_fill_mask = blank_mask & rel_present_mask
        if not to_fill_mask.any():
            return df

        lookup_df = df[~blank_mask & rel_present_mask][
            relative_col_names + [blank_col_name]
        ].drop_duplicates(subset=relative_col_names)

        if lookup_df.empty:
            return df

        to_fill_df = df.loc[to_fill_mask, relative_col_names + [blank_col_name]]

        merged = to_fill_df.merge(
            lookup_df,
            on=relative_col_names,
            how="left",
            suffixes=("", "_source"),
        )

        filled_values = merged[blank_col_name].fillna(
            merged[blank_col_name + "_source"]
        )

        df.loc[to_fill_mask, blank_col_name] = filled_values.to_numpy()

        return df
