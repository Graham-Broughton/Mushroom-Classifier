import gc
import glob
import hashlib
import multiprocessing
import os
import sqlite3

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def squeeze_dataframe(df):
    # Get columns in dataframe
    cols = dict(df.dtypes)

    # Check each column's type, downcast or categorize as appropriate
    for col, type in cols.items():
        if type == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif type == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif type == "object":
            df[col] = df[col].astype(CategoricalDtype(ordered=True))

    return df


def panda_mem_usage(df, detail="full"):
    dtypes = df.dtypes.reset_index(drop=False)
    memory = df.memory_usage(deep=True).reset_index(drop=False)

    df1 = pd.merge(dtypes, memory, on="index")
    df1 = df1.rename(columns={"index": "col", "0_x": "type", "0_y": "bytes"})
    total = df1["bytes"].sum()

    objects = df.select_dtypes(include=["object", "category"])
    df_objs = (
        objects.select_dtypes(include=["object", "category"]).describe().T.reset_index()
    )

    if detail == "full":
        print("")
        print(
            "{:<15} {:<15} {:>15} {:>8} {:>8}".format(
                "Column", "Data Type", "Bytes", "MBs", "GBs"
            )
        )
        print("{} {} {} {} {}".format("-" * 15, "-" * 15, "-" * 15, "-" * 8, "-" * 8))

        for index, row in df1.iterrows():
            print(
                "{:<15} {:<15} {:>15,.0f} {:>8,.1f} {:>8,.2f}".format(
                    row["col"],
                    str(row["type"]),
                    row["bytes"],
                    row["bytes"] / 1024**2,
                    row["bytes"] / 1024**3,
                )
            )

        print(
            "\nTotal: {:,.0f} Rows, {:,.0f} Bytes, {:,.1f} MBs, {:,.2f} GBs\n".format(
                len(df), total, total / 1024**2, total / 1024**3
            )
        )

        print("{:<15} {:>13} {:>13}".format("Column", "Count", "Unique"))
        print("{} {} {}".format("-" * 15, "-" * 13, "-" * 13))
        for index, row in df_objs.iterrows():
            print(
                "{:<15} {:>13,.0f} {:>13,.0f}".format(
                    row["index"], row["count"], row["unique"]
                )
            )

    elif detail == "return_short":
        return len(df), total


def write_file(df, fn, compression=""):
    fn_ext = os.path.splitext(fn)[1]

    if fn_ext == ".csv":
        df.to_csv(fn, index=False)

    elif fn_ext == ".zip":
        df.to_csv(fn, compression=dict(method="zip", archive_name="data"), index=False)

    elif fn_ext == ".parquet":
        compression = "brotli" if compression == "" else compression
        df.to_parquet(fn, engine="pyarrow", compression=compression)

    elif fn_ext == ".feather":
        compression = "zstd" if compression == "" else compression
        df.to_feather(fn, compression=compression)

    elif fn_ext == ".h5":
        compression = "blosc:lz4" if compression == "" else compression
        df.to_hdf(
            fn,
            key="data",
            mode="w",
            format="table",
            index=False,
            complevel=9,
            complib=compression,
        )

    elif fn_ext == ".pkl":
        compression = "zip" if compression == "" else compression
        df.to_pickle(fn, compression=compression)

    elif fn_ext == ".sqlite":
        con = sqlite3.connect(fn)
        df.to_sql("data", con=con, if_exists="replace", index=False)
        con.close()

    elif fn_ext == ".json":
        df.to_json(fn, orient="records")

    elif fn_ext == ".xlsx":
        writer = pd.ExcelWriter(fn, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="quotes", index=False)
        # add more sheets by repeating df.to_excel() and change sheet_name
        writer.save()

    else:
        print("oopsy in write_file()! File extension unknown:", fn_ext)
        quit(0)

    return


def read_file(fn, compression="", sql=""):
    fn_ext = os.path.splitext(fn)[1]

    if fn_ext == ".csv" or fn_ext == ".zip":
        df = pd.read_csv(fn, keep_default_na=False)

    elif fn_ext == ".parquet":
        df = pd.read_parquet(fn)

    elif fn_ext == ".feather":
        df = pd.read_feather(fn)

    elif fn_ext == ".h5":
        df = pd.read_hdf(fn, key="data")

    elif fn_ext == ".pkl":
        df = pd.read_pickle(
            fn, compression=compression
        ).copy()  # copy added because of some trouble with categories not fully read by mem util on first pass

    elif fn_ext == ".sqlite":
        if sql == "":
            sql = "SELECT * FROM data"
        con = sqlite3.connect(fn)
        df = pd.read_sql(sql, con)
        con.close()

    elif fn_ext == ".json":
        df = pd.read_json(fn, convert_dates=False)

    elif fn_ext == ".xlsx":
        df = pd.read_excel(fn, sheet_name="quotes", keep_default_na=False)

    else:
        print("oopsy in read_file()! File extension unknown:", fn_ext)
        quit(0)

    return df


def sample_id(string, samples=100):
    """Given a string, return a repeatable integer between 0 and q-1.
            - don't use expecting a perfect sample
            - this is really just a quick way to split up a dataset into roughly equal parts in a way that's repeatable
            - from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    """
    sample = int(hashlib.sha256(string.encode("utf8")).hexdigest(), 16) % samples

    return sample


def multiproc_run_target(*args):
    """Companion to multiproc_dataframe below - sends to data enhancement function and saves results to be read by parent function.
    Done this way so the data enhancement function can remain unchanged from one that does a straight apply().

    Args:
        *args: function to apply to dataframe, dataframe, and any other arguments to be passed to function
    """

    # First argument is filename of data split
    function = args[0]
    fn = args[1]

    args1 = []

    # read in file from multiproc_dataframe parent function - is is first df= argument for enhancement function
    df = pd.read_feather(fn)
    args1.append(df)

    # append any additional parameters that are provided for enhancement function
    for arg in args[2:]:
        args1.append(arg)

    # Run function
    df = function(*args1)

    # Save returned dataframe so it can be read by parent when all multitasking processes have finished
    df.to_feather(fn, compression="lz4")

    del df
    gc.collect()


def multiproc_dataframe(**kwargs):
    """Main function for multiprocessing pandas dataframes.
    
    Args:
        function (function): function to apply to dataframe - must have dataframe as first argument
        df (dataframe): dataframe to apply function to
        procs (int): number of processes to use
        splitby (str): column name to split dataframe by
        any other arguments to be passed to function
    
    Returns:
        dataframe: dataframe with function applied to it
    """
    # load required parameter to pass onto the function, and load optionally added parameters to a list
    added_args = []

    for key, val in kwargs.items():
        if key == "function":
            function = val

        elif key == "df":
            df = val

        elif key == "procs":
            procs = val

        elif key == "splitby":
            splitby = val

        else:
            added_args.append(val)

    if "procs" not in kwargs:
        procs = multiprocessing.cpu_count()

    if "splitby" not in kwargs or splitby == None:
        df_splits = np.array_split(df, procs)

        temp_files = []
        for i in range(len(df_splits)):
            temp_files.append("temp_file_" + str(i) + ".feather")
            dfs = df_splits[i].reset_index(drop=True)
            dfs.to_feather(temp_files[i], compression="lz4")

    else:  # create splits by splitting symbols and then finding their related starting and ending rows
        temp_files = []
        for i in range(procs):
            temp_files.append("temp_file_" + str(i) + ".feather")

        df1 = df[splitby].drop_duplicates(keep="first").reset_index(drop=False)

        df_splits = np.array_split(df1, procs)

        for i in range(len(df_splits)):
            start_split = df_splits[i].iloc[0]["index"]

            if i == len(df_splits) - 1:
                end_split = None
            else:
                end_split = df_splits[i + 1].iloc[0]["index"]

            dfs = df[start_split:end_split].reset_index(drop=True)
            dfs.to_feather(temp_files[i], compression="lz4")
            del dfs

    del df
    gc.collect()

    # Initialize
    manager = multiprocessing.Manager()
    processes = []

    # Process splits and concatenate all element in the return_list
    for i in range(procs):
        process = multiprocessing.Process(
            target=multiproc_run_target, args=[function, temp_files[i]] + added_args
        )
        processes.append(process)

        process.start()

    for process in processes:
        process.join()

    dfs = []
    for fn in temp_files:
        dfs.append(pd.read_feather(fn))
        os.remove(fn)

    df = pd.concat(dfs, ignore_index=True)
    df = df.reset_index(drop=True)

    return df


def del_files(path):
    files = glob.glob(path)

    if len(files) == 0:
        print('no files to delete matching "' + path + '"')
        return

    for file in files:
        print("Deleting", file)
        os.remove(file)
