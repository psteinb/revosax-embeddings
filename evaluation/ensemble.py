import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    print(Path(".").absolute())
    return


@app.cell
def _():
    import polars as pl


    totalsdf = pl.read_csv("evaluation/revosax-eval-totals.csv", 
                          row_index_name=None)
    ensdf = pl.read_csv("evaluation/revosax-eval-ensemble.csv", 
                          row_index_name=None)
    return ensdf, pl, totalsdf


@app.cell
def _(totalsdf):
    totalsdf
    return


@app.cell
def _(ensdf):
    ensdf
    return


@app.cell
def _(ensdf, totalsdf):
    print(ensdf.shape, totalsdf.shape)
    return


@app.cell
def _(totalsdf):
    totalsdf.columns
    return


@app.function
# totals_long = totalsdf.melt(
#    id_vars=[], variable_name="Category", value_name="Value"
# )
# /tmp/marimo_147023/__marimo__cell_Rzwy_.py:1: DeprecationWarning: `DataFrame.melt` is deprecated; use `DataFrame.unpivot` instead, with `index` instead of `id_vars` and `on` instead of `value_vars`

def melt(df):
    columns_to_unpivot = [
        col for col in df.columns if col != "index" and len(col) > len("index")
]
    value = df.select(columns_to_unpivot).unpivot(index=None)
    return value


@app.cell
def _(ensdf, totalsdf):
    ens_long = melt(ensdf)
    totals_long = melt(totalsdf)
    return ens_long, totals_long


@app.cell
def _(ens_long):
    ens_long
    return


@app.cell
def _(totals_long):
    totals_long
    return


@app.cell
def _(ens_long):

    ens_long_pd = ens_long.to_pandas()
    ens_long_pd["variable"] = ens_long_pd["variable"].astype("category") 
    ens_long_pd.head()
    return (ens_long_pd,)


@app.cell
def _(totals_long):
    totals_long
    return


@app.cell
def _(ens_long_pd):
    import matplotlib.pyplot as plt

    # Step 1: Group the values by category
    grouped_data = [
        group["value"].values for _, group in ens_long_pd.groupby("variable")
    ]
    colnames = ens_long_pd["variable"].unique()
    print(colnames)
    return colnames, grouped_data, plt


@app.cell
def _(colnames, grouped_data, plt):
    # Step 2: Create boxplot
    plt.boxplot(grouped_data, tick_labels=colnames, vert=False)

    # Optional: Improve readability
    plt.xlabel("Value")
    plt.ylabel("Metric")
    # plt.title("Boxplot by Metric")
    # plt.tight_layout()
    plt.grid(True)
    plt.savefig("overview.jpg", bbox_inches="tight")
    plt.savefig("overview.pdf")
    plt.savefig("overview.png")
    plt.show()
    return


@app.cell
def _(colnames, pl, totals_long):
    order_map = {name: idx for idx, name in enumerate(colnames)}

    # Add a temporary sort key using map_elements
    df = totals_long.with_columns(
        pl.col("variable").map_elements(lambda val: order_map.get(val, float("inf"))).alias("sort_key")
    )
    df_sorted = df.sort("sort_key").drop("sort_key")
    df_sorted
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
