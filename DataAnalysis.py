import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
#from scipy import stats

#load data: loads weather data and returns it as a dataframe: either for a single station or taking the average over mutliple stations
def load_weather_data(stations: dict=None, single_station=None):
    if single_station:
        df = pd.read_csv(stations[single_station], low_memory=False)
        #missing data is encoded with an 'M', so .to_numeric() is needed here
        df["tmpc"] = pd.to_numeric(df["tmpc"], errors="coerce")
        df.drop(columns=["station"],inplace=True)
    else:
        # read .csv files with weather data in stations
        dfs=[]
        for f in stations.values():
            df = pd.read_csv(f, low_memory=False)
            df["tmpc"] = pd.to_numeric(df["tmpc"], errors="coerce")
            dfs.append(df)
        #only keep columns 'valid' & 'tmpc'
        dfs_tmp = [df[['valid', 'tmpc']] for df in dfs]
        merged_df = pd.concat(dfs_tmp)
        # calculate the mean over all stations
        df = merged_df.groupby('valid', as_index=False)['tmpc'].mean()

    #use datetime to access year, month, day and minute individually
    df["valid"] = pd.to_datetime(df["valid"])

    #add grouping keys year month, day, hour, minute
    df["year"] = df["valid"].dt.year
    df["month"] = df["valid"].dt.month
    df["day"] = df["valid"].dt.day
    df["hour"] = df["valid"].dt.hour
    df["minute"] = df["valid"].dt.minute
    df.drop(columns=["valid"],inplace=True)
    return df

def choose_stations(stations):
    choice=""
    while choice not in ["y","n"]:
        choice = input("Would you like to pick a single station by yourself? Else we take the average over all available stations. y/n ")

    if choice == "y":
        station=""
        while station not in stations.keys():
            station = input(f"Which weather station should be used? You can choose between {', '.join(stations.keys())}: ")
        return station
    else:
        return None

def compute_mean(df, grouping_keys: list, col: str, name: str):
    """computes the mean over entries of a given data frame
    df:             the data frame used
    grouping_keys:  a list containing the keys for grouping the data frame
    col:            the column of which the mean should be calculated
    name:           new name for the column with the means"""
    return df.groupby(grouping_keys)[col].mean().reset_index().rename(columns={col:name})


#load energy consumption
def load_consumption_data(file: str):
    cons_dev = pd.read_csv(file, sep=";")
    cons_dev["MW"] = pd.to_numeric(cons_dev["MW"], errors="coerce")

    #mistake in the given data
    decision=""
    while decision not in ["y","n"]:
        print("\nTable of Energy Consumption deviation (2017), 4th of June 2017, 00:00:")
        print(cons_dev.loc[(cons_dev["timestamp (UTC)"] == "04.06.2017 00:00")])
        decision = input("\nThe entry for the deviation in consumption on 4 June 2017 at 00:00 is incorrect. Delete incorrect value? y/n ")

    if decision == "y":
        cons_dev.loc[7386, 'MW'] = float("nan")

    #use datetime to access year, month, day and minute individually
    cons_dev["timestamp (UTC)"] = pd.to_datetime(cons_dev["timestamp (UTC)"], dayfirst=True)

    #add grouping keys month, day, hour, minute, drop the original timestamp
    cons_dev["month"] = cons_dev["timestamp (UTC)"].dt.month
    cons_dev["day"] = cons_dev["timestamp (UTC)"].dt.day
    cons_dev["hour"] = cons_dev["timestamp (UTC)"].dt.hour
    cons_dev["minute"] = cons_dev["timestamp (UTC)"].dt.minute
    cons_dev.drop(columns=["timestamp (UTC)"], inplace=True)
    return cons_dev

def corr(df, k1: str, k2: str):
    corr = df[k1].corr(df[k2])
    print("Correlation of " + k1 + " and " + k2 +" :" , corr)
    return corr

#scatterplot, s1 and s2 are the relevant columns
def scatterplot(df1, df2, s1,s2):
    plt.scatter(df1[s1], df1[s2], c="blue", alpha=0.5, label="Winter")
    plt.scatter(df2[s1], df2[s2], c="red", alpha=0.5, label="Sommer")
    plt.legend()
    plt.xlabel(s1)
    plt.ylabel(s2)
    plt.autoscale()
    plt.show()

# linear regression and plot using statsmodels
def lin_reg_statsmodels(df, x_col, y_col, title):
    df = df.dropna(subset=[x_col, y_col])
    x = df[x_col]
    y = df[y_col]

    X = sm.add_constant(x)
    X.columns = ["intercept", "slope"]
    model = sm.OLS(y, X).fit()
    print(f"\n===== {title.upper()} =====")
    print(model.summary())

    x_fit = np.linspace(x.min(), x.max(), 100)
    X_fit = sm.add_constant(x_fit)
    y_fit = model.predict(X_fit)
    ci = model.get_prediction(X_fit).conf_int()

    plt.scatter(x, y, alpha=0.5, label="Daten")
    plt.plot(x_fit, y_fit, 'r-', label="Regression")
    plt.fill_between(x_fit, ci[:, 0], ci[:, 1], color='red', alpha=0.2, label="95% CI")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.show()

# Linear regression using Scipy
"""def lin_reg_scipy(df, x_col, y_col, title):
    df = df.dropna(subset=[x_col, y_col])
    x = df[x_col].values
    y = df[y_col].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)


    print(f"\n===== {title.upper()} (SciPy) =====")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R²: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Plot
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = intercept + slope * x_fit

    plt.scatter(x, y, alpha=0.5, label=f"{season_name} data")
    plt.plot(x_fit, y_fit, color='red', label=f"Fit line (R²={r_value**2:.2f})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{title}: Linear regression (SciPy)")
    plt.legend()
    plt.show()

    return slope, intercept, r_value**2, p_value, std_err"""

def main():
    #weather stations available for the analysis
    weather_stations = {"Paris-CDG":"LFPG.csv",
            "Paris-Orly":"LFPO.csv",
            "Marseille":"LFML.csv",
            "Lyon":"LFLY.csv",
            "Toulouse":"LFBO.csv",
            "Nizza":"LFMN.csv",
            "Nantes":"LFRS.csv",
            "Strasbourg":"LFST.csv",
            "Montpellier":"LFMT.csv",
            "Rennes":"LFRN.csv",
            "Bordeaux":"LFBD.csv",
            "Orleans":"LFOJ.csv",
            "Le Mans":"LFRM.csv",
            "Dijon":"LFSD.csv",
            "Le Havre":"LFOH.csv",
            "Biarritz-bayonne":"LFBZ.csv",
            "Perpignan":"LFMP.csv",
            "Cannes":"LFMD.csv",
            "Nancy":"LFSO.csv",
            "Lille":"LFQQ.csv",
    }

    # 1. load weather data
    station=choose_stations(weather_stations)
    df = load_weather_data(stations=weather_stations,single_station=station)

    #2. exclude the year 2017, separate table for 2017, drop columns from the table for 2017
    df_no2017 = df[df["year"] < 2017].copy()
    df_2017 = df[df["year"] == 2017].copy()

    #3. Compute means

    #3.1 Daily mean temperatures
    df_daily = compute_mean(df_no2017, ["year","month","day"], "tmpc","daily mean temp").drop(columns=["year"])
    df_daily_2017 = compute_mean(df_2017, ["month","day"], "tmpc","daily average temperature (2017)")
    #3.2 Mean temperatures over all years before 2017
    df_mean = compute_mean(df_daily, ["month","day"], "daily mean temp","mean temp (before 2017)")

    #4 Load consumption data & daily average
    df_cons = load_consumption_data("Consumption deviation 2017.csv")
    df_cons_daily = compute_mean(df_cons, ["month","day"], "MW","Avg. daily cons. deviation (2017)")

    #5. Add mean temp, temp 2017 and temp deviation
    df_cons_daily = df_cons_daily.merge(df_mean, on = ["month", "day"], how="left")
    df_cons_daily = df_cons_daily.merge(df_daily_2017, on = ["month", "day"], how="left")
    df_cons_daily["Temperature deviation (2017)"] = df_cons_daily["daily average temperature (2017)"]-df_cons_daily["mean temp (before 2017)"]

    #6. separate tables for summer and winter
    summer = df_cons_daily[df_cons_daily["month"].isin([6,7,8])]
    winter = df_cons_daily[df_cons_daily["month"].isin([12,1,2])]

    #7. Calculate correlation
    print("")
    print("Winter: ", end="" )
    corr_winter = corr(winter, "Temperature deviation (2017)", "Avg. daily cons. deviation (2017)")
    print("Summer: ", end="" )
    corr_summer = corr(summer, "Temperature deviation (2017)", "Avg. daily cons. deviation (2017)")

    #8. Scatterplot, linear regression and plot
    scatterplot(winter, summer , "Temperature deviation (2017)","Avg. daily cons. deviation (2017)")
    lin_reg_statsmodels(summer, "Temperature deviation (2017)","Avg. daily cons. deviation (2017)", "Summer")
    lin_reg_statsmodels(winter, "Temperature deviation (2017)","Avg. daily cons. deviation (2017)", "Winter")
    #lin_reg_scipy(summer, "summer")
    #lin_reg_scipy(winter, "winter")

if __name__ == "__main__":
    main()
