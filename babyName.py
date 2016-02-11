import glob
import pandas as pd
import numpy as np

# Returns True if SEX column contains both M and F, False otherwise
def has_both_sexes(input_frame):
    male = np.any(input_frame.SEX.str.contains("M", case=False))
    female = np.any(input_frame.SEX.str.contains("F", case=False))
    return male and female

    
# Read in files, construct a DataFrame from each, then concatenate all.
def get_data():
    # Constants for DataFrame read-in and construction
    COLUMN_NAMES = ["STATE", "SEX", "YEAR", "NAME", "COUNT"]
    datafiles = glob.glob('namesbystate/*.txt')
    fileframes = []
    
    for file in datafiles:
        fileframes.append(pd.read_csv(file, header=None, names=COLUMN_NAMES))
    names_df = pd.concat(fileframes)
    names_df["YEAR"] = pd.to_datetime(names_df["YEAR"], format='%Y')
    return names_df


# Find most popular name in dataset
# Returns a tuple of Name, Count
def most_popular_name(names_df,gender):

    dfgroups = names_df[names_df['SEX']==gender].groupby(["NAME"]).aggregate(np.sum)
    dfgroups = dfgroups.sort_values("COUNT", ascending=False)
    dfgroups.reset_index(inplace=True)
    return tuple(dfgroups.values[0]) 
  

# Ambiguity is calculated by difference in male and female totals, lower is more ambiguous
def ambiguous_name_by_year(year, names_df):
    ARGYEAR = str(year)
    
    # Filter only the names that occur in both sexes and in the correct year
    names_df_year = names_df[names_df.YEAR == ARGYEAR]
    names_df_year = names_df_year.groupby(["NAME"]).filter(has_both_sexes)

    # Find total counts of names for each name, grouped by sex
    names_df_year = names_df_year.groupby(["NAME", "SEX"]).aggregate(np.sum)
    
    # Unstack our MultiIndex so that the male and female counts are columns
    names_df_year = names_df_year.unstack(level=-1)
   
    # Calculate difference and total
    names_df_year["DIFFERENCE"] = abs(names_df_year["COUNT", "F"] - names_df_year["COUNT", "M"])
    names_df_year["TOTAL"] = names_df_year["COUNT", "F"] + names_df_year["COUNT", "M"]
    names_df_year.reset_index(inplace=True)
    
    # Sort our data for minimum difference and maximum total, in case of a tie.
    names_df_year.sort_values(["DIFFERENCE", "TOTAL"], ascending=[True, False], inplace=True)
   
    return ARGYEAR, names_df_year.iat[0,0], names_df_year.iat[0,1], names_df_year.iat[0,2]
    
    
def winners_and_losers(year, names_df, extrapolate=False):
    START_YEAR = str(year)
    LAST_YEAR = "2014"
    
    # Separate data in to the desired start year and the last year in the dataset
    # and merge them with an added label
    names_in_year = names_df[names_df.YEAR == START_YEAR].groupby("NAME").aggregate(np.sum)
    names_current = names_df[names_df.YEAR == LAST_YEAR].groupby("NAME").aggregate(np.sum)
    names_current = names_current.merge(names_in_year, how='outer', \
                                        left_index=True, right_index=True, \
                                        suffixes=['_LAST', '_START'])
    
    # Calculate percentage change across the time period.
    # Names that don't exist in one of the years are propagated as NaN and not considered,
    # unless extrapolate=True
    
    # Extrapolation gives possible winner and loser
    if extrapolate:
        names_current.fillna(value=1, axis='columns', inplace=True)
        
    names_current["PCT_CHANGE"] = (names_current["COUNT_LAST"] - names_current["COUNT_START"]) \
                                    / names_current["COUNT_START"]

    # Calculate largest and grab name + percentage change
    biggest_winner = names_current["PCT_CHANGE"].nlargest(1)
    biggest_winner_name = biggest_winner.index[0]
    biggest_winner_pct = biggest_winner.values[0] * 100
    
    # Calculate smallest and grab name + percentage change
    biggest_loser = names_current["PCT_CHANGE"].nsmallest(1)
    biggest_loser_name = biggest_loser.index[0]
    biggest_loser_pct = biggest_loser.values[0] * 100
    
    # If we're extrapolating the percentage changes aren't accurate.
    if extrapolate:
        return START_YEAR, LAST_YEAR, biggest_winner_name, biggest_loser_name
    else:
        return START_YEAR, LAST_YEAR, biggest_winner_name, biggest_winner_pct, \
                biggest_loser_name, biggest_loser_pct


def main():
    state_names = get_data()
    print("The most popular female name is %s, with a total count of %i" % most_popular_name(state_names,"F"))
    print("The most popular male name is %s, with a total count of %i" % most_popular_name(state_names,"M"))
    print("The most gender ambiguous name in %s is %s: %d girls and %d boys." % ambiguous_name_by_year(2013, state_names))
    print("The most gender ambiguous name in %s is %s: %d girls and %d boys." % ambiguous_name_by_year(1945, state_names))
    print("From %s to %s the name with the greatest increase in popularity was %s, with a %2.1f%% increase.\n The name with the greatest decline in popularity was %s, with a %2.1f%% change." % winners_and_losers(1980, state_names))
    print( "Inferring from extrapolated values, from %s to %s the name with the greatest growth was %s,\nand the name with the greatest decline was %s." % winners_and_losers(1980, state_names, extrapolate=True))
    
if __name__ == "__main__":
    main()