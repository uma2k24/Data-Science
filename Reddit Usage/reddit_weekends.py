# Utsav Anantbhat

import sys
import numpy as np
import pandas as pd
import scipy.stats as py_stats
import matplotlib.pyplot as plt
import datetime as dt


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]

    # ...

    counts = pd.read_json(reddit_counts, lines=True) #read the json file

    counts = counts[((counts['date'].dt.year == 2012) | (counts['date'].dt.year == 2013)) & (counts['subreddit'] == 'canada')] #only look at values in years 2012 and 2013 and in r/canada
    #counts = counts[counts['subreddit'] == 'canada'] #only look at the r/canada subreddit
    counts.drop('subreddit', axis=1, inplace=True)

    counts_wday = counts[((counts['date'].dt.weekday == 0) | (counts['date'].dt.weekday == 1) #separate the weekday and weekend data; Mon-Fri = 0-4, Sat & Sun = 5 & 6
                        | (counts['date'].dt.weekday == 2) | (counts['date'].dt.weekday == 3) 
                        | (counts['date'].dt.weekday == 4))] 
    counts_wend = counts[((counts['date'].dt.weekday == 5) | (counts['date'].dt.weekday == 6))]


    # Student's T-Test: use scipy.stats to do a T-test on the data to get a p-value

    student_ttest_p = py_stats.ttest_ind(counts_wday['comment_count'], counts_wend['comment_count'])[1]
    init_weekday_normality_p = py_stats.normaltest(counts_wday['comment_count'])[1]
    init_weekend_normality_p = py_stats.normaltest(counts_wend['comment_count'])[1]
    init_levene_p = py_stats.levene(counts_wday['comment_count'], counts_wend['comment_count'])[1]


    # Fix 1: transforming data might save us. 
    # Transform the counts so the data doesn't fail the normality test.

    counts_wday_transformed = counts_wday['comment_count'].to_numpy()
    counts_wday_transformed = np.log(counts_wday_transformed) #use np.log as it comes closest to normal distribution
    counts_wend_transformed = counts_wend['comment_count'].to_numpy()
    counts_wend_transformed = np.log(counts_wend_transformed)

    transformed_wday_normality_p = py_stats.normaltest(counts_wday_transformed)[1]
    transformed_wend_normality_p = py_stats.normaltest(counts_wend_transformed)[1]
    levene_p_transformed = py_stats.levene(counts_wday_transformed, counts_wend_transformed)[1]


    # Fix 2: the Central Limit Theorem might save us. 
    # Combine all weekdays and weekend days from each year/week pair and take the mean of their (non-transformed) counts.

    wday_isocalendar = counts_wday['date'].dt.isocalendar() #get a “year” and “week number” from the first two values returned by date.isocalendar()
    wend_isocalendar = counts_wend['date'].dt.isocalendar()
    wday_isocalendar = wday_isocalendar[['year', 'week']]
    wend_isocalendar = wend_isocalendar[['year', 'week']]
    counts_wday = pd.concat([counts_wday, wday_isocalendar], axis=1) #use Pandas to group by that value...
    counts_wend = pd.concat([counts_wend, wend_isocalendar], axis=1)
    
    counts_wday_CLT = counts_wday.groupby(['year', 'week']).aggregate('mean').reset_index() # ...and aggregate taking the mean.
    counts_wend_CLT = counts_wend.groupby(['year', 'week']).aggregate('mean').reset_index() # Note: the year returned by isocalendar isn't always the same as the date's year (around the new year). 
    #Use the year from isocalendar, which is correct for this.

    weekly_wday_normality_p = py_stats.normaltest(counts_wday_CLT['comment_count'])[1]
    weekly_wend_normality_p = py_stats.normaltest(counts_wend_CLT['comment_count'])[1]
    levene_p_weekly = py_stats.levene(counts_wday_CLT['comment_count'], counts_wend_CLT['comment_count'])[1]
    ttest_p_weekly = py_stats.ttest_ind(counts_wday_CLT['comment_count'], counts_wend_CLT['comment_count'], equal_var=True)[1]

    # Fix 3: a non-parametric test might save us. 
    # Perform a U-test on the (original non-transformed, non-aggregated) counts. 
    # Note that we should do a two-sided test here, which will match the other analyses. 
    # Make sure you get the arguments to the function correct.

    mann_whitney = py_stats.mannwhitneyu(counts_wday['comment_count'], counts_wend['comment_count'], alternative="two-sided")[1]


    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=student_ttest_p,
        initial_weekday_normality_p=init_weekday_normality_p,
        initial_weekend_normality_p=init_weekend_normality_p,
        initial_levene_p=init_levene_p,
        transformed_weekday_normality_p=transformed_wday_normality_p,
        transformed_weekend_normality_p=transformed_wend_normality_p,
        transformed_levene_p=levene_p_transformed,
        weekly_weekday_normality_p=weekly_wday_normality_p,
        weekly_weekend_normality_p=weekly_wend_normality_p,
        weekly_levene_p=levene_p_weekly,
        weekly_ttest_p=ttest_p_weekly,
        utest_p=mann_whitney,
    ))


if __name__ == '__main__':
    main()
