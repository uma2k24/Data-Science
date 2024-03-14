# Utsav Anantbhat

import sys
import numpy as np
import pandas as pd
import scipy.stats as py_stats


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]

    # ...

    json_data = pd.read_json('searches.json', orient='records', lines=True) #read the json data

    control_category = json_data[json_data['uid'] % 2 == 0] #even category is control
    treatment_category = json_data[json_data['uid'] % 2 == 1] #odd category is treatment

    json_data['class'] = json_data.apply(lambda x: filter_uid(x['uid']), axis=1) #axis 1 -> rows
    json_data['search >= 1'] = json_data.apply(lambda x: filter_search_count(x['search_count']), axis=1)
    contingency_table = pd.crosstab(json_data['class'], json_data['search >= 1']) #contingency table; pd.crosstab: compute cross tabulation with two or more elements (source = pandas documentation)

    # Hint 1
    _, p_value_chi2, _, _ = py_stats.chi2_contingency(contingency_table)

    # Hint 2
    _, p_value_u = py_stats.mannwhitneyu(control_category['search_count'], treatment_category['search_count'], alternative = 'two-sided')

    # Instructor data
    instructor_data = json_data[json_data['is_instructor'] == True] #filter out non-instructor data
    
    control_category = instructor_data[instructor_data['uid'] % 2 == 0]
    treatment_category = instructor_data[instructor_data['uid'] % 2 == 1]

    instructor_data['class'] = instructor_data.apply(lambda x: filter_uid(x['uid']), axis=1) #axis 1 -> rows
    instructor_data['search >= 1'] = instructor_data.apply(lambda x: filter_search_count(x['search_count']), axis=1)
    contingency_table = pd.crosstab(instructor_data['class'], instructor_data['search >= 1']) #contingency table; pd.crosstab: compute cross tabulation with two or more elements (source = pandas documentation)

    # Hint 1
    _, p_value_chi2_instruct, _, _ = py_stats.chi2_contingency(contingency_table)

    # Hint 2
    _, p_value_u_instruct = py_stats.mannwhitneyu(control_category['search_count'], treatment_category['search_count'], alternative = 'two-sided')

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p_value_chi2,
        more_searches_p=p_value_u,
        more_instr_p=p_value_chi2_instruct,
        more_instr_searches_p=p_value_u_instruct,
    ))



# Function to filter uid
def filter_uid(uid):
    if(uid % 2 == 0):
        return "control" #even -> control
    else:
        return "treatment" #odd -> treatment

# Function to filter the search count    
def filter_search_count(search_count):
    return(search_count != 0)


if __name__ == '__main__':
    main()
