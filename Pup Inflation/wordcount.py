# Utsav Anantbhat

import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types, Row
import string, re

wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation

def get_lines(in_directory):
    
    # 1. Read lines from the files with spark.read.text.
    lines = spark.read.text(in_directory)
    return lines

def main(in_directory, out_directory):
    # Load data into dataframe and filter out empty lines
    
    lines = get_lines(in_directory)
    lines = lines.filter(lines.value != "")

    # 2. Split the lines into words with the regular expression.
    # 2a. Use the split and explode functions.  
    lines = lines.withColumn("word", functions.explode(functions.split(lines.value, wordbreak)))
    
    # 2b. Normalize all of the strings to lower-case (so “word” and “Word” are not counted separately.)
    lines = lines.withColumn("word", functions.lower(lines.word))
    
    # 5. Remove empty strings from output
    lines = lines.filter(lines.word != "") 

    # 3. Count the number of times each word occurs.
    word_count = lines.groupby("word").count()

    # 4. Sort by decreasing count (i.e. frequent words first) and alphabetically if there's a tie.
    words = word_count.sort(functions.col("count").desc(), functions.col("word").asc())

    # 6. Write results as CSV files with the word in the first column, and count in the second (uncompressed: they aren't big enough to worry about).
    words.write.mode("overwrite").option("header", True).csv(out_directory)



if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)