"""
This makes use of scikit-learn
dependencies:
numpy
scipy
scikit-learn
matplotlib
pybrain
etc.
"""

import numpy as np
import pydot

from StringIO import StringIO

from pprint import pprint

import matplotlib.pyplot as plt


#training data is contained in "adult.data"

"""
the csv data is stored as such:
age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
these values are explained in the file adult.names
our first step is to parse the data
"""

#first we define a set of conversion functions from strings to integer values because working with strings is dumb
#especially since the computer doens't care when doing machine learning
def create_mapper(l):
    return {l[n] : n for n in xrange(len(l))}

workclass = create_mapper(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = create_mapper(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
marriage = create_mapper(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = create_mapper(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = create_mapper(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = create_mapper(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = create_mapper(["Female", "Male"])
country = create_mapper(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])
income = create_mapper(["<=50K", ">50K"])

converters = {
    1: lambda x: workclass[x],
    3: lambda x: education[x],
    5: lambda x: marriage[x],
    6: lambda x: occupation[x],
    7: lambda x: relationship[x],
    8: lambda x: race[x],
    9: lambda x: sex[x],
    13: lambda x: country[x],
    14: lambda x: income[x]
}

"""
load the data into numpy
this section is also written for use in a
"""
train = "adult.data"
test = "adult.test"


def load(filename):
    with open(filename) as data:
        adults = [line for line in data if "?" not in line]  # remove lines with unknown data

    return np.loadtxt(adults,
                      delimiter=', ',
                      converters=converters,
                      dtype='u4',
                      skiprows=1
                      )


def start_adult():
    """
    tx - training x axes
    ty - training y axis
    rx - result (testing) x axes
    ry - result (testing) y axis
    """
    tr = load(train)
    te = load(test)
    tx, ty = np.hsplit(tr, [14])
    rx, ry = np.hsplit(te, [14])
    ty = ty.flatten()
    ry = ry.flatten()
    return tx, ty, rx, ry

if __name__ == "__main__":
    tx, ty, rx, ry = start_adult()