import pickle
import json
from time import sleep
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
a=int(input("ENTER INPUT FOR N "))
b=int(input("ENTER INPUT FOR P "))
c=int(input("ENTER INPUT FOR K "))
d=float(input("ENTER INPUT FOR TEMP "))
e=float(input("ENTER INPUT FOR HUMIDITY "))
f=float(input("ENTER INPUT FOR PH "))
g=float(input("ENTER INPUT FOR RAINFALL "))
person_reports = [[a,b,c,d,e,f,g]]
predicted = loaded_model.predict(person_reports)
print("ANALYSING....")
print(predicted)