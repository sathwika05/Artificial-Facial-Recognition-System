from datetime import datetime

import pandas as pd

import os

def checkattendence(Attendence_path):


    if "Attendence_login.csv" in Attendence_path:
        if os.path.exists(Attendence_path):
            pass
        else:
            Date = []
            Name = []
            nameDate = []
            LoginTime = []
            LogOutTime = []

            data = {
                "Date" : Date,
                "namedate" : nameDate,
                "Name" : Name,
                "LoginTime" : LoginTime,
            }

            data_file = pd.DataFrame(data=data)

            data_file.to_csv(Attendence_path, index=False)

    if "Attendence_logout.csv" in Attendence_path:

        if os.path.exists(Attendence_path):
            pass
        else:
            Date = []
            Name = []
            nameDate = []
            LoginTime = []
            LogOutTime = []

            data = {
                "Date" : Date,
                "namedate" : nameDate,
                "Name" : Name,
                "LogOutTime" : LogOutTime,
            }

            data_file = pd.DataFrame(data=data)
            data_file.to_csv(Attendence_path, index=False)


Attendence_path_login = "Attendence_login.csv"
Attendence_path_logout = "Attendence_logout.csv"



def addnewrow(Attendence_path, Date=[], nameDate=[], Name=[], LogOutTime=[], LoginTime=[]):

    if "Attendence_login.csv" in Attendence_path:
        df_login = pd.read_csv(Attendence_path_login)
        df_login.loc[len(df_login.index)] = [Date, nameDate, Name, LoginTime] 
        df_login.to_csv(Attendence_path, index=False)
    if "Attendence_logout.csv" in Attendence_path:
        df_logout = pd.read_csv(Attendence_path_logout)
        df_logout.loc[len(df_logout.index)] = [Date, nameDate, Name, LogOutTime] 
        df_logout.to_csv(Attendence_path, index=False)


def merge_file(Attendence_path_login, Attendence_path_logout):
    csv1 = pd.read_csv(Attendence_path_login)
    csv2 = pd.read_csv(Attendence_path_logout)

    merged_data = csv1.merge(csv2,on=["namedate"])

    return merged_data

