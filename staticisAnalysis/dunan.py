def data_wind_speed(value):
    wind_speed33 = spark.sql("select logtime,wind_speed from mydb.dunan2f where wtid = 33")
    values33 = spark.sql("select logtime,value from mydb.%s where logtime in (select logtime from mydb.dunan2f where wtid = 33) and wtid = 33"%value)
    wind_speed43 = spark.sql("select logtime,wind_speed from mydb.dunan2f where wtid = 43")
    values43 = spark.sql("select logtime,value from mydb.%s where logtime in (select logtime from mydb.dunan2f where wtid = 43) and wtid = 43"%value)
    wind_speed33 = wind_speed33.toPandas()
    values33 = values33.toPandas()
    wind_speed43 = wind_speed43.toPandas()
    values43 = values43.toPandas()
    wind_speed33 = wind_speed33.sort_values("logtime")
    values33 = values33.sort_values("logtime")
    wind_speed43 = wind_speed43.sort_values("logtime")
    values43 = values43.sort_values("logtime")
    wind_speed33["feature"] = values33["value"]
    wind_speed43["feature"] = values43["value"]
    wind_speed_data = wind_speed33.append(wind_speed43)
    return wind_speed_data
def data_ambient_temp(value):
    ambient_temp33 = spark.sql("select logtime,value from mydb.ambient_temp where wtid = 33 and logtime in (select logtime from mydb.dunan2f where wtid = 33) and value <40")
    values33 = spark.sql("select logtime ,value from mydb.%s where wtid = 33 and logtime in (select logtime from mydb.ambient_temp where wtid = 33 and logtime in (select logtime from mydb.dunan2f where wtid = 33) and value <40)"%value)
    ambient_temp43 = spark.sql("select logtime,value from mydb.ambient_temp where wtid = 43 and logtime in (select logtime from mydb.dunan2f where wtid = 43) and value <40")
    values43 = spark.sql("select logtime, value from mydb.%s where wtid = 43 and logtime in (select logtime from mydb.ambient_temp where wtid = 43 and logtime in (select logtime from mydb.dunan2f where wtid = 43) and value <40)"%value)
    ambient_temp33 = ambient_temp33.toPandas()
    values33 = values33.toPandas()
    ambient_temp43 = ambient_temp43.toPandas()
    values43 = values43.toPandas()
    ambient_temp33 = ambient_temp33.sort_values("logtime")
    values33 = values33.sort_values("logtime")
    ambient_temp43 = ambient_temp43.sort_values("logtime")
    values43 = values43.sort_values("logtime")
    ambient_temp33["feature"] = values33["value"]
    ambient_temp43["feature"] = values43["value"]
    ambient_temp_data = ambient_temp33.append(ambient_temp43)
    ambient_temp_data = ambient_temp_data.rename(columns = {"value":"ambient_temp"})
    return ambient_temp_data
    
def wind_creat_rules(a,b,data):
    import pandas as pd
    rule_min = []
    rule_med = []
    rule_up = []
    rule_dn = []
    wind_speed_up = []
    wind_speed_dn = []
    for (i,j) in zip(a,b):
        c = data.query("wind_speed>%f and wind_speed<=%f"%(i,j))["feature"].describe()
        wind_speed_up.append(i)
        wind_speed_dn.append(j)
        rule_min.append(c[3])
        rule_med.append(c[5])
        rule_up.append(c[4])
        rule_dn.append(c[6])
    wind_speed_rule = pd.DataFrame({"wind_speed_up":wind_speed_up,"wind_speed_dn":wind_speed_dn,"rule_min":rule_min,"rule_med":rule_med,"rule_up":rule_up,"rule_dn":rule_dn})
    return wind_speed_rule
def ambient_creat_rules(a,b,data):
    import pandas as pd
    rule_min = []
    rule_med = []
    rule_up = []
    rule_dn = []
    ambient_temp_up = []
    ambient_temp_dn = []
    for (i,j) in zip(a,b):
        c = data.query("ambient_temp>%f and ambient_temp<=%f"%(i,j))["feature"].describe()
        ambient_temp_up.append(i)
        ambient_temp_dn.append(j)
        rule_min.append(c[3])
        rule_med.append(c[5])
        rule_up.append(c[4])
        rule_dn.append(c[6])
    ambient_temp_rule = pd.DataFrame({"ambient_temp_up":ambient_temp_up,"ambient_temp_dn":ambient_temp_dn,"rule_min":rule_min,"rule_med":rule_med,"rule_up":rule_up,"rule_dn":rule_dn})
    return ambient_temp_rule
    
def plot_wind(data):
    import matplotlib.pyplot as plt
    x = data.wind_speed
    y = data.feature
    plt.xlabel("wind_speed")
    plt.ylabel("feature")
    plt.scatter(x,y,c = "r",marker = ".")
    plt.show()
def plot_ambient(data):
    import matplotlib.pyplot as plt
    x = data.ambient_temp
    y = data.feature
    plt.xlabel("ambient_temp")
    plt.ylabel("feature")
    plt.scatter(x,y,c = "r",marker = ".")
    plt.show()  
    
#gear_box_speed   
wind_speed_data = data_wind_speed("gear_box_speed")
ambient_temp_data = data_ambient_temp("gear_box_speed")

# plot_wind(wind_speed_data)
a = [x/100.0 for x in range(400,1457,7)]+[x/100.0 for x in range(1457,1625,14)]+[x/100.0 for x in range(1625,1737,28)]+[x/100.0 for x in range(1737,1937,50)]+[19.37,26.65]
# a = [x/100.0 for x in range(400,2665,20)]
b = a[1:]
a = a[0:-1]
# c = []
# for (i,j) in zip(a,b):
#     c.append(len(wind_speed_data.query("wind_speed>%f and wind_speed<=%f"%(i,j))))
# for i in range(len(c)):
#     if c[i]<10000:
#         print zip(a,b)[i],c[i],i
        
wind_speed_rule = wind_creat_rules(a,b,wind_speed_data)
wind_speed_rule.to_excel("wind_gear_box_speed1.xlsx")

# plot_ambient(ambient_temp_data)
# a =[x for x in range(-20,39,1)]
a = [-20,-16]+[x for x in range(-15,36)]+[36,39]
b = a[1:]
a = a[0:-1]
# c = []
# for (i,j) in zip(a,b):
#     c.append(len(ambient_temp_data.query("ambient_temp>%f and ambient_temp<=%f"%(i,j))))       
# for i in range(len(c)):
#     if c[i]<10000:
#         print zip(a,b)[i],c[i],i
        
ambient_temp_rule = ambient_creat_rules(a,b,ambient_temp_data)
ambient_temp_rule.to_excel("ambient_gear_box_speed1.xlsx")
