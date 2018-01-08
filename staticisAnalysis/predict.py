import pandas as pd
hebing=pd.read_csv('hebing.csv')

temp=0   #输入参数最小值
max_params=40  #输入参数最大值
datas=hebing['ambient_temp']
rang=[temp-0.01,max_params]
step=0.07
params=['low_ambient_temp','high_ambient_temp','gear_box_speed']

#生成表格函数
def predict(datas,rang,step,params):
    ##datas：数据集  rang：参数最小值、参数最小值    step：步长增加长度   params:参数
    num=10000000
    low_params=[]
    high_params=[]
    low_predict=[]
    mid_predict=[]
    high_predict=[]
    long_step=step
    while rang[0]<rang[1]:
        tmp=hebing[(datas>rang[0]) & (datas<rang[0]+long_step)]  
        tmp=tmp.sort(params[2])
        tmp=tmp[params[2]]
        num=len(tmp)
        while num<10000 and temp+long_step<rang[1]:            
            long_step+=step
            tmp=hebing[(datas>temp) & (datas<temp+long_step)]  
            tmp=tmp.sort(params[2])
            tmp=tmp[params[2]]
            num=len(tmp)
        low_params.append(rang[0]+0.01)
        high_params.append(rang[0]+long_step)
        low_predict.append(tmp.describe()[4])
        mid_predict.append(tmp.describe()[5])
        high_predict.append(tmp.describe()[6])
        rang[0]=rang[0]+long_step       
    #生成表格
    import pandas as pd
    ls = pd.DataFrame({'sq':low_params})
    # ls=pd.DataFrame.from_dict(tmp,orient='index').T
    ls[params[0]]=low_params
    ls[params[1]]=high_params
    ls['25%']=low_predict
    ls['50%']=mid_predict
    ls['75%']=high_predict
    ls = ls.drop('sq',axis=1)
    return ls
    # ls.to_excel('temp_gear_box_speed.xls')
    
p=predict(datas,rang,step,params)
p.to_excel('temp_gear_box_speed.xls')
print(p)
