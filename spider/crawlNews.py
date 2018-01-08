# -*- coding=UTF-8 -*-
import MySQLdb
import requests
from bs4 import BeautifulSoup
import re
import sys
import time
import json
import urllib2
import urllib
from lxml import etree
reload(sys)
sys.setdefaultencoding('utf-8')

# 打开数据库
db2 = MySQLdb.connect(host="120.24.43.150",
                      user="myuser",
                      passwd="321427",
                      db="myNews",
                      charset='utf8')
cursor = db2.cursor()
# 获取当日日期
gettime=time.time()
now_time = time.strftime("%Y-%m-%d",time.localtime(gettime))
db_nowtime=int(now_time[8:])+1
db_beftime=int(time.strftime("%Y-%m-%d",time.localtime(gettime-24*60*60*10))[8:])# 十天前日期去0

cursor.execute("DELETE FROM news WHERE date = %d"%(db_beftime))

session = requests.Session()

flag = 1
iterator = 0
# 北航
buaa_time="["+now_time+"]"
print "北航"
while flag == 1:
    if iterator == 0:
        url = "http://news.buaa.edu.cn/zhxw/index.htm"
    else:
        url = "http://news.buaa.edu.cn/zhxw/index" + str(iterator) + ".htm"
    print "迭代"+str(iterator)
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    a_tag = soup.find_all(href=re.compile("^\d+\.htm$"))
    for i in a_tag:
        span_siblings = i.next_siblings
        check = 0
        for sibling in span_siblings:
            if sibling is not None and sibling.string is not None and len(sibling.string) > 11 and sibling.string[0:12] == buaa_time:
                check = 1
            elif sibling is not None and sibling.string is not None and len(sibling.string) > 4 and (sibling.string[0:5] == buaa_time[0:5]\
                                                                or sibling.string[0:5] == buaa_time[0]+str(int(buaa_time[1:5])-1)):
                check = 2
        if check == 1:
            bhstr = "http://news.buaa.edu.cn/zhxw/" + i.get('href')
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("北京航空航天大学", i.string.strip(), bhstr,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (i.string.strip(), "    http://news.buaa.edu.cn/zhxw/" + i.get('href'))
        elif check == 2:
            flag = 0
    if a_tag is None or len(a_tag) == 0:
        flag = 0
    iterator += 1

# 人大
flag = 1
iterator = 0
ruc_time=now_time
print "人大"
while flag == 1:
    if iterator == 0:
        url = "http://news.ruc.edu.cn/archives/category/important_news"
    else:
        url = "http://news.ruc.edu.cn/archives/category/important_news/page/" + str(iterator+1)
    print "迭代"+str(iterator)
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    a_tag = soup.find_all(href=re.compile("^http://news\.ruc\.edu\.cn/archives/\d+$"))#侧面的部分也给爬到了
    for i in a_tag:
        span_siblings = i.next_siblings
        check = 0
        for sibling in span_siblings:
            if sibling is not None and sibling.string is not None and len(sibling.string) > 9 and sibling.string[0:10] == ruc_time:
                check = 1
            elif sibling is not None and sibling.string is not None and len(sibling.string) > 3 and (sibling.string[0:4] == ruc_time[0:4]\
                                                                                                     or sibling.string[0:4] == str(int(ruc_time[0:4])-1)):
                check = 2
        if check == 1:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)', ('人民大学', i.string.strip(), i.get('href'),str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (i.string.strip(), "    " + i.get('href'))

        elif check == 2:
            flag = 0
    if a_tag is None or len(a_tag) == 0:
        flag = 0
    iterator += 1
# 上交
sjtu_time="("+now_time+")"
print "上交"
url = "http://news.sjtu.edu.cn/jdyw.htm"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
a_tag = soup.find_all(href=re.compile("^info/1002/\d+\.htm$"))
for i in a_tag:
    sjtu_td_parent = i.parent
    sjtu_title=i.get("title")
    sjtu_href="http://news.sjtu.edu.cn/"+i.get('href')
    td_siblings = sjtu_td_parent.next_siblings
    check = 0
    if td_siblings is not None:
        for sibling in td_siblings:
            if sibling is not None and sibling.string is not None and len(sibling.string) > 11 and sibling.string[0:12] == sjtu_time:
                 check = 1
        if check == 1:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('上海交通大学', sjtu_title.strip(), sjtu_href,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (sjtu_title.strip()+ "    " + sjtu_href)

# 同济
# 获取前一天日期
yesterday_time = time.strftime("%Y-%m-%d", time.localtime(time.time() - 24 * 60 * 60))
tongji_now = u"(今"
tongji_time = "(" + yesterday_time[5:7] + u"月" + yesterday_time[8:10] + u"日" + ")"
tongji_lastyear = "(" + str(int(now_time[0:4]) - 1) + "-"
flag = 1
iterator = 0
print "同济"
while flag == 1:
    if iterator == 0:
        url = "http://news.tongji.edu.cn/classid-6.html"
    else:
        url = "http://news.tongji.edu.cn/classid-6-" + str(iterator + 1) + ".html"
    print "迭代" + str(iterator)
    s = session.get(url)
    tongji_soup = BeautifulSoup(s.content, 'html.parser')
    a_tag = tongji_soup.find_all(href=re.compile("^classid-6-newsid-\d+-t-show\.html$"))
    for i in a_tag:
        span_siblings = i.next_siblings
        check = 0
        for sibling in span_siblings:
            if sibling is not None and sibling.string is not None and len(
                    sibling.string) > 2 and sibling.string[0:2] \
                    == tongji_now:
                check = 1
            elif sibling is not None and sibling.string is not None and (
                (len(sibling.string) > 3 and sibling.string[3] == tongji_time[3]) or (
                    len(sibling.string) > 4 and sibling.string[0:5] == tongji_lastyear)):
                check = 2
        if check == 1:
            tjstr = "http://news.tongji.edu.cn/" + i.get('href')
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('同济大学', i.string.strip(), tjstr,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (i.string.strip(), "    http://news.tongji.edu.cn/" + i.get('href'))
        elif check == 2:
            flag = 0
    if a_tag is None or len(a_tag) == 0:
        flag = 0
    iterator += 1

# 北外
bfsu_time = now_time
flag = 1
iterator = 0
print "北外"
while flag == 1:
    if iterator == 0:
        url = "http://news.bfsu.edu.cn/archives/category/bwsx"
    else:
        url = "https://news.bfsu.edu.cn/archives/category/bwsx/page/" + str(iterator+1)
    print "迭代"+str(iterator)
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    a_tag = soup.find_all(href=re.compile("^http://news.bfsu.edu.cn/archives/\d+$"))
    for i in a_tag:
        span_siblings = i.next_siblings
        check = 0
        for sibling in span_siblings:
            if sibling is not None and sibling.string is not None and len(sibling.string) > 9 and sibling.string[0:10] == bfsu_time:
                check = 1
            elif sibling is not None and sibling.string is not None and len(sibling.string) > 3 and (sibling.string[0:4] == bfsu_time[0:4]\
                                                                or sibling.string[0:4] == str(int(bfsu_time[0:4])-1)):
                check = 2
        if check == 1:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('北京外国语大学', i.string.strip(), i.get('href'),str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (i.string.strip(), "    " + i.get('href'))
        elif check == 2:
            flag = 0
    if a_tag is None or len(a_tag) == 0:
        flag = 0
    iterator += 1
# 北师
bnu_time=now_time
flag = 1
iterator = 0
print "北师"
while flag == 1:
    if iterator == 0:
        url = "http://news.bnu.edu.cn/xwzh/yxxw/index.htm"
    else:
        url = "http://news.bnu.edu.cn/xwzh/yxxw/index" + str(iterator) + ".htm"
    print "迭代"+str(iterator)
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    a_tag = soup.find_all(href=re.compile("^\d+\.htm$"))
    for i in a_tag:
        bnu_url ="http://news.bnu.edu.cn/xwzh/yxxw/" + i.get('href')
        check = 0
        bnu_s=session.get(bnu_url)
        bnu_soup=BeautifulSoup(bnu_s.content,'html.parser')
        bnu_tag = bnu_soup.find_all(text=bnu_time)
        if bnu_tag is None or len(bnu_tag) == 0:
            check = 2
        if check == 0:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('北京师范大学', i.string.strip(), bnu_url,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (i.string.strip(), "   "+bnu_url)
        elif check == 2:
            flag = 0
    if a_tag is None or len(a_tag) == 0:
        flag = 0
    iterator += 1
# 央财
cufe_time = now_time[0:10]
print "央财"
url = "http://news.cufe.edu.cn/"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
a_tag = soup.find_all(href=re.compile("^\w{4}/\d{5}\.htm$"))
for i in a_tag:
    cufe_url ="http://news.cufe.edu.cn/" + i.get('href')
    check = 0
    cufe_sibling = i.parent.span
    if cufe_sibling is None:
        check = 2
    elif cufe_sibling.string == cufe_time:
        check = 1
    if check == 1:
        cufe_s = session.get(cufe_url)
        cufe_soup=BeautifulSoup(cufe_s.content,'html.parser')
        cufe_tag=cufe_soup.find_all('h3')
        for j in cufe_tag:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('中央财经大学', j.string.strip(), cufe_url,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (j.string.strip(), "   " + cufe_url)
    elif check == 2:
        flag = 0
if a_tag is None or len(a_tag) == 0:
    flag = 0
iterator += 1
# 北理
bit_time="("+now_time+")"
flag = 1
iterator = 0
print "北理"
while flag == 1:
    if iterator == 0:
        url = "http://www.bit.edu.cn/xww/xwtt/index.htm"
    else:
        url = "http://www.bit.edu.cn/xww/xwtt/index" + str(iterator) + ".htm"
    print "迭代"+str(iterator)
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    a_tag = soup.find_all(href=re.compile("^\.\./\w+/\d+\.htm$"))
    for i in a_tag:
        span_siblings = i.next_siblings
        check = 0
        for sibling in span_siblings:
            if sibling is not None and sibling.string is not None and len(sibling.string) > 11 and sibling.string[0:12] == bit_time:
                check = 1
            elif sibling is not None and sibling.string is not None and len(sibling.string) > 4 and (sibling.string[0:5] == bit_time[0:5]\
                                                                or sibling.string[0:5] == bit_time[0]+str(int(bit_time[1:5])-1)):
                check = 2
        if check == 1:
            bit_url=i.get("href")
            bl_str = "http://www.bit.edu.cn/xww" + bit_url[2:]
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('北京理工大学', i.string.strip(), bl_str,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print (i.string.strip(), "    http://www.bit.edu.cn/xww/" + bit_url[2:])
        elif check == 2:
            flag = 0
    if a_tag is None or len(a_tag) == 0:
        flag = 0
    iterator += 1

# 北京大学
session = requests.Session()
url = "http://pkunews.pku.edu.cn/xwzh/xwzh.htm"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
t = time.strftime("%Y-%m/%d", time.localtime(time.time()))
out = soup.find_all(href=re.compile(t))
for i in out:
    j = str(i)
    bdStr = "http://pkunews.pku.edu.cn/xwzh/" + i.get('href')
    try:
        cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('北京大学', i.string, bdStr,str(db_nowtime)))
        db2.commit()
    except:
        db2.rollback()
    print i.string, "    http://pkunews.pku.edu.cn/xwzh/" + i.get('href')

# 南京大学
t = time.strftime("%Y-%m-%d", time.localtime(time.time()))
url = "http://news.nju.edu.cn/list_1.html"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
out = soup.find_all(href=re.compile("show_article"))
for i in out:
    siblings = i.previous_siblings
    for sibling in siblings:
        if sibling.string == t:
            njStr = "http://news.nju.edu.cn/" + i.get('href')
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('南京大学', i.string, njStr,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print i.string + "    http://news.nju.edu.cn/" + i.get('href')

# 浙江大学
t = time.strftime("[%Y-%m-%d]", time.localtime(time.time()))
session = requests.get("http://www.zju.edu.cn/c20968/catalog.html")
session.encoding = 'gbk'
soup = BeautifulSoup(session.content, 'html.parser', from_encoding='gbk')
out = soup.find_all(href=re.compile("content"))
for i in out:
    siblings = i.next_siblings
    for sibling in siblings:
        if sibling.string == t:
            zjStr = "http://www.zju.edu.cn" + i.get('href')
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('浙江大学', i.get('title'), zjStr,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print i.get('title') + "    http://www.zju.edu.cn" + i.get('href')

# 武汉大学
session = requests.Session()
url = "http://news.whu.edu.cn/index.htm"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
t = time.strftime("%m-%d", time.localtime(time.time()))
t2 = time.strftime("[%Y-%m-%d]", time.localtime(time.time()))
out = soup.find_all(href=re.compile("info"))
for i in out:
    siblings = i.next_siblings
    for sibling in siblings:
        if sibling.name == "span":
            if sibling.get_text().strip() == t or sibling.get_text().strip() == t2:
                whStr = "http://news.whu.edu.cn/" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('武汉大学', i.string, whStr,str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string + "    http://news.whu.edu.cn/" + i.get('href')

# 西安交大
t = time.strftime("[%Y-%m-%d", time.localtime(time.time()))
session = requests.get("http://news.xjtu.edu.cn/zhxw.htm")
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(href=re.compile("info"))
for i in out:
    siblings=i.next_siblings
    for sibling in siblings:
        if sibling.string[:11] == t:
            if i.get('title') is not None:
                xjdStr = "http://news.xjtu.edu.cn/" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('西安交大', i.get('title'), xjdStr,str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.get('title') + "    http://news.xjtu.edu.cn/" + i.get('href')

# 中南大学
session = requests.get("http://news.csu.edu.cn/zhxw.htm")
t = time.strftime(u"[%Y年%m月%d日]", time.localtime(time.time()))
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(href=re.compile("info"))
for i in out:
    siblings = i.next_siblings
    for sibling in siblings:
        if sibling.string.strip() == t:
            znStr = "http://news.csu.edu.cn/" + i.get('href')
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)', ('中南大学', i.string, znStr,str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print i.string+"    http://news.csu.edu.cn/" + i.get('href')

# 哈工大
session = requests.Session()
url = "http://news.hit.edu.cn/xxyw/list.htm"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
t = time.strftime("%Y-%m-%d", time.localtime(time.time()))
out = soup.find_all(class_=re.compile("fields ex_fields"))
for i in out:
    k = i.children
    for kc in k:
        if kc.string == t:
            ip = i.previous_siblings
            for ipc in ip:
                if ipc is not None and ipc.string is None:
                    ipcc = ipc.children
                    for ipccs in ipcc:
                        if ipccs.string is None and ipccs.children is not None:
                            for ipccss in ipccs:
                                if ipccss.string is not None and ipccss.get('href')is not None:
                                    hgdStr = "http://news.hit.edu.cn/" + ipccss.get('href')
                                    try:
                                        cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)',
                                                       ('哈工大', ipccss.string, hgdStr,str(db_nowtime)))
                                        db2.commit()
                                    except:
                                        db2.rollback()
                                    print ipccss.string + "    http://news.hit.edu.cn/" + ipccss.get('href')

# 东南大学
session = requests.Session()
url = "http://news.seu.edu.cn/5486/list.htm"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
t = time.strftime("%Y-%m-%d", time.localtime(time.time()))
out = soup.find_all(href=re.compile("page.htm"))
for i in out:
    ipa = i.parent
    ipan = ipa.next_siblings
    for ipans in ipan:
        if ipans is not None and ipans.string is not None and ipans.string.strip() != '':
            if ipans.string == t:
                dnStr = "http://news.seu.edu.cn" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)',
                                   ('东南大学', i.string, dnStr,str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string+"    http://news.seu.edu.cn" + i.get('href')

# 牛培敏
def http_post(url, postdata):
    encode_data = urllib.urlencode(postdata)
    req = urllib2.Request(url=url, data=encode_data)
    resp = urllib2.urlopen(req)
    return resp.read()

def http_post1(url, postdata):
    encode_data = urllib.urlencode(postdata)
    req = urllib2.Request(url=url, data=encode_data)
    resp = urllib2.urlopen(req)
    return resp.read().decode("gbk").encode("utf-8")

session = requests.Session()
# 清华
url = "http://news.tsinghua.edu.cn/publish/thunews/newsCollections/d_today.json"
response = urllib2.urlopen(url)
html = response.read()
array = json.loads(html)['data']
today = -1
for day in array:
    today = max(today, int(day))
todayNews = array[str(today)]
todayNews.reverse()
print "\n清华大学今日新闻：\n"
for item in todayNews:
    qhStr = "http://news.tsinghua.edu.cn/" + item['htmlurl'][:-5]+"_.html"
    try:
        cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)',
                       ('清华大学', item['title'], qhStr,str(db_nowtime)))
        db2.commit()
    except:
        db2.rollback()
    print(item['title'] + "    http://news.tsinghua.edu.cn/" + item['htmlurl'][:-5]+"_.html")

# 复旦
url = "http://news.fudan.edu.cn/news/"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
t = time.strftime('%Y/%m%d', time.localtime(time.time()))
out = soup.find_all(href=re.compile(t))
print "\n复旦大学今日新闻：\n"
for i in out:
    if i.string is not None:
        fdStr = "http://news.fudan.edu.cn/" + i.get('href')
        try:
            cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)',
                           ('复旦大学', i.string, fdStr,str(db_nowtime)))
            db2.commit()
        except:
            db2.rollback()
        print i.string, "    http://news.fudan.edu.cn/" + i.get('href')


# 南开
url = "http://news.nankai.edu.cn/"
s = session.get(url)
s.encoding = 'gbk'
soup = BeautifulSoup(s.content, 'html.parser', from_encoding='gbk')
t = time.strftime('%Y/%m/%d', time.localtime(time.time()))
out = soup.find_all(href=re.compile(t))
print "\n南开大学今日新闻：\n"
for i in out:
    if i.string is not None:
        try:
            cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)',
                           ('南开大学', i.string, i.get('href'),str(db_nowtime)))
            db2.commit()
        except:
            db2.rollback()
        print i.string, "    " + i.get('href')

# 天津大学
t = time.strftime('%Y%m%d', time.localtime(time.time()))
# d=datetime.datetime.today()-datetime.timedelta(days=1)
# t=d.strftime('%Y%m%d')
print "\n天津大学今日新闻：\n"
for num in range(0, 3):
    if num == 0:
        url = "http://news.tju.edu.cn/zx/qb/index.htm"
    else:
        url = "http://news.tju.edu.cn/zx/qb/index" + str(num) + ".htm"
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    out = soup.find_all(href=re.compile(t))
    for i in out:
        if i.string is not None:
            if '../../' in i.get('href'):
                tdStr = "http://news.tju.edu.cn/" + i.get('href')[6:]
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ('天津大学', i.string, tdStr,str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string, "    http://news.tju.edu.cn/" + i.get('href')[6:]
            elif '../' in i.get('href'):
                tdStr = "http://news.tju.edu.cn/zx/" + i.get('href')[3:]
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s,%s)',
                                   ('天津大学', i.string, tdStr,str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string, "    http://news.tju.edu.cn/zx/" + i.get('href')[3:]
            else:
                tdStr = "http://news.tju.edu.cn/zx/qb/" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ('天津大学', i.string, tdStr,str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string, "    http://news.tju.edu.cn/zx/qb/" + i.get('href')

# 中科大
url = "http://news.ustc.edu.cn/"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
t = time.strftime('%Y%m%d', time.localtime(time.time()))
out = soup.find_all(href=re.compile(t))
print "\n中国科技大学今日新闻：\n"
for i in out:
    if i.string is not None:
        zkdStr = "http://news.ustc.edu.cn/" + i.get('href')
        try:
            cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                           ('中科大', i.string, zkdStr, str(db_nowtime)))
            db2.commit()
        except:
            db2.rollback()
        print i.string, "    http://news.ustc.edu.cn/" + i.get('href')

# 山东大学
t = time.strftime('%Y-%m-%d', time.localtime(time.time()))
url = "http://www.view.sdu.edu.cn/ssjgy.jsp?wbtreeid=1001"
body = http_post(url=url,
                 postdata={
                     'newskeycode2': 'ICA=',
                     '_lucenesearchtype': 2,
                     'topageurl': '/ssjgy.jsp?wbtreeid=1001&searchScope=1&currentnum=',
                     'wbtreeid': 1001
                 })
response = etree.HTML(text=body)
items = response.xpath('/html/body/div[4]/div[2]/div[1]/div/ul/li')
print "\n山东大学今日新闻：\n"
for item in items:
    date = item.xpath('span/text()')
    title = item.xpath('a/text()')
    href = item.xpath('a/@href')
    if date[0] == t:
        if 'http:' in href[0]:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ('山东大学', title[0], href[0], str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0] + "      " + href[0]
        else:
            sdStr = "http://www.view.sdu.edu.cn/" + href[0]
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ('山东大学', title[0], sdStr, str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0] + "      http://www.view.sdu.edu.cn/" + href[0]

# 厦大
t = time.strftime('%Y-%m-%d', time.localtime(time.time()))
url = "http://news.xmu.edu.cn/_web/search/doSearch.do?_p=YXQ9MzkxJmQ9MTEzNyZwPTEmbT1TTiY_&locale=zh_CN&request_locale=zh_CN"
body = http_post(url=url,
                 postdata={
                     'pageIndex': 1,
                     'beginTime': t,
                     'catalog': 0,
                     'searchType': 'all',
                     'searchFilter': 1,
                     'orderTip': 1,
                     'filter_text': 1,
                     'keyword': '%',
                     'isShow': 1
                 })
response = etree.HTML(text=body)
items = response.xpath('//*[@id="search_body"]/div[1]/div/div')
print "\n厦门大学今日新闻：\n"
for item in items:
    title = item.xpath('h3/a/div/text()')
    href = item.xpath('h3/a/@href')
    try:
        cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                       ('厦门大学', title[0], href[0], str(db_nowtime)))
        db2.commit()
    except:
        db2.rollback()
    print title[0] + "      " + href[0]

#华东师范大学
url="http://news.ecnu.edu.cn/_web/search/doSearch.do?_p=YXM9NjQmdD05NiZkPTI4NyZwPTEmbT1TTiY_"
t = time.strftime('%Y-%m-%d', time.localtime(time.time()))
body = http_post(url=url,
                 postdata={
                    'pageIndex':1,
                    'catalog':0,
                    'searchType':'all',
                    'searchFilter':1,
                    'orderType':'publishTime',
                    'orderTip':1,
                    'columnPath':'false',
                    'filter_text':1,
                    'keyword':'%',
                    'isShow':1,
                 })
response=etree.HTML(text=body,parser=etree.HTMLParser(encoding='utf-8'))
items=response.xpath('//*[@id="search_body"]/div[1]/div/div')
print "\n华东师范大学今日新闻：\n"
for item in items:
    title=item.xpath('h3/a/div/@title')
    href=item.xpath('h3/a/@href')
    date=item.xpath('*/*/span[@class="meta_time"]/text()')
    if '2017-05-05' in date[0]:
        if 'http' in href[0]:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("华东师范大学", title[0], href[0], str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0]+ "      " + href[0]
        else:
            hdStr = "http://news.scut.edu.cn" + href[0]
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("华东师范大学", title[0], hdStr, str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0] + "      http://news.scut.edu.cn" + href[0]

#中国海洋大学
print "\n中国海洋大学今日新闻：\n"
for num in range(1, 3):
    url = "http://xinwen.ouc.edu.cn/search.aspx?searchtype=1&ModelId=1&nodeId=132&Keyword=_&fieldOption=keyword&page="\
          +str(num)
    body = urllib2.urlopen(url).read()
    response = etree.HTML(text=body, parser=etree.HTMLParser(encoding='utf-8'))
    items = response.xpath('//*[@id="main_right_box"]/div[4]/div[2]/li')
    for item in items:
        title = item.xpath('a/text()')
        href = item.xpath('a/@href')
        date=item.xpath('text()')
        if(len(title) and len(href) and len(date)):
            if(date[0]==' '+t):
                hyStr = "http://xinwen.ouc.edu.cn" + href[0]
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ("中国海洋大学", title[0], hyStr, str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print title[0] + "      http://xinwen.ouc.edu.cn" + href[0]

#重庆大学
t = time.strftime('[%Y-%m-%d]', time.localtime(time.time()))
print "\n重庆大学今日新闻：\n"
for num in range(1, 3):
    url = "http://news.cqu.edu.cn/newsv2/index.php?m=search&c=index&a=init&siteid=1&q=%25&page="+str(num)
    body = urllib2.urlopen(url).read()
    response = etree.HTML(text=body, parser=etree.HTMLParser(encoding='utf-8'))
    items = response.xpath('/html/body/div[4]/div[1]/div[2]/div/div')
    for item in items:
        title = item.xpath('div[2]/h5/a/text()')
        href = item.xpath('div[2]/h5/a/@href')
        date=item.xpath('div[2]/h5/text()')
        if(len(title) and len(href) and len(date) and t==date):
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("重庆大学", title[0], href[0], str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0] + "      " + href[0]

#湖南大学
url="http://news.hnu.edu.cn/e/search/result/?searchid=914"
t = time.strftime('%Y-%m-%d', time.localtime(time.time()))
body=urllib2.urlopen(url).read()
response=etree.HTML(text=body,parser=etree.HTMLParser(encoding='utf-8'))
items=response.xpath('//*[@id="search_news"]/div/h2')
print "\n湖南大学今日新闻：\n"
for item in items:
    title=item.xpath('a/text()')
    href=item.xpath('a/@href')
    if t in href[0]:
        if 'http' in href[0]:
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("湖南大学", title[0], href[0], str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0] + "      " + href[0]
        else:
            hnStr = "http://news.hnu.edu.cn" + href[0]
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("湖南大学", title[0], hnStr, str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print title[0] + "      http://news.hnu.edu.cn" + href[0]

#吉林大学
url="http://news.jlu.edu.cn/sjz_list.jsp?wbtreeid=1216"
t = time.strftime('%Y-%m-%d', time.localtime(time.time()))
body = http_post(url=url,
                 postdata={
                    'calendarselecteddate':t,
                    'year122907':t[:4],
                    'month122907':t[6:7],
                 })
response=etree.HTML(text=body,parser=etree.HTMLParser(encoding='utf-8'))
items=response.xpath('//a[@class="c122908"]')
print "\n吉林大学今日新闻：\n"
for item in items:
    title=item.xpath('@title')
    href=item.xpath('@href')
    if 'http' in href[0]:
        try:
            cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                           ("吉林大学", title[0], href[0], str(db_nowtime)))
            db2.commit()
        except:
            db2.rollback()
        print title[0] + "      " + href[0]
    else:
        jlStr = "http://news.jlu.edu.cn/" + href[0]
        try:
            cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                           ("吉林大学", title[0], jlStr, str(db_nowtime)))
            db2.commit()
        except:
            db2.rollback()
        print title[0] + "      http://news.jlu.edu.cn/" + href[0]

# 华中科技大学
url = "http://news.hustonline.net/search?type=title&keywords="
body=urllib2.urlopen(url).read()
response=etree.HTML(text=body,parser=etree.HTMLParser(encoding='utf-8'))
items=response.xpath('/html/body/div[2]/div[1]/ul/li')
t = time.strftime('%Y-%m-%d', time.localtime(time.time()))
print "\n华中科技大学今日新闻：\n"
for item in items:
    data=item.xpath('span/text()')
    if data[0]==t:
        title=item.xpath('a/text()')
        href=item.xpath('a/@href')
        try:
            cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                           ("华中科技大学", title[0], href[0], str(db_nowtime)))
            db2.commit()
        except:
            db2.rollback()
        print title[0] + "      " + href[0]

# 中山大学
session = requests.get("http://news2.sysu.edu.cn/news01/index.htm")
t = time.strftime("%Y-%m-%d", time.localtime(time.time()))
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(href=re.compile("htm"))
print u"中山大学"
for i in out:
    ip = i.previous_siblings
    for ips in ip:
        if ips.string == t:
            if i.string is not None:
                zsStr = "http://news2.sysu.edu.cn/news01/" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ("中山大学", i.string.strip(), zsStr, str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string.strip() + "  http://news2.sysu.edu.cn/news01/" + i.get('href')

# 西安电子科技大学
session = requests.get("http://news.xidian.edu.cn/yw.htm")
t = time.strftime("%Y-%m-%d", time.localtime(time.time()))
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(class_=re.compile("pc-news-bt"))
print u"西安电子科技大学"
for i in out:
    for ip in i.next_siblings:
        for ipp in ip:
            if ipp == t:
                ipa = i.parent
                ipap = ipa.parent
                xaddStr = "http://news.xidian.edu.cn/" + ipap.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ("西安电子科技大学", i.string, xaddStr, str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string + " http://news.xidian.edu.cn/" + ipap.get('href')

# 东北大学
session = requests.get("http://neunews.neu.edu.cn/campus/part/DDYW.html")
t = time.strftime("%Y-%m-%d", time.localtime(time.time()))
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(href=re.compile("html"))
print u"东北大学"
for i in out:
    ip = i.previous_siblings
    for ips in ip:
        if ips.string == t:
            if i.string is not None:
                dbStr = "http://neunews.neu.edu.cn" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ("东北大学", i.string.strip()[1:], dbStr, str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.string.strip()[1:] + " http://neunews.neu.edu.cn" + i.get('href')

# 中国石油大学
session = requests.get("http://news.upc.edu.cn/sdyw/list.htm")
t = time.strftime("[%Y-%m-%d]", time.localtime(time.time()))
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(href=re.compile("page.htm"))
print u"中国石油大学"
for i in out:
    ip = i.previous_siblings
    for ips in ip:
        if ips.string == t:
            if i.get('title') is not None:
                zsyStr = "http://news.upc.edu.cn" + i.get('href')
                try:
                    cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                                   ("中国石油大学", i.get('title'), zsyStr, str(db_nowtime)))
                    db2.commit()
                except:
                    db2.rollback()
                print i.get('title') + " http://news.upc.edu.cn" + i.get('href')

# 大连理工
session = requests.get("http://news.dlut.edu.cn/xw/zhxw.htm")
t = time.strftime(u"%Y年%m月%d日", time.localtime(time.time()))
soup = BeautifulSoup(session.content, 'html.parser')
out = soup.find_all(href=re.compile(".htm"))
print u"大连理工大学"
for i in out:
    for ip in i.next_siblings:
        if ip.string == t:
            dlStr = "http://news.dlut.edu.cn" + i.get('href')[2:]
            try:
                cursor.execute('INSERT INTO news VALUES (%s, %s, %s, %s)',
                               ("大连理工大学", i.string, dlStr, str(db_nowtime)))
                db2.commit()
            except:
                db2.rollback()
            print i.string + " http://news.dlut.edu.cn" + i.get('href')[2:]

# 把我校换成相应学校
cursor.execute("UPDATE news SET title = replace(title, '我校', news.school)")
db2.commit()
cursor.execute("UPDATE news SET title = replace(title, '学校', news.school)")
db2.commit()
cursor.execute("UPDATE news SET title=LEFT(title,(LENGTH(title)-3)/3) WHERE title LIKE '%...';")
db2.commit()

# 大数据新闻
delete = "DELETE FROM BigData WHERE date = %d"%(db_beftime)
cursor.execute(delete)
db2.commit()
url3 = "http://www.36dsj.com/archives/category/bigdata"
session = requests.session()
s3 = session.get(url3)
soup3 = BeautifulSoup(s3.content,"html.parser")
outBigData = soup3.find_all(class_=re.compile("excerpt"))
bigdataTitle = ""
bigdataHref = ""
for one in outBigData:
    if one is not None:
        bigdataFlag=1
        bigdataDiv=one.div
        if bigdataDiv is not None:
            bigdataImg=bigdataDiv.img
            if bigdataImg is None:
                bigdataFlag=0
            else:
                bigdataPic=bigdataImg['data-original']
                print bigdataPic
        bigdataH2=one.h2
        if bigdataH2 is not None and bigdataFlag==1:
            bigdataA=bigdataH2.a
            if bigdataA is None:
                bigdataFlag=0
            else:
                bigdataTitle=bigdataA['title'][:-7]
                bigdataHref=bigdataA['href']
                print bigdataTitle
                print bigdataHref
        bigdataP=one.find_all('p')
        bigdataSum=''
        if bigdataFlag==1:
            for p in bigdataP:
                if p['class'] == [u'note']:
                    bigdataString=p.string.strip()
                    if bigdataString.find('文 |') !=-1 or bigdataString.find('作者：') !=-1 or bigdataString.find('文|') !=-1 \
                            or bigdataString.find('文 |') !=-1 or bigdataString.find('制作：') !=-1:
                        bigdataIndex=bigdataString.find('\n')
                        bigdataSum=bigdataString[bigdataIndex:].replace('\n',' ')
                    else:
                        bigdataSum=bigdataString.replace('\n',' ')
        if bigdataFlag==1:
            print bigdataSum
            cursor.execute('INSERT INTO BigData(title, url, sum, date) VALUES (%s, %s, %s, %s)',
                           (bigdataTitle, bigdataHref, bigdataSum, str(db_nowtime)))
            db2.commit()

session = requests.Session()
url = "http://www.cbdio.com/node_4640.htm"
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
out = soup.find_all(href=re.compile("2017"))
out2 = soup.find_all('p', {'class': 'cb-media-summary'})
for i in out:
    s = str(i.string)
    s2 = "http://www.cbdio.com/" + i.get("href")
    if s != "None":
        s3 = str(out2[0].get_text())[0:-7]
        del out2[0]
        print(s, s2, s3)
        cursor.execute('INSERT INTO BigData(title, url, sum,date) VALUES (%s, %s, %s, %s)', (s, s2, s3,str(db_nowtime)))
        db2.commit()

url2 = "http://www.cbdio.com/node_4640_2.htm"
s2 = session.get(url2)
soup2 = BeautifulSoup(s2.content, 'html.parser')
out3 = soup2.find_all(href=re.compile("2017"))
out4 = soup2.find_all('p', {'class': 'cb-media-summary'})
for i in out3:
    s = str(i.string)
    if s != "None":
        s3 = str(out4[0].get_text())[0:-7]
        del out4[0]
        s2 = "http://www.cbdio.com/" + i.get("href")
        cursor.execute('INSERT INTO BigData(title, url, sum,date) VALUES (%s, %s, %s, %s)', (s, s2, s3,str(db_nowtime)))
        db2.commit()
        print(s)
        print("http://www.cbdio.com/" + i.get("href"))
for i2 in out4:
    print("" + str(i2.get_text())[0:-7])

cursor.close()
db2.close()
