## Backend运行：

1.安装好所有需要的python包

2.terminal 命令（先进入back-end文件夹）：

python app.py



## Frontend运行：

terminal命令（先进入front-end文件夹）：：

npm install之后会自动出现node_modules文件夹，注意这一步有报错是非常正常的

npm run dev



## 修改sql信息：

App.py 105行和127行，db = pymysql.connect(host="localhost", user="root", password="12345678", database="test", charset="utf8")

改成你自己的数据库用户名、密码、数据库名


Sql表格：

表名user: userId主键且设置自动递增, username, pwd
