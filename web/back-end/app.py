import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *
from processor.AIDetector_pytorch import Detector
import pymysql
#import getjson

import core.main

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'tif'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


class User(object):
    def __init__(self, username, pwd):
        self.username = username
        self.pwd = pwd

    def getUsername(self):
        return self.username

    def getPwd(self):
        return self.pwd

    def setUsername(self, username):
        self.username = username

    def setPwd(self, pwd):
        self.pwd = pwd


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        pid, image_info = core.main.c_main( #在这里使用了main.py
            image_path, current_app.model, file.filename.rsplit('.', 1)[1])
        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:5003/tmp/ct/' + pid,
                        'draw_url': 'http://127.0.0.1:5003/tmp/draw/' + pid,
                        'image_info': image_info})

    return jsonify({'status': 0})


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


@app.route('/login', methods=['POST'])
def login():
    print("aaaa")
    # 查询用户名及密码是否匹配及存在
    # 连接数据库,此前在数据库中创建数据库TESTDB
    db = pymysql.connect(host="localhost", user="root", password="12345678", database="test", charset="utf8")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    username = request.values.get("username")
    pwd = request.values.get("pwd")
    print("username", username)
    print("pwd", pwd)
    # SQL 查询语句
    sql = "SELECT * FROM user WHERE username = '%s' and pwd = '%s'" % (username, pwd)
    cursor.execute(sql)
    results = cursor.fetchall()
    print(len(results))
    if len(results) == 1:
        response = make_response("1")
        return response
    else:
        response = make_response("0")
        return response

@app.route('/regi', methods=['POST'])
def regi():
    print("regi")
    db = pymysql.connect(host="localhost", user="root", password="12345678", database="test", charset="utf8")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    username = request.values.get("username")
    pwd = request.values.get("pwd")
    print("username", username)
    print("pwd", pwd)
    # SQL 查询语句
    sql = "SELECT * FROM user WHERE username = '%s'" % username
    cursor.execute(sql)
    results = cursor.fetchall()
    print("len results ", len(results))
    if len(results) > 0:
        response = make_response("0")
        print("res ", response.response)
        return response
    #sql2 = "INSERT INTO user VALUES ('xew','123456')"
    sql2 = "INSERT INTO user(username, pwd) VALUES ('%s','%s')" % (username, pwd)
    cursor.execute(sql2)
    db.commit()
    print("sql2")
    results = cursor.fetchall()
    response = make_response("1")
    print("res ", response.response)
    return response



if __name__ == '__main__':
    files = [
        'uploads', 'tmp/ct', 'tmp/draw',
        'tmp/image', 'tmp/mask', 'tmp/uploads'
    ]
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    with app.app_context():
        current_app.model = Detector() #使用AIDetector.py
    app.run(host='127.0.0.1', port=5003, debug=True)
