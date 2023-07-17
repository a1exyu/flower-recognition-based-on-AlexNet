from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from predict import predict_flower
import os
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Wwy521..@localhost/test'
app.config['SECRET_KEY'] = 'my_secret_key_123'
db = SQLAlchemy(app)

# 定义 User 模型
class User(db.Model):
    username = db.Column(db.String(50), primary_key=True)
    password = db.Column(db.String(50))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 在数据库中查找用户
        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            # 保存当前用户
            session['username'] = username
            # 登录成功，跳转到主页
            return redirect(url_for('dashboard'))
        else:
            # 登录失败，显示错误消息
            error = 'Invalid username or password'
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 创建新用户
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        # 注册成功，跳转到登录页面
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():

    return render_template('dashboard.html')

@app.route('/flower_recognition', methods=['POST'])
def flower_recognition():
    # 创建用户目录
    username = session['username']
    upload_folder = 'uploads'
    user_folder = os.path.join(upload_folder, username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # 获取上传的图片并进行处理
    image = request.files['image']
    image_name=image.filename
    image_path = os.path.join(user_folder,image_name)
    image.save(image_path)

    # 假设识别结果为 result
    result = predict_flower(image_path)

    # 返回识别结果页面
    return render_template('result.html', result=result)

@app.route('/history')
def history():
    # 获取当前用户
    current_user = User.query.filter_by(username=session['username']).first()

    # 检查用户目录是否存在
    user_folder = os.path.join('uploads', current_user.username)
    if not os.path.exists(user_folder):
        return []

    # 遍历用户目录的文件
    history = []
    for filename in os.listdir(user_folder):
        image_path = os.path.join(user_folder, filename)
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(image_path)).strftime('%Y-%m-%d %H:%M:%S')
        result = predict_flower(image_path)
        sort = result[0]
        history.append((filename, sort, modified_time))


    return render_template('history.html', history=history)



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
