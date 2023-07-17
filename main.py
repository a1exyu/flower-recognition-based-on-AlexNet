# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import time

###################
# 模型所需库包
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
# , map_location='cpu'
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

# 关闭 Dropout
model.eval()

###################
from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# 图片装换操作
def tran(img_path):
    # 预处理
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    return img


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    path = ""
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        path = secure_filename(f.filename)
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print(basepath)
        print(path)
        print(upload_path)
        img = tran(path)
        ##########################
        # 预测图片
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img))  # 将输出压缩，即压缩掉 batch 这个维度
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            res = class_indict[str(predict_cla)]
            pred = predict[predict_cla].item()
            # print(class_indict[str(predict_cla)], predict[predict_cla].item())
        res_chinese = ""
        if res == "daisy":
            res_chinese = "雏菊"
        if res == "dandelion":
            res_chinese = "蒲公英"
        if res == "roses":
            res_chinese = "玫瑰"
        if res == "sunflower":
            res_chinese = "向日葵"
        if res == "tulips":
            res_chinese = "郁金香"

        # print('result:', class_indict[str(predict_class)], 'accuracy:', prediction[predict_class])
        ##########################
        f.save(upload_path)
        pred = pred * 100
        return render_template('upload_ok.html', path=path, res_chinese=res_chinese, pred=pred, val1=time.time())

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=80, debug=True)