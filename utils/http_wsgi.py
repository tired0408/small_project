from wsgiref.simple_server import make_server
import urllib.parse
from html import escape
import json
import threading
import requests
import traceback

class HttpServer(object):
    def __init__(self):
        # 创建server实例
        # host写成0.0.0.0的话，其他人可以访问，代表监听多块网卡，127.0.0.1表示监听本地, port可以指定端口
        self.my_server = make_server(host='0.0.0.0', port=5902, app=self.app)

    def server_run(self):
        """
        服务端，
        :return:
        """
        # 启动server监听，监听http请求
        print("Serving HTTP on port 5902...")
        t = threading.Thread(target=self.my_server.serve_forever)
        t.start()

    def app(self, environ, start_response):
        """
        定义应用，根据url区分不同的功能模块
        :param environ:
        :param start_response:
        :return:
        """
        status = '200 OK' # HTTP响应状态
        # HTTP响应头，注意格式, 可根据响应内容设置为application/json等
        response_header = [('Content-type', 'text/html;charset=utf-8')]
        # 将响应状态和响应头交给WSGI server
        start_response(status, response_header)
        # 判断请求url地址
        path_info = environ['PATH_INFO']
        if path_info == "/api":
            return_data = self.api(environ)
            print("return the data: %s" % str(return_data))
            return [json.dumps(return_data).encode('utf-8')]  # 返回响应正文
        else:
            print("Request url error!!!")
            return [b"<h1>Not has this function</h1>"]

    @ staticmethod
    def api(req):
        """
        判断请求方式，获取参数后执行功能函数
        :param req:
        :return:
        """
        # 判断http的请求方式
        req_method = req.get("REQUEST_METHOD")
        if req_method == "GET": # get请求的获取方式
            params = urllib.parse.parse_qs(req['QUERY_STRING'])
            return_data = {}
            for key, value in params.items():
                return_data[key] = value[0]
            return return_data
        else:
            assert req_method== "POST"
            request_body_size = int(req.get('CONTENT_LENGTH', 0))
            request_body = req['wsgi.input'].read(request_body_size)
            content_type = req.get("CONTENT_TYPE")  # 获取请求体数据类型
            return_data = None
            if content_type == 'application/x-www-form-urlencoded':  # 浏览器的原生 form 表单
                return_data = {}
                params_byte = urllib.parse.parse_qs(request_body)  # 返回的dict中的键值对类型 {bytes:[bytes,..]}
                for key, value in params_byte.items():
                    key = key.decode("utf-8")
                    value = value[0].decode("utf-8")
                    # escape总是对用户输入进行转义来避免脚本注入。
                    value = escape(value)
                    return_data[key] = value
                camera_id = escape(params_byte.get(bytes('camera_id', encoding="utf-8"), ["".encode("utf-8")])[0].decode("utf-8"))
            else:
                json.loads(request_body)
            return return_data

    @staticmethod
    def client_send():
        """
        HTTP客户端，向服务端发送请求
        :return:
        """
        data_dict = {
            "GateAct": {
                "container_detection_sys": "container_info"
            },
            "AdditionalInfo": {
                "area_no": "1",
                "lane": "1",
                "timestamp": "146868630098"
            }
        }
        # data_dict = {
        #         "area_no": "1",
        #         "lane": "1",
        #         "timestamp": "146868630098"
        #     }
        url = r"http://172.16.115.18:15902/xxxx"
        try:
            headers = {'content-type': 'application/json'}
            res = requests.post(url, data=data_dict, headers=headers)
            print(res.text)
        except:
            print(traceback.format_exc())
if __name__ == '__main__':
    http_server = HttpServer()
    http_server.server_run()
