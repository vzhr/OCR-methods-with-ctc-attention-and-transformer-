# -*- coding: utf-8 -*-
import sys, cv2
import requests
import urllib
import json
import time
import os
import glob
import re
import base64
import numpy
ft_path = '/data/notebooks/yuandaxing1/ft_lib/'
sys.path.append(ft_path)
import ft2
batch_size = 1
class SHOCR(object):
    def __init__(self, server="gpu3ss.jx.shbt.qihoo.net", detection_port=9000, recongition_port=18866):
        self.server = server
        self.detection_port = detection_port
        self.recongition_port = recongition_port
        self.detect_url, self.reco_url = [
            'http://%s:%d' % (self.server, self.detection_port),
            'http://%s:%d' % (self.server, self.recongition_port),
            ]
    def ocr(self, image_list):
        '''
        image list is opencv image list
        '''
        params = {'images' : []}
        for idx, cur_image in enumerate(image_list):
            params['images'].append({'name' : '%d.jpg' %(idx) ,
                            'content' : base64.b64encode(cv2.imencode('.png', cur_image)[1])})
        beg = time.time()
        rects = requests.post(self.detect_url, data = json.dumps(params), headers = {'content-type': 'application/json'}).json()
        image_slice = []
        for r, cur_image in zip(rects['result'], image_list):
            for idx, b in enumerate(r['rect']):
                sl = cur_image[ b[1]:b[3], b[0]:b[2]]
                data = base64.b64encode(cv2.imencode('.jpg', sl)[1])
                image_slice.append({'name' : 'slice%d.jpg' % (idx),
                                    'content': data})
        reg = requests.post(self.reco_url, data = json.dumps({'images' : image_slice}), headers = {'content-type': 'application/json'}).json()
        reg_result, idx = reg['result'], 0
        for r in rects['result']:
            for box in r['rect']:
                box.extend(reg_result[idx]['text'])
                idx+=1
        return rects
    def Run(self, input_dir, output_dir):
        if (not os.path.exists(output_dir)):
            os.makedirs(output_dir)
        self.ft = ft2.put_chinese_text(os.path.join(ft_path, 'msyh.ttf'))
        image_list = glob.glob(os.path.join(input_dir, "*.jpg"))
        for start in range(0, len(image_list), batch_size):
            try:
                cur_image_list = image_list[start :min(start+batch_size, len(image_list))]
                l = [cv2.imread(image) for image in cur_image_list]
                start = time.time()
                ocr_result = self.ocr(l)
                print('cost', (time.time() - start))
                for image, result, name in zip(l, ocr_result['result'], cur_image_list):
                    cur_image = numpy.zeros(image.shape)
                    print(result['rect'])
                    for rect in result['rect']:
                        if not rect[5]: continue
                        cur_image = cv2.rectangle(cur_image,tuple(rect[0:2]), tuple(rect[2:4]),(0,255,0),1)
                        cur_image = self.ft.draw_text(cur_image, rect[0:2], rect[5], 15, (0, 255, 0))
                    new_name = os.path.join(output_dir, os.path.splitext(os.path.basename(name))[0]+"_debug.jpg")
                    cur_image = numpy.hstack((image, cur_image))
                    cv2.imwrite(new_name, cur_image)
                    json_name = os.path.splitext(new_name)[0]+".json"
                    f = open(json_name, "wb")
                    json.dump(ocr_result['result'], f)
                    f.close()
            except Exception as e:
                print(e)



if __name__ == "__main__":
    ocr = SHOCR()
    #ocr.Run('/data/notebooks/yuandaxing1/OCR/ocr_src', '/data/notebooks/yuandaxing1/OCR/ocr_src_test')
    ocr.Run('/data/notebooks/yuandaxing1/OCR/text-detection-ctpn-master/testing/0608','/data/notebooks/yuandaxing1/OCR/text-detection-ctpn-master/testing/0608_test_result/')
    #ocr.Run('/data/notebooks/yuandaxing1/OCR/CNN_LSTM_CTC_Tensorflow/test', './test')
