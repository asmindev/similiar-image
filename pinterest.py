import urllib.parse
import requests
import cv2
import numpy as np
import os
import mtcnn
from typing import Literal, Text, List, Dict, Union, Optional

class Pinterest:
    def __init__(self, query:Text,
                 max_count: int=100,
                 detect_face:bool=False, use_mtcnn=False):
        self.__query = query
        self.__raw_data = None
        self.__max_count = max_count
        self.__count_downloaded = 0
        self.__detect_face = detect_face
        self.__use_mtcnn = use_mtcnn
        self.__path = f"static/img/dataset-image/{query}"
        self.__url = "https://id.pinterest.com/resource/BaseSearchResource/get/"
        self.__bookmark = "Y2JVSG81V2sxcmNHRlpWM1J5VFVaU1YxcEhSbFJTYTNBd1ZGWmtSMVV3TVZkV1dHaFhVa1ZLVkZreU1WSmtNREZXVm14d1RrMXRhRkJXYlhCSFkyMVJlRlZZWkdGU1ZGWnlWRlZTVjAxR1dYbE5TR2hWVFZWd1IxWXlOVk5XVjBwSFUyNXNZVlpXY0ROV01GcExaRWRXUjJOSGVHaGxiRmwzVm10YWIyUXlSWGxTYTJScVUwWktWbFpyVlRGVU1WWnlWbTVLYkdKR1NscFpNR1F3WVZaYWRHVkdiRlpOYWtaNlYxWmFZVkl5VGtsVmJHaHBVbXR3TmxkWGVGWk5WMDVYWVROd1lWSlViRmhVVldSNlpVWmFSMWt6YUZWTmExcElXVEJhVjFadFJuUmhSbHBhVmtWYWFGWXhXbmRqYkVwVllrWkdWbFpFUVRWYWExcFhVMWRLTmxWdGVGTk5XRUpIVmxSSmVHTXhVbk5UV0doWVltdEtWbGxyWkZOVVJteFdWbFJHV0ZKck5UQlVWbVJIVmpGS2NtTkVRbGRTUlZwVVdUSnpNVlpyT1ZaV2JGSllVMFZLVWxadGRHRlNhekZYVld4YVlWSnJjSE5WYkZKWFUxWlZlVTFJYUZWaVJuQkhWbTF3VjFkSFNrZFRhMDVoVmpOTk1WVXdXa3RrUjBaR1RsZDRhRTFJUWpSV2Frb3dWVEZKZVZKc1pHcFNiRnBYVm10YVMxVldWbkphUms1cVlrWktNRmt3Vmt0aVIwWTJVbTVvVm1KR1NrUldNbk40WTJzeFJWSnNWbWhoTTBKUlYxZDRWbVZIVWtkWGJrWm9VbXhhYjFSV1duZFhiR1IwWkVWYVVGWnJTbE5WUmxGNFQwVXhObFpVVmxwTmEydDVWRlZTY2sxck5WVmhlazVhWWxaRk1WUlhjRTVOVm14WVVsUkdUbEl3YkRSVVZsSkhZVEExTmxKdGJFOVdSbXcwVjFaU1ZrMUdjRWhTV0doaFVrWkZlVlJ0Y0ZkaVZteFlWbTE0VG1Gck1IaFhWekZHWlZVMVZWVnRiRkJXTUZwdlZHNXdiMkZWTVhWbFJUbFRWbTFSTkdaRVNYbE9SRkV4V1hwck5VNTZaek5aTWtwdFRrZFJkMDlFVVROWmFtZDRXbFJqZWsweVNYaGFha3B0VFcxTk1sbFVVVE5PUjBWNFdYcFplazFFUVRCT1IwcHJXVzFOTkZsdFdUTmFSMWt5VGtSSmQxcHFaRGhVYTFaWVprRTlQUT09fFVIbzVhMkZFV1RSVE1sWnJUREF4Y1ZSVU1XWk9NVGgwVFZoM01VMTZXbXhOZWxreFdrUmpORnBxVW1wUFYwa3dUbnBvYTFsVVRtbFBSMGwzV20xR2FVNTZXVEZhYW14b1dtcEtiVTVIVlRKTk1rMHdUVlJSTVUweVNURk9WMVY1VGtkRk5GcFhSVEpaTWtWNlRucEpNV1pGTlVaV00zYzl8Nzk1YTQ2N2JiZjIwNDQ5NTdlOTJlZTY0YmZmMjQyZGEyZDRjMTVlNTg2MzBjNzNhNTAxMmNhNmRlNWIwMzcwYnxORVd8"
    def __encode_params(self):
        params = {
            'source_url': ['/search/pins/?rs=typed&q=' + self.__query],
            'data': [
                '{"options":{"article":"","appliedProductFilters":"---","query":"%s","scope":"pins","top_pin_id":"","filters":"","bookmarks":["%s"],"no_fetch_context_on_resource":true},"context":{}}' % (self.__query, self.__bookmark)
            ]
        }
        params = urllib.parse.urlencode(params, doseq=True)
        return params
    def __get_images(self):
        raw_images = self.__raw_data["resource_response"]["data"]["results"]
        return list(map(lambda image: image["images"]["474x"]["url"], raw_images))
    def __getNewBookmark(self):
        if self.__raw_data:
            bookmark = self.__raw_data["resource_response"]["bookmark"]
        return bookmark
    def __download_image(self, url):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            filename = f"{self.__path}/{self.__query}_{self.__count_downloaded}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
    def __download_image_with_face_detection(self, url):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image = np.asarray(bytearray(response.content), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if img is not None:
                if self.__use_mtcnn:
                    face_detector = mtcnn.MTCNN()
                    faces = face_detector.detect_faces(img)
                    if len(faces) > 0:
                        for index, face in enumerate(faces):
                            x, y, width, height = face['box']
                            face_img = img[y:y+height, x:x+width]
                            face_img = cv2.resize(face_img, (128, 128))
                            cv2.imwrite(f"{self.__path}/{self.__query}_{self.__count_downloaded}_{index}.jpg", face_img)
                else:
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        for index, (x, y, w, h) in enumerate(faces):
                            face_img = img[y:y+h, x:x+w]
                            face_img = cv2.resize(face_img, (128, 128))
                            cv2.imwrite(f"{self.__path}/{self.__query}_{self.__count_downloaded}_{index}.jpg", face_img)

    def __download_images(self, bookmark=None):
        if bookmark:
            self.__bookmark = bookmark
        params = self.__encode_params()
        response = requests.get(self.__url, params=params)
        self.__raw_data = response.json()
        images = self.__get_images()
        for image in images:
            if self.__count_downloaded == self.__max_count:
                break
            self.__count_downloaded += 1
            print("Downloading image %s      " % self.__count_downloaded, end="\r")
            if self.__detect_face:
                self.__download_image_with_face_detection(image)
            else:
                self.__download_image(image)
        bookmark = self.__getNewBookmark()
        if self.__count_downloaded < self.__max_count:
            self.__download_images(bookmark)
    def merge_folder(self, folder):
        path = f"static/img/dataset-image/{folder}"
        if os.path.exists(path):
            self.__path = path
            print(f"Folder {folder} found, merging to {self.__path}")
        else:
            print(f"Folder {folder} not found, use default folder {self.__path}")
    def run(self):
        os.makedirs(self.__path, exist_ok=True)
        self.__download_images()
        print(f"Downloaded {self.__count_downloaded} images")

if __name__ == "__main__":
    pin = Pinterest("Naruto", max_count=100)
    # pin.merge_folder("angry ma")
    pin.run()
