import requests
import os
import mtcnn
import cv2
import numpy as np

class UnsplashImageDownloader:
    def __init__(self, keyword, max_count=100, detect_face=False, use_mtcnn=False):
        self.__keyword = keyword
        self.__max_count = max_count
        self.__count_downloaded = 0
        self.__detect_face = detect_face
        self.__use_mtcnn = use_mtcnn
        self.__path = f"static/img/dataset-image/{keyword}"
        if self.__detect_face:
            if self.__use_mtcnn:
                print("Using MTCNN")
            else:
                print("Using Haar Cascade")
    def merger_folder(self, folder):
        if os.path.exists(f"static/img/dataset-image/{folder}"):
            self.__path = f"static/img/dataset-image/{folder}"
            print(f"Folder {folder} found, using folder {folder} as path")
        else:
            print(f"Folder {folder} not found, using folder {self.__keyword} as path")
    def unsplash_search(self, page=1, per_page=20):
        payload = {
            'query': self.__keyword,
            'per_page': per_page,
            'page': page,
            'xp': 'search-synonym:control'
        }
        uri = "https://unsplash.com/napi/search/photos"
        r = requests.get(uri, params=payload)
        data = r.json()
        results = data['results']
        return data.get('total_pages'), results

    def only_get_image_url(self, results):
        image_urls = []
        for result in results:
            image_urls.append(result['urls']['small'])
        return image_urls

    def download_image(self, url, filename):
        # get the image from url
        r = requests.get(url, allow_redirects=True).content
        image = np.asarray(bytearray(r), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if img is not None:
            if self.__detect_face:
                if self.__use_mtcnn:
                    face_detector = mtcnn.MTCNN()
                    faces = face_detector.detect_faces(img)
                    if len(faces) > 0:
                        for index, face in enumerate(faces):
                            x, y, width, height = face['box']
                            face_img = img[y:y+height, x:x+width]
                            face_img = cv2.resize(face_img, (128, 128))
                            cv2.imwrite(f"{self.__path}/{self.__keyword}_{self.__count_downloaded}_{index}.jpg", face_img)
                            self.__count_downloaded += 1
                            percent = self.__count_downloaded / self.__max_count * 100
                            print(f"Downloaded {self.__count_downloaded}/{self.__max_count} images ({percent:.2f}%)", end='\r')
                else:
                    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        for index, (x, y, width, height) in enumerate(faces):
                            face_img = img[y:y+height, x:x+width]
                            face_img = cv2.resize(face_img, (128, 128))
                            cv2.imwrite(f"{self.__path}/{self.__keyword}_{self.__count_downloaded}_{index}.jpg", face_img)
                            self.__count_downloaded += 1
                            percent = self.__count_downloaded / self.__max_count * 100
                            print(f"Downloaded {self.__count_downloaded}/{self.__max_count} images ({percent:.2f}%)", end='\r')
            else:
                cv2.imwrite(filename, img)
                self.__count_downloaded += 1
                percent = self.__count_downloaded / self.__max_count * 100
                print(f"Downloaded {self.__count_downloaded}/{self.__max_count} images ({percent:.2f}%)", end='\r')



    def download_images(self, page=1):
        total_pages, results = self.unsplash_search(page=page)
        image_urls = self.only_get_image_url(results)
        for image_url in image_urls:
            filename = f"{self.__path}/{self.__keyword}_{self.__count_downloaded}.jpg"
            self.download_image(image_url, filename)
        if self.__count_downloaded < self.__max_count:
            if total_pages > 0 and page < total_pages:
                self.download_images(page=page+1)
        print(f"\nDone with {self.__count_downloaded} images downloaded")
    def run(self):
        os.makedirs(self.__path, exist_ok=True)
        self.download_images()

# Contoh penggunaan class UnsplashImageDownloader
downloader = UnsplashImageDownloader('camel', max_count=100)
downloader.merger_folder('dog')
downloader.run()
