from flask import Flask, render_template, request, redirect, url_for
import os
import random

from v2 import main as search
import re

app = Flask(__name__)
cari = search.ImageSearch()


def get_all_image(folder: str = None):
    files = []
    for dirnames, _, filenames in os.walk(folder):
        for filename in filenames:
            file = {
                "url": f"/{dirnames}/{filename}",
                "name": re.sub(r"\.jpg|\.png", "", filename),
            }
            files.append(file)
    return files


@app.route("/")
def index():
    images = get_all_image("./static/img/dataset-image")
    random.shuffle(images)
    return render_template("index.html", images=images[:15])


@app.route("/dog/<string:dog_name>")
def dog(dog_name):
    images = get_all_image("./static/img/dataset-image")
    dog_name = list(filter(lambda x: x["name"] == dog_name, images))
    if len(dog_name) == 0:
        return redirect(url_for("index"))
    return render_template("dog.html", dog_name=dog_name[0])


@app.post("/similarity")
def similarity():
    body = request.get_json()
    name = body["image"]
    print(name)
    data = get_all_image("./static/img/dataset-image")
    data = list(filter(lambda x: x["name"] == name, data))
    print(data)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = f"{current_dir}{data[0]['url']}"
    result = cari.search_and_get_results(path)
    # print(result)
    return {"data": result}


if __name__ == "__main__":
    app.run(debug=True)
