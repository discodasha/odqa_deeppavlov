from deeppavlov import configs, build_model
from flask import Flask
import json

app = Flask(__name__)

kbqa_model = build_model(configs.kbqa.kbqa_cq_rus, download=False)


@app.route("/question/<question>")
def get_odqa(question):
    if question[-1] != '?':
        question = question + '?'
    return kbqa_model([question])[0]


if __name__ == '__main__':
    app.run(port=8989)
