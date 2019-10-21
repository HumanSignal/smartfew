import flask
import logging
import json
import numpy as np
import argparse
import multiprocessing as mp
import time
import os

from operator import itemgetter
from smartfew.encoders.image import ImageEncoder
from smartfew.learners.protonet import train_model, load_model, apply_model
from smartfew.providers.image import ImageURLsFromFileProvider

app = flask.Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


provider = None

input_queue = mp.Queue()
output_queue = mp.Queue()

vector_file = 'my_vector.npy'


def learning_process(input_queue):
    print('Start learning process')

    train_samples = None
    train_targets = None

    encoder = ImageEncoder()

    for images, targets in iter(input_queue.get, None):

        samples = encoder.encode(images)
        targets = np.array(targets, dtype=np.int32)

        if train_samples is None and train_targets is None:
            train_samples = samples
            train_targets = targets
        else:
            train_samples = np.vstack((train_samples, samples))
            train_targets = np.hstack((train_targets, targets))

        train_model(
            train_samples, train_targets,
            warm_start=False, num_support_per_class=4, num_query_per_class=4,
            num_episodes=100
        )
        model = load_model()
        train_encodings, unique_targets = apply_model(train_samples, model, train_targets)
        target_encoding = np.squeeze(train_encodings[np.where(unique_targets == 1)[0]])
        np.save(vector_file, target_encoding)
        print('Model learned!')


def predicting_process(output_queue, input_file):
    print('Start predicting process')
    provider = ImageURLsFromFileProvider(input_file, batch_size=50)
    encoder = ImageEncoder()
    while True:
        print('Load model...')
        model = load_model()
        if model is None or not os.path.exists(vector_file):
            time.sleep(10)
            continue
        print('Load vector')
        target_encoding = np.load(vector_file)
        print('Get images')
        new_images = next(provider)
        print('Encode images')
        new_samples = encoder.encode(new_images)
        print('Apply model')
        new_encodings = apply_model(new_samples, model)

        dist = np.sum(new_encodings ** 2, axis=1) - np.dot(new_encodings, target_encoding)
        best_idx = np.argsort(dist)[:16]
        print(f'Dist: {dist[best_idx]}')
        images = [new_images[i] for i in best_idx]
        output_queue.put((images,))
        print('Predictions finished!')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/')
def index():
    pass


@app.route('/update', methods=['POST'])
def update():
    data = json.loads(flask.request.data)
    if len(data['images']) == 0:
        provider = ImageURLsFromFileProvider(args.input)
        new_images = next(provider)
        images_to_show = [{'url': new_images[i], 'selected': False} for i in range(16)]
        return flask.jsonify({
            'images': images_to_show
        })

    images = list(map(itemgetter('url'), data['images']))
    targets = list(map(itemgetter('selected'), data['images']))
    input_queue.put((images, targets))
    print('Getting new images...')
    new_images, = output_queue.get()
    images_to_show = [{'url': image, 'selected': False} for image in new_images]
    return flask.jsonify({
        'images': images_to_show
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', dest='host', default='0.0.0.0')
    parser.add_argument('--port', dest='port', default='14321')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('-i', '--input', dest='input', help='Input file', required=True)
    args = parser.parse_args()

    if os.path.exists(vector_file):
        os.remove(vector_file)
    learning_proc = mp.Process(target=learning_process, args=(input_queue,))
    predicting_proc = mp.Process(target=predicting_process, args=(output_queue, args.input))
    learning_proc.start()
    predicting_proc.start()

    app.run(host=args.host, port=args.port, debug=args.debug)
