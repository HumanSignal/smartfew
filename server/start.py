import flask
import logging
import json
import numpy as np
import argparse
import multiprocessing as mp
import time
import os
import glob

from operator import itemgetter
from smartfew.encoders.image import ImageEncoder
from smartfew.learners.protonet import train_model, load_model, apply_model
from smartfew.providers.image import ImageURLsFromFileProvider
from smartfew.utils.io import get_data_dir


logging.basicConfig(level=logging.DEBUG)
app = flask.Flask(__name__)
logger = logging.getLogger(__name__)

encoding_file = os.path.join(get_data_dir(), 'encoding.npy')

provider = None

input_queue = mp.Queue()
output_queue = mp.Queue()

_NUM_ITEMS_TO_SHOW = 16
_NUM_EPISODES = 1000
_BATCH_SIZE_FOR_SEARCH = 100


def learning_process(input_queue):
    logger.info(f'Learning process started with PID={os.getpid()}')

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
            warm_start=False,
            num_support_per_class=4,
            num_query_per_class=4,
            num_episodes=_NUM_EPISODES
        )
        model = load_model()
        train_encodings, unique_targets = apply_model(train_samples, model, train_targets)
        target_encoding = np.squeeze(train_encodings[np.where(unique_targets == 1)[0]])

        np.save(encoding_file, target_encoding)


def prediction_process(output_queue, input_file):
    logger.info(f'Prediction process started with PID={os.getpid()}')

    provider = ImageURLsFromFileProvider(input_file, batch_size=_BATCH_SIZE_FOR_SEARCH)
    encoder = ImageEncoder()
    while True:
        model = load_model()
        if model is None or not os.path.exists(encoding_file):
            logger.info(f'Can\'t fetch model from {encoding_file}: waiting for learning process...')
            if output_queue.empty():
                some_images = next(provider)
                output_queue.put((some_images[:_NUM_ITEMS_TO_SHOW],))
            else:
                time.sleep(10)
            continue
        target_encoding = np.load(encoding_file)
        new_images = next(provider)
        new_samples = encoder.encode(new_images)
        new_encodings = apply_model(new_samples, model)

        dist = np.sum(new_encodings ** 2, axis=1) - np.dot(new_encodings, target_encoding)
        best_idx = np.argsort(dist)[:_NUM_ITEMS_TO_SHOW]
        images = [new_images[i] for i in best_idx]
        output_queue.put((images,))


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/')
def index():
    js_includes = list(glob.glob('../ui/build/static/js/*.js'))
    css_includes = list(glob.glob('../ui/build/static/css/*.css'))
    return flask.render_template('index.html', js_includes=js_includes, css_includes=css_includes)


@app.route('/ui/build/static/<path:path>')
def serve_static(path):
    return flask.send_from_directory('../ui/build/static', path)


@app.route('/update', methods=['POST'])
def update():
    data = json.loads(flask.request.data)
    if len(data['images']) > 0:
        images = list(map(itemgetter('url'), data['images']))
        targets = list(map(itemgetter('selected'), data['images']))
        input_queue.put((images, targets))
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

    if os.path.exists(encoding_file):
        os.remove(encoding_file)
    learning_proc = mp.Process(target=learning_process, args=(input_queue,))
    predicting_proc = mp.Process(target=prediction_process, args=(output_queue, args.input))
    learning_proc.start()
    predicting_proc.start()

    app.run(host=args.host, port=args.port, debug=args.debug)
