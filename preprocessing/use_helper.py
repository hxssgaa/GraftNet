import tensorflow as tf
import tensorflow_hub as hub
from time import time

MODULE_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'


class UseVector(object):
    def __init__(self, module_url=MODULE_URL):
        self._module_url = module_url
        self._create_embeddings = self.create_embedding()

    def _init_graph(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            t = time()
            self._text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module(self._module_url, trainable=True)
            self._embedded_text = embed(self._text_input)
            self._init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        self._graph.finalize()
        print('Time for initialize graph: ', time() - t)

    def create_embedding(self):
        self._init_graph()
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        t = time()
        session = tf.compat.v1.Session(graph=self._graph, config=config)
        session.run(self._init_op)
        print('Time for init session: ', time() - t)
        while True:
            text = (yield)
            result = session.run(self._embedded_text, feed_dict={self._text_input: text})
            yield result

    def get_vector(self, query):
        return self.get_vectors([query])

    def get_vectors(self, sentences):
        next(self._create_embeddings)
        res = self._create_embeddings.send(sentences)
        return res


if __name__ == '__main__':
    cvt = UseVector()
    t = time()
    vec = cvt.get_vector('What is your name')
    print('time for vec1: ', time() - t)
    t = time()
    vec = cvt.get_vector(
        'Nancy Pelosi today, on @GMA, actually said that Adam Schiffty Schiff didn’t fabricate my words in a major speech before Congress. She either had no idea what she was saying, in other words lost it, or she lied. Even Clinton lover @GStephanopoulos strongly called her out. Sue her?')
    print('time for vec2: ', time() - t)
    vec = cvt.get_vector('What is your name')
    print('time for vec3: ', time() - t)
