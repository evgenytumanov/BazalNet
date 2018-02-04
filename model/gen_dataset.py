import numpy as np
import config
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, sentence):
        """encode string into ints representation
        """
        x = np.zeros(len(sentence), dtype=np.int)
        for i, c in enumerate(sentence):
            x[i] = self.char_indices[c]
        return x


    def one_hot_encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def one_hot_decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

def gen_dataset(op, op_sym, training_size, digits):
    
    # Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
    # int is DIGITS.
    maxlen = digits + 1 + digits

    # All the numbers, plus sign and space for padding.
    input_chars = '0123456789{} '.format(op_sym)
    global input_ctable
    input_ctable = CharacterTable(input_chars)

    output_chars = '0123456789 '
    global output_ctable
    output_ctable = CharacterTable(output_chars)

    questions = []
    expected = []
    seen = set()
    print('Generating data...')


    while len(questions) < training_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789'))
                        for i in range(np.random.randint(1, digits + 1))))
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Pad the data with spaces such that it is always MAXLEN.
        q = '{}{}{}'.format(a, op_sym, b)
        query = q + ' ' * (maxlen - len(q))
        ans = str(op(a, b))
        # Answers can be of maximum size DIGITS + 1.
        ans += ' ' * (maxlen - len(ans))
       
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    print('Vectorization...')
    X = np.zeros((len(questions), maxlen, len(chars)), dtype=np.float)
    y = np.zeros((len(questions), maxlen, len(chars)), dtype=np.float)
    for i, sentence in enumerate(questions):
        X[i] = input_ctable.one_hot_encode(sentence, maxlen)
    for i, sentence in enumerate(expected):
        y[i] = output_ctable.one_hot_encode(sentence, maxlen)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(X) - len(X) // 10
    (X_train, X_val) = X[:split_at], X[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print('Training Data:')
    print(X_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(X_val.shape)
    print(y_val.shape)

    return X_train, X_val, y_train, y_val

