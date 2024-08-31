import sys
sys.path.append('mrc/')
sys.path.append('/root/maojingwei579/entity_recognize_for_flight')
import tokenization
import tensorflow as tf
import modeling
import os
import collections
import six
import math
from constants_jw import PRD, ENTITY_C2E_DICT
from utils_mjw import post_discharge_records
import logging
from client import tf_serving_url
import requests, json, copy


logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer('max_count', 10, 'max count')

flags.DEFINE_integer('keep_checkpoint_max', None, 'max number of checkpoints to keep')

flags.DEFINE_string(
    "bert_config_file", "",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", '',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 300,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 50,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer(
    "n_best_size", 10,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 200,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer("rand_seed", 12345, "set random seed")

flags.DEFINE_float("no_answer_threshold", 1, "The .")

FLAGS.output_dir = os.environ.get("OUTPUT_DIR", "/root/maojingwei579/cmrc2018-master/baseline/discharge_records_0730/tf1_13_bcls")
FLAGS.predict_batch_size = int(os.environ.get("PREDICT_BATCH_SIZE", 8))
if os.path.exists(os.path.join(FLAGS.output_dir, 'vocab.txt')):
    FLAGS.vocab_file = os.path.join(FLAGS.output_dir, 'vocab.txt')
elif os.path.exists(os.path.join(FLAGS.output_dir, 'vocab_mjw.txt')):
    FLAGS.vocab_file = os.path.join(FLAGS.output_dir, 'vocab_mjw.txt')
FLAGS.bert_config_file = os.path.join(FLAGS.output_dir, 'config.json')

class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    return s



class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               input_span_mask,
               start_position=None,
               end_position=None,
               no_answer=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.input_span_mask = input_span_mask
    self.start_position = start_position
    self.end_position = end_position
    self.no_answer = no_answer


def customize_tokenizer(text, do_lower_case=False):
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
  temp_x = ""
  text = tokenization.convert_to_unicode(text)
  i = 0
  while i < len(text):
      c = text[i]
      if (text[i:i + 11]).lower() == 'mjwemptymjw':
          # import ipdb
          # ipdb.set_trace()
          temp_x += text[i:i + 11]
          i += 10
      elif tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(
              c) or tokenization._is_control(c):
          temp_x += " " + c + " "
      else:
          temp_x += " " + c + " "
      i += 1
  if do_lower_case:
    temp_x = temp_x.lower()
  return temp_x.split()


class ChineseFullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=False):
    self.vocab = tokenization.load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
    self.do_lower_case = do_lower_case
  def tokenize(self, text):
    split_tokens = []
    for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
      # if token == 'mjwemptymjw':
      #     import ipdb
      #     ipdb.set_trace()
      if token in self.vocab:
          split_tokens.append(token)
      else:
          for sub_token in self.wordpiece_tokenizer.tokenize(token):
            split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return tokenization.convert_by_vocab(self.inv_vocab, ids)


def read_squad_examples(input_file, is_training=False, is_predict=False):
  """Read a SQuAD json file into a list of SquadExample."""
  if isinstance(input_file, str):
    with tf.gfile.Open(input_file, "r") as reader:
      input_data = json.load(reader)["data"]

  elif isinstance(input_file, dict):
    input_data = input_file["data"]

  examples = []
  empty_count = 0
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=FLAGS.do_lower_case)
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True

      k = 0
      temp_word = ""
      for c in paragraph_text:
        if tokenization._is_whitespace(c):
          char_to_word_offset.append(k-1)
          continue
        # elif c in [str(i) for i in range(10)]:
        #   flag = True
        #   temp_word += c
        #   char_to_word_offset.append(k)
        #   continue
        # else:
        #   if flag:
        #     temp_word = c
        #     char_to_word_offset.append(k)
        else:
          temp_word += c
          char_to_word_offset.append(k)

        if FLAGS.do_lower_case:
          temp_word = temp_word.lower()

        if temp_word == raw_doc_tokens[k]:
          doc_tokens.append(temp_word)
          temp_word = ""
          k += 1

      assert k==len(raw_doc_tokens)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None

        if is_training or is_predict:
          answer = qa["answers"][0]
          orig_answer_text = answer["text"]

          if orig_answer_text == '':
            empty_count += 1
            start_position = 0
            end_position = 0

            example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position)

            examples.append(example) # 一段文本一般对应多个问题，一个问题一个example, 如果长度大于最大长度，那么一个example会被分成多个features
            continue

          if orig_answer_text not in paragraph_text:
            logger.warning("Could not find answer")
          else:
            answer_offset = paragraph_text.index(orig_answer_text)
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = "".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = "".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if FLAGS.do_lower_case:
              cleaned_answer_text = cleaned_answer_text.lower()
            if actual_text.find(cleaned_answer_text) == -1:
              import ipdb
              ipdb.set_trace()
              logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
              continue

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)

  logger.info("**********read_squad_examples complete!**********")

  return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, input_span_mask,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)


  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  final_hidden_cls = final_hidden[:, 0, :]
  output_weights_cls = tf.get_variable(
    'cls/squad/output_weights_cls', [2, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias_cls = tf.get_variable('cls/squad/output_bias_cls', [2], initializer=tf.zeros_initializer())
  logits_cls = tf.matmul(tf.squeeze(final_hidden_cls), output_weights_cls, transpose_b=True)
  logits_cls = tf.nn.bias_add(logits_cls, output_bias_cls)

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # apply output mask
  adder           = (1.0 - tf.cast(input_span_mask, tf.float32)) * -10000.0
  start_logits   += adder
  end_logits     += adder

  return (start_logits, end_logits, logits_cls)


def model_fn_builder(bert_config=None, init_checkpoint=None, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None, use_tpu=None,
                     use_one_hot_embeddings=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    logger.info("*** Features ***")
    for name in sorted(features.keys()):
      logger.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    input_span_mask = features["input_span_mask"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # with tf.device("/gpu:3"):
    (start_logits, end_logits, logits_cls) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        input_span_mask=input_span_mask,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    logger.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      no_answers = features['no_answers']

      def compute_loss(logits, positions):
        on_hot_pos    = tf.one_hot(positions, depth=seq_length-1, dtype=tf.float32)
        log_probs     = tf.nn.log_softmax(logits, axis=-1)
        loss          = -tf.reduce_mean(tf.reduce_sum(on_hot_pos * log_probs, axis=-1) * (1 - tf.cast(no_answers, tf.float32)))
        return loss

      start_positions = features["start_positions"]
      end_positions   = features["end_positions"]

      cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(no_answers, depth=2, dtype=tf.float32), logits=logits_cls)
      start_loss  = compute_loss(start_logits[:,1:], start_positions-1)
      end_loss    = compute_loss(end_logits[:,1:], end_positions-1)
      total_loss  = (start_loss + end_loss) / 2 + FLAGS.CLS_COEF * cls_loss

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      # output_spec = tf.estimator.EstimatorSpec(
      #     mode=mode,
      #     loss=total_loss,
      #     train_op=train_op,
      #     training_hooks=[
      #         tf.train.LoggingTensorHook([start_loss, end_loss, cls_loss, no_answers, tf.argmax(logits_cls, axis=-1)],
      #                                    every_n_iter=1, at_end=True)])

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[tf.train.LoggingTensorHook([start_loss, end_loss, cls_loss, no_answers, tf.argmax(logits_cls, axis=-1)], every_n_iter=1, at_end=True)])  # tf.one_hot(start_positions-1, depth=seq_length-1, dtype=tf.float32), tf.nn.log_softmax(start_logits[:,1:], axis=-1)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      start_logits_new = tf.nn.log_softmax(start_logits[:,1:], axis=-1)
      end_logits_new = tf.nn.log_softmax(end_logits[:,1:], axis=-1)
      no_answer_predict = tf.argmax(logits_cls, axis=-1)

      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits_new,
          "end_logits": end_logits_new,
          'no_answer': no_answer_predict,
          'logits_cls': logits_cls
      }

      # output_spec = tf.estimator.EstimatorSpec(
      #     export_outputs={"default": tf.estimator.export.PredictOutput(predictions)}, mode=mode,
      #     predictions=predictions)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn) # , prediction_hooks=[tf.train.LoggingTensorHook([start_logits, end_logits, tf.argmax(start_logits, axis=-1), tf.argmax(end_logits, axis=-1)], every_n_iter=1, at_end=True)]
          # , prediction_hooks=[tf.train.LoggingTensorHook([modeling.get_shape_list(input_ids)[0]], every_n_iter=1, at_end=True)]
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training=False, is_predict=False, drop_remainder=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_span_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["no_answers"] = tf.io.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    # batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=FLAGS.train_batch_size,
                drop_remainder=drop_remainder))
    else:
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=FLAGS.predict_batch_size,
                drop_remainder=drop_remainder))

    return d

  return input_fn


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training=False, is_predict=False,
                                 output_fn=None):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  tokenizer = ChineseFullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  out_of_span_count = 0
  empty_count = 0
  cnt_pos, cnt_neg, cnt_cont = 0, 0, 0
  if os.path.exists('temp.txt'):
    os.remove('temp.txt')
  for (example_index, example) in enumerate(examples):
    # if example.start_position == 0 and example.end_position == 0:
    #   import ipdb
    #   ipdb.set_trace()

    # if example_index >= 300:
    #   break

    if example_index % 100 == 0:
      print('Converting {}/{} pos {} neg {} cont {}'.format(
          example_index, len(examples), cnt_pos, cnt_neg, cnt_cont))
    
    
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      # if token == "A335824":
      #     import ipdb
      #     ipdb.set_trace()
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training or is_predict:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    out_of_span_flag = True
    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      input_span_mask = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      input_span_mask.append(1)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
        input_span_mask.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)
      input_span_mask.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
        input_span_mask.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)
      input_span_mask.append(0)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        input_span_mask.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(input_span_mask) == max_seq_length

      start_position = None
      end_position = None

      no_answer_flag =False
      if is_predict:
        out_of_span_flag = False
        if example.orig_answer_text == '':
          empty_count += 1
          # start_position = 0
          # end_position = 0
          no_answer_flag = True
        # else:
        #   continue

      # if is_training or is_predict:
      # import ipdb
      # ipdb.set_trace()
      if is_training:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
          out_of_span = True

        if out_of_span and example.orig_answer_text:
          cnt_cont += 1
          with open('temp.txt', 'a') as wf:
            wf.write(example.qas_id+'-'+str(doc_start)+'\n')
          out_of_span_count += 1
          continue
          start_position = 0
          end_position = 0
        elif example.orig_answer_text == '':
          cnt_neg += 1
          empty_count += 1
          feature = InputFeatures(
              unique_id=unique_id,
              example_index=example_index,
              doc_span_index=doc_span_index,
              tokens=tokens,
              token_to_orig_map=token_to_orig_map,
              token_is_max_context=token_is_max_context,
              input_ids=input_ids,
              input_mask=input_mask,
              segment_ids=segment_ids,
              input_span_mask=input_span_mask,
              start_position=2,
              end_position=5,
              no_answer=1)
          output_fn(feature)
          unique_id += 1
          continue
        else:
          cnt_pos += 1
          with open('temp.txt', 'a') as wf:
            wf.write('in span '+example.qas_id+'-'+str(doc_start)+'\n')
          out_of_span_flag = False
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          input_span_mask=input_span_mask,
          start_position=start_position,
          end_position=end_position,
          no_answer=1 if no_answer_flag else 0)

      # Run callback
      output_fn(feature)
      unique_id += 1
      

    # if out_of_span_flag:
    #   out_count += 1
    #   logg.write_print(example.qas_id+'\t'+str(out_count))

  # if not is_training:
  #   while unique_id % FLAGS.predict_batch_size != 0:
  #     feature = InputFeatures(
  #           unique_id=unique_id,
  #           example_index=example_index,
  #           doc_span_index=doc_span_index,
  #           tokens=tokens,
  #           token_to_orig_map=token_to_orig_map,
  #           token_is_max_context=token_is_max_context,
  #           input_ids=input_ids,
  #           input_mask=input_mask,
  #           segment_ids=segment_ids,
  #           input_span_mask=input_span_mask,
  #           start_position=start_position,
  #           end_position=end_position,
  #           no_answer=1 if no_answer_flag else 0)
  #     output_fn(feature)
  #     unique_id += 1

  # logg.write_print('empty count: {}'.format(empty_count))
  # print('out of span count: {}'.format(out_of_span_count))
  print('cnt pos: {}, cnt neg: {}, cnt cont: {}'.format(cnt_pos, cnt_neg, cnt_cont))


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      logger.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      logger.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      logger.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text



def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file=None,
                      output_nbest_file=None, ENTITY_C2E=None):
  """Write final predictions to the json file and log-odds of null if needed."""
  if output_prediction_file and output_nbest_file:
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  all_sentence_prediction_pairs = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    # if '住院号' in example.qas_id:
    #   import ipdb
    #   ipdb.set_trace()
    features = example_index_to_features[example_index]
    prelim_predictions = []
    
    if not features:
      continue

    no_answer_count = 0
    for (feature_index, feature) in enumerate(features):  # multi-trunk
      result = unique_id_to_result[feature.unique_id]
      if result.no_answer:
        no_answer_count += 1
        continue

      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      for start_index_ in start_indexes:
        for end_index_ in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.

          start_index = start_index_ + 1
          end_index = end_index_ + 1
          if start_index == 0 and end_index == 0:
            # import ipdb
            # ipdb.set_trace()
            # prelim_predictions.append(
            #   _PrelimPrediction(
            #       feature_index=feature_index,
            #       start_index=start_index,
            #       end_index=end_index,
            #       start_logit=result.start_logits[start_index],
            #       end_logit=result.end_logits[end_index]))
            continue

          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          # if not feature.token_is_max_context.get(start_index, False):
          #   continue 
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index_],
                  end_logit=result.end_logits[end_index_]))
    
    if no_answer_count >= len(features)/FLAGS.no_answer_threshold:
      all_predictions[example.qas_id] = ''
      continue
    
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index", "sentence"])

    seen_predictions = []
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]

      def index2final_text(start_ind, end_ind):
        tok_tokens = feature.tokens[start_ind:(end_ind + 1)]
        orig_doc_start = feature.token_to_orig_map[start_ind]
        orig_doc_end = feature.token_to_orig_map[end_ind]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        final_text = final_text.replace(' ','')
        return final_text

      if pred.start_index > 0:  # this is a non-null prediction
        final_text = index2final_text(pred.start_index, pred.end_index)
        if final_text in seen_predictions:
            continue
        seen_predictions.append(final_text)

        start_index_2, end_index_2 = pred.start_index, pred.end_index
        count = 0
        while count < FLAGS.max_count:
          if start_index_2 not in feature.token_to_orig_map:
            break
          count += 1
          start_index_2 = pred.start_index - count
        start_index_2 += 1
        count = 0
        while count < FLAGS.max_count:
          if end_index_2 not in feature.token_to_orig_map:
            break
          count += 1
          end_index_2 = pred.end_index + count
        end_index_2 -= 1
        final_sentence = index2final_text(start_index_2, end_index_2)
      else:
        # import ipdb
        # ipdb.set_trace()
        final_text = ""
        seen_predictions.append(final_text)

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
              start_index=pred.start_index,
              end_index=pred.end_index,
              sentence=final_sentence))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      import ipdb
      ipdb.set_trace()
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["start_index"] = entry.start_index
      output["end_index"] = entry.end_index
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    all_predictions[example.qas_id] = post_discharge_records(best_non_null_entry.text, ENTITY_C2E[example.qas_id.split("query_")[-1]])
    # if no_answer_count >= len(features)/FLAGS.no_answer_threshold:
    #   all_predictions[example.qas_id] = ''

    # if not all_predictions[example.qas_id] and feature.no_answer:
    #   # 无法回答并且预测为空
    #   all_sentence_prediction_pairs[example.qas_id] = best_non_null_entry.sentence + '\t' + example.question_text + '\t' + best_non_null_entry.text + '\t' + '0'
    if feature.no_answer and all_predictions[example.qas_id]:
      # 无法回答但是预测不为空
      all_sentence_prediction_pairs[example.qas_id] = best_non_null_entry.sentence + '\t' + example.question_text + '\t' + best_non_null_entry.text + '\t' + '0'
    # elif not feature.no_answer and not all_predictions[example.qas_id]:
    #   # 可以回答但是预测为空
    #   all_sentence_prediction_pairs[example.qas_id] = best_non_null_entry.sentence + '\t' + example.question_text + '\t' + best_non_null_entry.text + '\t' + '1'
    elif not feature.no_answer and all_predictions[example.qas_id] == example.orig_answer_text:
      # 可以回答并且预测的回答正确
      all_sentence_prediction_pairs[example.qas_id] = best_non_null_entry.sentence + '\t' + example.question_text + '\t' + best_non_null_entry.text + '\t' + '1'
    # elif not feature.no_answer and all_predictions[example.qas_id] != example.orig_answer_text and not all_predictions[example.qas_id]:
    #   # 可以回答但是预测的回答错误
    #   all_sentence_prediction_pairs[example.qas_id] = best_non_null_entry.sentence + '\t' + example.question_text + '\t' + best_non_null_entry.text + '\t' + '1'
    
    
    # all_nbest_json[example.qas_id] = nbest_json

  logger.info(all_predictions[example.qas_id])
  logger.info('{}, {}'.format(best_non_null_entry.start_index, best_non_null_entry.end_index))
  logger.info(all_predictions)
  return all_predictions

  if output_prediction_file:
    with tf.gfile.Open(output_prediction_file, "w") as writer:
      writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

  # with tf.gfile.Open(output_prediction_file.split('.')[0]+'_sentence.json', "w") as writer:
  #   writer.write(json.dumps(all_sentence_prediction_pairs , indent=4, ensure_ascii=False) + "\n")

  # with tf.gfile.Open(output_nbest_file, "w") as writer:
  #   writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training=False, is_predict=False):
    self.filename = filename
    self.is_training = is_training
    self.is_predict = is_predict
    # self.num_features = 0
    self._writer = tf.io.TFRecordWriter(filename)
    self.feature_ls = []

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.feature_ls.append(feature)
    # self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["input_span_mask"] = create_int_feature(feature.input_span_mask)

    if self.is_training:
      try:
        features["start_positions"] = create_int_feature([feature.start_position])
      except:
        import ipdb; ipdb.set_trace()
      features["end_positions"] = create_int_feature([feature.end_position])
      features['no_answers'] = create_int_feature([feature.no_answer])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", 'no_answer', 'logits_cls'])


def mrc_initialize():
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if os.environ.get("GPU_FRACTION", None):
      gpu_config = tf.ConfigProto()
      gpu_config.gpu_options.per_process_gpu_memory_fraction = float(os.environ["GPU_FRACTION"])
  else:
      gpu_config = None

  model_fn = model_fn_builder(
      bert_config=bert_config,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  run_config = tf.contrib.tpu.RunConfig(
      model_dir=FLAGS.output_dir, session_config=gpu_config)
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.predict_batch_size)

  # run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir)
  # estimator = tf.estimator.Estimator(
  #     model_fn=model_fn,
  #     config=run_config)
  
  return estimator, tokenizer


def mrc_extract(paras, estimator, tokenizer, MRC_QUESTIONS, **kwargs):
  dic = {'data': []}
  for ind, para in enumerate(paras):
    cur_dic = {'paragraphs':[{'id':'demo_'+str(ind), 'context':para, 'qas':[]}], 'id':'demo_'+str(ind), 'title':'demo_example'}
    for subind, ques in enumerate(MRC_QUESTIONS):
      cur_dic['paragraphs'][0]['qas'].append({'question': ques, 'id': 'demo_'+str(ind)+'_query_'+ques.replace("是什么？", ""), 'answers': [{'text':para[3:10], 'answer_start': 3}]})
    dic['data'].append(cur_dic)

  logger.info(dic)
  demo_examples = read_squad_examples(
        input_file=dic, is_predict=True)
  demo_writer = FeatureWriter(
        filename="demo.tf_record",
        is_predict=True)
  demo_features = collections.deque()

  def append_feature(feature):
    demo_features.append(feature)
    demo_writer.process_feature(feature)

  convert_examples_to_features(
      examples=demo_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_predict=True,
      output_fn=append_feature)

  demo_writer.close()

  logger.info('predict batch size: {}'.format(FLAGS.predict_batch_size))

  all_results = []

  demo_features_copy = copy.deepcopy(demo_features)

  if os.environ.get("TF_SERVING", None):
      while demo_features:
          unique_id_ls, input_ids_ls, input_mask_ls, segment_ids_ls, input_span_mask_ls = [[] for _ in range(5)]
          while len(unique_id_ls) < FLAGS.predict_batch_size and demo_features:
              demo_feat = demo_features.popleft()
              unique_id_ls.append(demo_feat.unique_id)
              input_ids_ls.append(demo_feat.input_ids)
              input_mask_ls.append(demo_feat.input_mask)
              segment_ids_ls.append(demo_feat.segment_ids)
              input_span_mask_ls.append(demo_feat.input_span_mask)

          tmp_dic = {"inputs": dict(zip(["unique_ids", "input_ids", "input_mask", "segment_ids", "input_span_mask"],
                                        [unique_id_ls, input_ids_ls, input_mask_ls, segment_ids_ls, input_span_mask_ls]))}

          logger.info(tmp_dic["inputs"]["unique_ids"])
          with open(os.path.join(FLAGS.output_dir, 'input.json'), 'w', encoding='utf8') as wf:
              json.dump(tmp_dic, wf)
          logger.info("request tf serving ...")
          while True:
              try:
                  response = requests.post(tf_serving_url, data=json.dumps(tmp_dic))
              except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                  if "10054" in str(e) or "104" in str(e):
                      continue
                  else:
                      raise Exception(e)
              break

          # response = get_grpc(tmp_dic["inputs"])
          logger.info(response)
          response_dic = json.loads(response.text)
          outputs_dic = response_dic["outputs"]
          for i in range(len(unique_id_ls)):
              # unique_id = int(outputs_dic["unique_ids"][i])
              # start_logits = [float(x) for x in outputs_dic["start_logits"][i].flat]
              # end_logits = [float(x) for x in outputs_dic["end_logits"][i].flat]
              all_results.append(RawResult(
                        unique_id=outputs_dic["unique_ids"][i],
                        start_logits=outputs_dic["start_logits"][i],
                        end_logits=outputs_dic["end_logits"][i],
                        no_answer=outputs_dic["no_answer"][i],
                        logits_cls=outputs_dic["logits_cls"][i]))

  else:
      demo_input_fn = input_fn_builder(
          input_file=demo_writer.filename,
          seq_length=FLAGS.max_seq_length,
          is_predict=True,
          drop_remainder=False)

      for result in estimator.predict(demo_input_fn, yield_single_examples=True):
          logger.info("Predicting feature: %d" % (len(all_results))) # tf.logging
          unique_id = int(result["unique_ids"])
          start_logits = [float(x) for x in result["start_logits"].flat]
          end_logits = [float(x) for x in result["end_logits"].flat]
          all_results.append(RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    no_answer=result['no_answer'],
                    logits_cls=result['logits_cls']))

  return write_predictions(demo_examples, demo_features_copy, all_results,
                           FLAGS.n_best_size, FLAGS.max_answer_length,
                           FLAGS.do_lower_case, **kwargs)




if __name__ == '__main__':
    estimator, tokenizer = mrc_initialize()
    mrc_extract(['静声理州罗市人民医院苏州大学附属高邮医院出院记录姓名:柏中富科室:五病区床号:26住院号:1818352下少量积液,左侧额颞叶脑挫伤,左颞骨骨折,鞍区占位",2018-07-09头颅M"两侧放射冠区腔隙性脑梗塞及缺血灶,左额颞叶脑挫伤,鞍内占位,考虑垂体瘤可能,左侧乳突积血"。颅内出血渐吸收,2018-06-25,甲功五项:三碘甲状腺原氨酸0.53nmol/L,甲状腺素57.97nmo1/L,游离三碘甲状腺原氨酸2.96pmo1/L,游离甲状腺素11.40pmo1/L,促甲状腺激素0.66uIU/m1:2018-06-25,性激素六项:孕酮0.19ng/ml,雌二醇9.09pg/mL,促卵泡刺激素8.41mIU/mL,泌乳素903.90uIU/ml,睾酮0.92ng/mL,促黄体生成素3.61mIU/mL;2018-07-06,甲功五项:三碘甲状腺原氨酸0.98nmol/L,甲状腺素111.81nmo1/L,游离三碘甲状腺原氨酸4.11pmol1/L,游离甲状腺素13.07pmo1/L,促甲状腺激素0.44uIU/ml:2018-07-06,性激素六项:孕酮0.19ng/ml,雌二醇8.97pg/mL,促卵泡刺激素9.69mIU/mL,泌乳素1381.00uIU/ml,睾酮1.49ng/mL,促黄体生成素4.22mIU/mL;支持垂体腺瘤,予激素替代治疗,患者心电监护示"心率慢",予行动态心电图,并请心内科会诊,2018-07-03心脏超声"左房稍大伴二尖瓣轻中度关闭不全,左室舒张功能减退,左室收缩功能正常",2018-07-13心脏CTA"前降支近段混合斑块形成,管腔中度狭窄。前降支中段心肌桥(表浅型)"0心内科会诊建议口服阿司匹林等,现忠者无明显不适,今予出院。出院情况:(口治愈区好转口未愈口未治口转院口自动出院)伤口愈合:患者一般情况可,未诉明显不适,无发热,无恶心呕吐,食欲可,大小便自解。查体:神志清,精神一般,GCS15分,双瞳孔等大等圆,直径约2.5mm,对光反射灵敏,粗测视力视野无异常,耳鼻无异常分泌物,鼻唇沟对称,伸舌居中,颈软,粗测左耳听力差,四肢活动自如,肌力V级,肌张力正常,深浅感觉正常,生理反射正常,病理反射未引出。出院医嘱:注意休息,继续口服药物,半月后我科复诊,心内科门诊定期随诊,不适随诊X线号:00275707CT号:3024465630247808MRI号:病理检验号:主治医师:医师:任万印在', '静声理州罗市人民医院苏州大学附属高邮医院出院记录姓名:柏中富科室:五病区床号:26住院号:1818352下少量积液,左侧额颞叶脑挫伤,左颞骨骨折,鞍区占位",2018-07-09头颅M"两侧放射冠区腔隙性脑梗塞及缺血灶,左额颞叶脑挫伤,鞍内占位,考虑垂体瘤可能,左侧乳突积血"。颅内出血渐吸收,2018-06-25,甲功五项:三碘甲状腺原氨酸0.53nmol/L,甲状腺素57.97nmo1/L,游离三碘甲状腺原氨酸2.96pmo1/L,游离甲状腺素11.40pmo1/L,促甲状腺激素0.66uIU/m1:2018-06-25,性激素六项:孕酮0.19ng/ml,雌二醇9.09pg/mL,促卵泡刺激素8.41mIU/mL,泌乳素903.90uIU/ml,睾酮0.92ng/mL,促黄体生成素3.61mIU/mL;2018-07-06,甲功五项:三碘甲状腺原氨酸0.98nmol/L,甲状腺素111.81nmo1/L,游离三碘甲状腺原氨酸4.11pmol1/L,游离甲状腺素13.07pmo1/L,促甲状腺激素0.44uIU/ml:2018-07-06,性激素六项:孕酮0.19ng/ml,雌二醇8.97pg/mL,促卵泡刺激素9.69mIU/mL,泌乳素1381.00uIU/ml,睾酮1.49ng/mL,促黄体生成素4.22mIU/mL;支持垂体腺瘤,予激素替代治疗,患者心电监护示"心率慢",予行动态心电图,并请心内科会诊,2018-07-03心脏超声"左房稍大伴二尖瓣轻中度关闭不全,左室舒张功能减退,左室收缩功能正常",2018-07-13心脏CTA"前降支近段混合斑块形成,管腔中度狭窄。前降支中段心肌桥(表浅型)"0心内科会诊建议口服阿司匹林等,现忠者无明显不适,今予出院。出院情况:(口治愈区好转口未愈口未治口转院口自动出院)伤口愈合:患者一般情况可,未诉明显不适,无发热,无恶心呕吐,食欲可,大小便自解。查体:神志清,精神一般,GCS15分,双瞳孔等大等圆,直径约2.5mm,对光反射灵敏,粗测视力视野无异常,耳鼻无异常分泌物,鼻唇沟对称,伸舌居中,颈软,粗测左耳听力差,四肢活动自如,肌力V级,肌张力正常,深浅感觉正常,生理反射正常,病理反射未引出。出院医嘱:注意休息,继续口服药物,半月后我科复诊,心内科门诊定期随诊,不适随诊X线号:00275707CT号:3024465630247808MRI号:病理检验号:主治医师:医师:任万印在'], estimator, tokenizer)
