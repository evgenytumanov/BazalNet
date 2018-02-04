

import numpy as np
import tensorflow as tf
import config


def tf_cut_function(val, vlo, vhi, glo, ghi):
  if vlo is None:
    return val
  a = tf.clip_by_value(val, vlo, vhi)
  if glo is None:
    return a
  assert ghi >= vhi > vlo >= glo
  zz = tf.clip_by_value(val, glo, ghi)
  return zz - tf.stop_gradient(zz - a)

def sigmoid_cutoff(x, cutoff):
  """Sigmoid with cutoff, e.g., 1.2sigmoid(x) - 0.1."""
  y = tf.sigmoid(x)
  if cutoff < 1.01: return y
  d = (cutoff - 1.0) / 2.0
  z = cutoff * y - d
  dd = (config.smooth_grad - 1.0) / 2.0 if config.smooth_grad else None
  glo, ghi = (-dd, 1+dd) if config.smooth_grad else (None, None)
  return tf_cut_function(z, 0, 1, glo, ghi)

def tanh_cutoff(x, cutoff):
  """Tanh with cutoff, e.g., 1.1tanh(x) cut to [-1. 1]."""
  y = tf.tanh(x)
  if cutoff < 1.01: return y
  z = cutoff * y
  tcut = config.smooth_grad_tanh
  glo, ghi = (-tcut, tcut) if tcut else (None, None)
  return tf_cut_function(z, -1, 1, glo, ghi)

def conv_linear(arg, kw, kh, nout, prefix, bias=0):
  """Convolutional linear map."""
  strides = [1, 1, 1, 1]
  if isinstance(arg, list):
    if len(arg) == 1:
      arg = arg[0]
    else:
      arg = tf.concat(len(mytf.shape_list(arg[0]))-1, arg)
  nin = mytf.shape_list(arg)[-1]
  with tf.variable_scope(prefix):
    k = tf.get_variable("CvK", [kw, kh, nin, nout])
    res = mytf.conv2d(arg, k, strides, "SAME")

    if bias is None:
      return res
    bias_term = tf.get_variable("CvB", [nout],
                                initializer=tf.constant_initializer(0.0))
    return res + bias_term + float(bias)

def conv_gru(mem, kw, kh, nmaps, cutoff, prefix, extras=[]):
  """Convolutional GRU."""
  # mem shape: bs x length x height x nmaps
  def conv_lin(arg, suffix, bias_start):
    return conv_linear(extras + [arg], kw, kh, nmaps,
                       prefix + "/" + suffix, bias=bias_start)
  reset = sigmoid_cutoff(conv_lin(mem, "r", 1), cutoff)
  candidate = tanh_cutoff(conv_lin(reset * mem, "c", 0), config.cutoff_tanh)
  gate = sigmoid_cutoff(conv_lin(mem, "g", 1), cutoff)
  return gate * mem + (1 - gate) * candidate
