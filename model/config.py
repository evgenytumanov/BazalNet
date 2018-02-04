
op = lambda x, y: x+y
op_sym = '+'

training_size= 50000

numbers_alphabet_size = 10

digits = 3
maxlen = digits * 2 + 1

nmaps = 11
CGRU_apply_times = 2

lr = 0.01
cutoff = 1.2
cutoff_tanh = 0.0

batch_size = 32
kw = 3
kh = 3

smooth_grad = 0.0
smooth_grad_tanh = 0.0

'''

  tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate.")
  tf.app.flags.DEFINE_float("init_weight", 1.0, "Initial weights deviation.")
  tf.app.flags.DEFINE_float("max_grad_norm", 1.0, "Clip gradients to this norm.")
  tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
  tf.app.flags.DEFINE_float("cutoff_tanh", 0.0, "Cutoff at tanh.")
  tf.app.flags.DEFINE_float("pull", 0.0005, "Starting pull of the relaxations.")
  tf.app.flags.DEFINE_float("pull_incr", 1.2, "Increase pull by that much.")
  tf.app.flags.DEFINE_float("curriculum_bound", 0.15, "Move curriculum < this.")
  tf.app.flags.DEFINE_float("dropout", 0.15, "Dropout that much.")
  tf.app.flags.DEFINE_integer("max_steps", 0, "Quit after this many steps.")
  tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
  tf.app.flags.DEFINE_integer("low_batch_size", 16, "Low batch size.")
  tf.app.flags.DEFINE_integer("steps_per_epoch", 200, "Steps per epoch.")
  tf.app.flags.DEFINE_integer("nmaps", 24, "Number of floats in each cell.")
  tf.app.flags.DEFINE_integer("niclass", 33, "Number of classes (0 is padding).")
  tf.app.flags.DEFINE_integer("noclass", 33, "Number of classes (0 is padding).")
  tf.app.flags.DEFINE_integer("max_length", 41, "Maximum length.")
  tf.app.flags.DEFINE_integer("rx_step", 6, "Relax that many recursive steps.")
  tf.app.flags.DEFINE_integer("random_seed", 125459, "Random seed.")
  tf.app.flags.DEFINE_integer("time_till_ckpt", 30, "How many tests per checkpoint")
  tf.app.flags.DEFINE_integer("time_till_eval", 2, "Number of steps between evals")
  tf.app.flags.DEFINE_integer("nconvs", 2, "How many convolutions / 1 step.")
  tf.app.flags.DEFINE_integer("kw", 3, "Kernel width.")
  tf.app.flags.DEFINE_integer("kh", 3, "Kernel height.")
  tf.app.flags.DEFINE_integer("height", 4, "Height.")
  tf.app.flags.DEFINE_integer("forward_max", 401, "Maximum forward length.")
  tf.app.flags.DEFINE_integer("nprint", 0, "How many test examples to print out.")
  tf.app.flags.DEFINE_integer("mode", 0, "Mode: 0-train other-decode.")
  tf.app.flags.DEFINE_bool("animate", False, "Whether to produce an animation.")
  tf.app.flags.DEFINE_float("smooth_grad", 0.0, "Whether to avoid clipping gradient")
  tf.app.flags.DEFINE_float("smooth_grad_tanh", 0.0, "Whether to avoid clipping tanh gradient")
  tf.app.flags.DEFINE_string("task", "badd", "Which task are we learning?")
  tf.app.flags.DEFINE_string("train_dir", "/tmp/neural", "Directory to store models.")

  tf.app.flags.DEFINE_float("layer_scale", 1.0, "Number of layers to use")

  # Batchnorm:     0 = none

'''