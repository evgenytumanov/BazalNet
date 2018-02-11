
op = lambda x, y: x+y
op_sym = '+'

training_size= 50000

numbers_alphabet_size = 2

digits = 3
maxlen = 21 #digits * 2 + 1

mental_width = 4
nmaps = 24
CGRU_apply_times = 2

lr = 0.001
cutoff = 0.0
cutoff_tanh = 0.0

batch_size = 64
val_batch_size = 32

print_loss_every = 100
calc_val_loss_every = 100

kw = 3
kh = 3

smooth_grad = 0.0
smooth_grad_tanh = 0.0

iters = 10000
