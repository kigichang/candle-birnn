import torch
import torch.nn as nn
import time

INPUT_SIZE = 768
HIDDEN_SIZE = 768
NUM_LAYERS = 1
SEQ_LEN = 256
BATCH_SIZE = 1

# LSTM with batch_first=False
rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=False)

input = torch.randn(1, 1, INPUT_SIZE)
start = time.time()
output, (hn, cn) = rnn(input)
end = time.time()
print("lstm one: ", end - start)

input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
start = time.time()
output, (hn, cn) = rnn(input)
end = time.time()
print("lstm_batch_false: ", end - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "lstm_batch_false.pt")

# BiLSTM with batch_first=False
rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=False, bidirectional=True)

input = torch.randn(1, 1, INPUT_SIZE)
start = time.time()
output, (hn, cn) = rnn(input)
end = time.time()
print("bilstm one: ", end - start)

input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
start = time.time()
output, (hn, cn) = rnn(input)
end = time.time()
print("bilstm_batch_false: ", end - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "bi_lstm_batch_false.pt")

# LSTM with batch_first=True
rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)

input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
start = time.time()
output, (hn, cn) = rnn(input)
end = time.time()
print("lstm_batch_true: ", end - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "lstm_batch_true.pt")

# BiLSTM with batch_first=True
rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, bidirectional=True)

input = torch.randn(SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
start = time.time()
output, (hn, cn) = rnn(input)
end = time.time()
print("bilstm_batch_true: ", end - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "bi_lstm_batch_true.pt")