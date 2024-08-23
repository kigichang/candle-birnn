import torch
import torch.nn as nn
import time

rnn = nn.LSTM(768, 768, 1)
input = torch.randn(256, 1, 768)
output, (hn, cn) = rnn(input)

start = time.time()
output, (hn, cn) = rnn(input)
print("Time taken: ", time.time() - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "lstm_test.pt")


# Bi-LSTM

rnn = nn.LSTM(768, 768, 1, bidirectional=True)
input = torch.randn(256, 1, 768)
output, (hn, cn) = rnn(input)

start = time.time()
output, (hn, cn) = rnn(input)
print("Time taken: ", time.time() - start)

state_dict = rnn.state_dict()
state_dict['input'] = input
state_dict['output'] = output
state_dict['hn'] = hn
state_dict['cn'] = cn
torch.save(state_dict, "bi_lstm_test.pt")