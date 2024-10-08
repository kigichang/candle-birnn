# Candle BiRNN

Implementing PyTorch LSTM inference using Candle, including the implementation of bidirectional LSTM inference.

## Test Data

1. lstm_test.pt: Results generated using a PyTorch demo program. The code is as follows:

    ```python
    import torch
    import torch.nn as nn

    rnn = nn.LSTM(10, 20, 1)
    input = torch.randn(5, 3, 10)
    output, (hn, cn) = rnn(input)

    state_dict = rnn.state_dict()
    state_dict['input'] = input
    state_dict['output'] = output
    state_dict['hn'] = hn
    state_dict['cn'] = cn
    torch.save(state_dict, "lstm_test.pt")
    ```

1. bi_lstm_test.pt: Results generated using a PyTorch demo program. The code is as follows:

    ```python
    import torch
    import torch.nn as nn

    rnn = nn.LSTM(10, 20, 1, bidirectional=True)
    input = torch.randn(5, 3, 10)
    output, (hn, cn) = rnn(input)

    state_dict = rnn.state_dict()
    state_dict['input'] = input
    state_dict['output'] = output
    state_dict['hn'] = hn
    state_dict['cn'] = cn
    torch.save(state_dict, "bi_lstm_test.pt")
    ```
