# Candle BiRNN

使用 Candle 實作 PyTorch LSTM 推論，並實作雙向 LSTM 推論。

## 測試資料

1. lstm_test.pt: 使用 Pytorch demo 程式產生結果。程式碥如下：

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

1. bi_lstm_test.pt: 使用 Pytorch demo 程式產生結果。程式碥如下：

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
