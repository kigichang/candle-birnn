//
//                       _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//               佛祖保佑         永無BUG
//
// FROM: https://gist.github.com/edokeh/7580064
//

use candle_core::{Result, Tensor};
use candle_nn::{rnn::LSTMState, LSTMConfig, VarBuilder, LSTM, RNN};

pub use candle_nn::lstm;

pub fn reverse_lstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: VarBuilder,
) -> Result<LSTM> {
    let vb = vb.rename_f(|s| format!("{}_reverse", s));
    lstm(in_dim, hidden_dim, config, vb)
}

pub trait TchRNN: RNN {
    fn tch_seq(&self, input: &Tensor) -> Result<Vec<Self::State>> {
        let (_seq_len, batch_dim, _features) = input.dims3()?;
        let init_state = self.zero_state(batch_dim)?;
        self.tch_seq_init(input, &init_state)
    }

    fn tch_seq_init(&self, input: &Tensor, init_state: &Self::State) -> Result<Vec<Self::State>> {
        let (seq_len, _batch_dim, _features) = input.dims3()?;
        let mut output = Vec::with_capacity(seq_len);
        for seq_index in 0..seq_len {
            let input = input.get(seq_index)?;
            let state = if seq_index == 0 {
                self.step(&input, init_state)?
            } else {
                self.step(&input, &output[seq_index - 1])?
            };
            output.push(state);
        }
        Ok(output)
    }

    fn tch_states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor>;
}

impl TchRNN for LSTM {
    fn tch_states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h().clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 0)
    }
}

pub trait BiRNN<'a> {
    type State: Clone;
    type Item: RNN<State = Self::State>;

    fn forward(&'a self) -> &'a Self::Item;
    fn backward(&'a self) -> &'a Self::Item;

    fn zero_state(&'a self, batch_dim: usize) -> Result<(Self::State, Self::State)> {
        let forward = self.forward().zero_state(batch_dim)?;
        let backward = self.backward().zero_state(batch_dim)?;
        Ok((forward, backward))
    }

    fn step(
        &'a self,
        input: (&Tensor, &Tensor),
        state: &(Self::State, Self::State),
    ) -> Result<(Self::State, Self::State)> {
        Ok((
            self.forward().step(input.0, &state.0)?,
            self.backward().step(input.1, &state.1)?,
        ))
    }

    fn tch_seq(&'a self, input: &Tensor) -> Result<Vec<(Self::State, Self::State)>> {
        let (_seq_len, batch_dim, _features) = input.dims3()?;
        let init_state = self.zero_state(batch_dim)?;
        self.tch_seq_init(input, &init_state)
    }

    fn tch_seq_init(
        &'a self,
        input: &Tensor,
        init_state: &(Self::State, Self::State),
    ) -> Result<Vec<(Self::State, Self::State)>> {
        let (seq_len, _batch_dim, _features) = input.dims3()?;
        let mut output: Vec<(_, _)> = Vec::with_capacity(seq_len);
        let seq_len = seq_len - 1;
        let f = self.forward();
        let b = self.backward();
        for seq_index in 0..=seq_len {
            let input_f = input.get(seq_index)?;
            let state_f = if seq_index == 0 {
                f.step(&input_f, &init_state.0)?
            } else {
                f.step(&input_f, &output[seq_index - 1].0)?
            };

            let input_b = input.get(seq_len - seq_index)?;
            let state_b = if seq_index == 0 {
                b.step(&input_b, &init_state.1)?
            } else {
                b.step(&input_b, &output[seq_index - 1].1)?
            };
            output.push((state_f, state_b));
        }
        Ok(output)
    }

    fn tch_states_to_tensor(&'a self, states: &[(Self::State, Self::State)]) -> Result<Tensor>;
}

pub struct BiLSTM {
    forward: LSTM,
    backward: LSTM,
}

impl<'a> BiRNN<'a> for BiLSTM {
    type State = LSTMState;
    type Item = LSTM;

    fn forward(&'a self) -> &'a Self::Item {
        &self.forward
    }

    fn backward(&'a self) -> &'a Self::Item {
        &self.backward
    }

    fn tch_states_to_tensor(&'a self, states: &[(Self::State, Self::State)]) -> Result<Tensor> {
        let mut f = Vec::with_capacity(states.len() * 2);
        let mut b = Vec::with_capacity(states.len());

        for state in states {
            f.push(state.0.h().clone());
            b.push(state.1.h().clone());
        }
        b.reverse();
        f.extend(b);

        Tensor::stack(&f, 0)
    }
}

#[cfg(test)]
mod tests {
    static IN_DIM: usize = 10;
    static HIDDEN_DIM: usize = 20;

    use std::path::Path;

    use anyhow::Result;
    use candle_core::{DType, Device, D};
    use candle_nn::VarBuilder;

    use super::*;

    fn assert_eq_vec1(a: &Tensor, b: &Tensor, v: f32) -> Result<()> {
        assert_eq!(a.dims(), b.dims());

        let t = (a - b)?.abs()?.min(D::Minus1)?.to_scalar::<f32>()?;
        assert!(t < v);
        Ok(())
    }

    fn load_pt<P: AsRef<Path>>(path: P) -> Result<VarBuilder<'static>> {
        Ok(VarBuilder::from_pth(path, DType::F32, &Device::Cpu)?)
    }

    #[test]
    fn test_reverse_lstm() -> Result<()> {
        let vb = load_pt("bi_lstm_test.pt")?;
        reverse_lstm(IN_DIM, HIDDEN_DIM, LSTMConfig::default(), vb)?;
        Ok(())
    }

    #[test]
    fn test_sub() -> Result<()> {
        let a = Tensor::rand(-0.1_f32, 1.0_f32, (3, 5), &Device::Cpu)?;
        println!("a: {:?}", a);
        let b = Tensor::rand(-0.1_f32, 1.0_f32, (3, 5), &Device::Cpu)?;
        println!("b: {:?}", b);

        let t = (a - b)?
            .abs()?
            .min(D::Minus1)?
            .min(D::Minus1)?
            .to_scalar::<f32>()?;
        println!("t: {:?}", t);

        Ok(())
    }
}
