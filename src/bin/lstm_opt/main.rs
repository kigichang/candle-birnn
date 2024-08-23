use std::time;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{lstm, rnn::LSTMState, LSTMConfig, VarBuilder, RNN};

fn main() -> Result<()> {
    let cpu = &Device::Cpu;
    let vb = VarBuilder::from_pth("lstm_test.pt", DType::F32, cpu)?;

    let lstm = lstm(768, 768, LSTMConfig::default(), vb.clone())?;
    let input = vb.get((256, 1, 768), "input")?;
    let output = vb.get((256, 1, 768), "output")?;

    let input_t = input.transpose(0, 1)?;
    let start = time::Instant::now();
    let states = lstm.seq(&input_t)?;
    let result = lstm.states_to_tensor(&states)?;
    let elapsed = start.elapsed();
    let result = result.transpose(0, 1)?;
    println!("Elapsed: {:?}", elapsed.as_secs_f32());
    assert_tensor(&result, &output, 3, 1e-5)?;

    let start = time::Instant::now();
    let result = mylstm(vb.clone(), &input)?;
    let elapsed = start.elapsed();
    assert_tensor(&result, &output, 3, 1e-5)?;
    println!("result: {:?}", result.shape());
    println!("Elapsed: {:?}", elapsed.as_secs_f32());
    Ok(())
}

fn assert_tensor(a: &Tensor, b: &Tensor, dim: usize, v: f32) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let mut t = (a - b)?.abs()?;

    for _i in 0..dim {
        t = t.max(D::Minus1)?;
    }

    let t = t.to_scalar::<f32>()?;
    assert!(t < v, "delta: {}, check: {}", t, v);
    Ok(())
}

fn mylstm(vb: VarBuilder, input: &Tensor) -> Result<Tensor> {
    let w_ih = vb.get((4 * 768, 768), "weight_ih_l0")?.t()?;
    let w_hh = vb.get((4 * 768, 768), "weight_hh_l0")?.t()?;

    let b_ih = vb.get(4 * 768, "bias_ih_l0")?;
    let b_hh = vb.get(4 * 768, "bias_hh_l0")?;

    let zeros = Tensor::zeros((1, 768), DType::F32, &Device::Cpu)?;
    let init_state = LSTMState {
        h: zeros.clone(),
        c: zeros.clone(),
    };

    let input_t = input.transpose(0, 1)?;
    let input_t = input_t.get(0)?.matmul(&w_ih)?.broadcast_add(&b_ih)?;

    let mut states = Vec::with_capacity(256 + 1);
    states.push(init_state);

    for i in 0..256 {
        let h = &states[i].h;
        let c = &states[i].c;

        let hh = h.matmul(&w_hh)?.broadcast_add(&b_hh)?;
        let chunks = &input_t.get(i)?.broadcast_add(&hh)?.chunk(4, 1)?;
        let in_gate = candle_nn::ops::sigmoid(&chunks[0])?;
        let forget_gate = candle_nn::ops::sigmoid(&chunks[1])?;
        let cell_gate = chunks[2].tanh()?;
        let out_gate = candle_nn::ops::sigmoid(&chunks[3])?;

        let next_c = ((forget_gate * c)? + (in_gate * cell_gate)?)?;
        let next_h = (out_gate * next_c.tanh()?)?;
        states.push(LSTMState {
            c: next_c,
            h: next_h,
        });
    }

    let output = states.into_iter().skip(1).map(|s| s.h).collect::<Vec<_>>();

    Ok(Tensor::stack(&output, 0)?)
}
