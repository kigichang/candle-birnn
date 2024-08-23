use candle_core::{IndexOp, Result, Tensor, D};

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct MyLSTMState {
    h: Tensor,
    c: Tensor,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct MyLSTMConfig {
    pub w_ih_init: candle_nn::Init,
    pub w_hh_init: candle_nn::Init,
    pub b_ih_init: Option<candle_nn::Init>,
    pub b_hh_init: Option<candle_nn::Init>,
    pub layer_idx: usize,
    pub batch_first: bool,
}

impl Default for MyLSTMConfig {
    fn default() -> Self {
        Self {
            w_ih_init: candle_nn::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: candle_nn::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(candle_nn::Init::Const(0.)),
            b_hh_init: Some(candle_nn::Init::Const(0.)),
            layer_idx: 0,
            batch_first: false,
        }
    }
}

impl MyLSTMConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: candle_nn::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: candle_nn::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
            batch_first: false,
        }
    }
}

impl MyLSTMState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms, unused)]
#[derive(Clone, Debug)]
pub struct MyLSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: MyLSTMConfig,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

/// Creates a LSTM layer.
pub fn mylstm(
    in_dim: usize,
    hidden_dim: usize,
    config: MyLSTMConfig,
    vb: crate::VarBuilder,
) -> Result<MyLSTM> {
    let layer_idx = config.layer_idx;
    let w_ih = vb.get_with_hints(
        (4 * hidden_dim, in_dim),
        &format!("weight_ih_l{layer_idx}"), // Only a single layer is supported.
        config.w_ih_init,
    )?;
    let w_hh = vb.get_with_hints(
        (4 * hidden_dim, hidden_dim),
        &format!("weight_hh_l{layer_idx}"), // Only a single layer is supported.
        config.w_hh_init,
    )?;
    let b_ih = match config.b_ih_init {
        Some(init) => {
            Some(vb.get_with_hints(4 * hidden_dim, &format!("bias_ih_l{layer_idx}"), init)?)
        }
        None => None,
    };
    let b_hh = match config.b_hh_init {
        Some(init) => {
            Some(vb.get_with_hints(4 * hidden_dim, &format!("bias_hh_l{layer_idx}"), init)?)
        }
        None => None,
    };
    Ok(MyLSTM {
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hidden_dim,
        config,
        device: vb.device().clone(),
        dtype: vb.dtype(),
    })
}

impl MyLSTM {
    fn zero_state(&self, batch_dim: usize) -> Result<MyLSTMState> {
        let zeros =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(MyLSTMState {
            h: zeros.clone(),
            c: zeros.clone(),
        })
    }

    fn inner_step(&self, input: &Tensor, in_state: &MyLSTMState) -> Result<MyLSTMState> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks = (&w_ih + &w_hh)?.chunk(4, 1)?;
        let in_gate = candle_nn::ops::sigmoid(&chunks[0])?;
        let forget_gate = candle_nn::ops::sigmoid(&chunks[1])?;
        let cell_gate = chunks[2].tanh()?;
        let out_gate = candle_nn::ops::sigmoid(&chunks[3])?;

        let next_c = ((forget_gate * &in_state.c)? + (in_gate * cell_gate)?)?;
        let next_h = (out_gate * next_c.tanh()?)?;
        Ok(MyLSTMState {
            c: next_c,
            h: next_h,
        })
    }

    fn seq(&self, input: &Tensor) -> Result<Vec<MyLSTMState>> {
        let batch_dim = input.dim(1)?;
        let state = self.zero_state(batch_dim)?;
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, init_state: &MyLSTMState) -> Result<Vec<MyLSTMState>> {
        let (seq_len, _batch_size, _features) = input.dims3()?;
        let mut output = Vec::with_capacity(seq_len);
        for seq_index in 0..seq_len {
            let input = input.i((seq_index, .., ..))?.contiguous()?;
            let state = if seq_index == 0 {
                self.inner_step(&input, init_state)?
            } else {
                self.inner_step(&input, &output[seq_index - 1])?
            };
            output.push(state);
        }
        Ok(output)
    }

    fn step(&self, all_input: &Tensor, init_state: &MyLSTMState) -> Result<Vec<MyLSTMState>> {
        let (seq_len, _batch_size, _features) = all_input.dims3()?;

        let mut output = Vec::with_capacity(seq_len);
        let self_w_ih_t = self.w_ih.t()?;
        let self_w_hh_t = self.w_hh.t()?;

        // let a = std::time::Instant::now();
        // let test = all_input.broadcast_matmul(&self_w_ih_t)?;
        // println!("broadcast_matmul: {:?}", a.elapsed().as_secs_f32());
        //println!("test: {:?}", test.dims());

        for seq_index in 0..seq_len {
            let input = all_input.get(seq_index)?.contiguous()?;
            //println!("input: {:?}", input.dims());
            let prev_state = if seq_index == 0 {
                init_state
            } else {
                &output[seq_index - 1]
            };
            let state = {
                //let astart = std::time::Instant::now();
                let start = std::time::Instant::now();
                let w_ih = input.matmul(&self_w_ih_t)?;
                println!("matmul: {:?}", start.elapsed().as_secs_f32());

                let start = std::time::Instant::now();
                let w_hh = prev_state.h.matmul(&self_w_hh_t)?;
                println!("matmul: {:?}", start.elapsed().as_secs_f32());

                let start = std::time::Instant::now();
                let w_ih = match &self.b_ih {
                    None => w_ih,
                    Some(b_ih) => w_ih.broadcast_add(b_ih)?,
                };
                println!("broadcast_add: {:?}", start.elapsed().as_secs_f32());

                let start = std::time::Instant::now();
                let w_hh = match &self.b_hh {
                    None => w_hh,
                    Some(b_hh) => w_hh.broadcast_add(b_hh)?,
                };
                println!("broadcast_add: {:?}", start.elapsed().as_secs_f32());

                //let start = std::time::Instant::now();
                let chunks = (&w_ih + &w_hh)?.chunk(4, 1)?;
                //println!("chunk: {:?}", start.elapsed().as_secs_f32());

                //let start = std::time::Instant::now();
                let in_gate = candle_nn::ops::sigmoid(&chunks[0])?;
                //println!("in_gate sigmoid: {:?}", start.elapsed().as_secs_f32());

                //let start = std::time::Instant::now();
                let forget_gate = candle_nn::ops::sigmoid(&chunks[1])?;
                //println!("forget_gate sigmoid: {:?}", start.elapsed().as_secs_f32());

                //let start = std::time::Instant::now();
                let cell_gate = chunks[2].tanh()?;
                //println!("cell_gate tanh: {:?}", start.elapsed().as_secs_f32());

                //let start = std::time::Instant::now();
                let out_gate = candle_nn::ops::sigmoid(&chunks[3])?;
                //println!("out_gate sigmoid: {:?}", start.elapsed().as_secs_f32());
                //println!("total: {:?}", astart.elapsed().as_secs_f32());

                let next_c = ((forget_gate * &prev_state.c)? + (in_gate * cell_gate)?)?;
                let next_h = (out_gate * next_c.tanh()?)?;
                MyLSTMState {
                    c: next_c,
                    h: next_h,
                }
            };

            output.push(state);
        }
        Ok(output)
    }

    fn states_to_tensor(&self, states: &[MyLSTMState]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 0)
    }
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

#[cfg(test)]
mod tests {

    use super::*;
    use anyhow::Result;
    use candle_core::{DType, Device};
    use candle_nn::{linear, lstm, LSTMConfig, Module, VarBuilder, VarMap, RNN};

    #[test]
    fn test_linear() -> Result<()> {
        const COUNT: usize = 256;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let input = Tensor::rand(-1.0_f32, 1.0_f32, (COUNT, 1, 768), &Device::Cpu)?;

        let a1 = linear(768, 768, vb.pp("a1"))?;

        let start = std::time::Instant::now();

        for _x in 0..100 {
            for i in 0..COUNT {
                let b1 = input.get(i)?;
                a1.forward(&b1)?; // input layer
            }
        }

        let c1 = start.elapsed().as_secs_f32();
        println!("a1: {:?}", c1);

        let a2 = linear(768, 768, vb.pp("a2"))?;
        let b2 = Tensor::rand(-1.0_f32, 1.0_f32, (COUNT, 1, 768), &Device::Cpu)?;

        let start = std::time::Instant::now();
        for _x in 0..100 {
            let _out = a1.forward(&b2)?;
        }
        let c2 = start.elapsed().as_secs_f32();
        println!("a2: {:?}", c2);
        //println!("out: {:?}", out.shape());

        println!("ratio: {:?}", c2 / c1);

        Ok(())
    }

    #[test]
    fn double_lstm() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let lstm = lstm(768 * 2, 768 * 2, LSTMConfig::default(), vb.clone())?;
        let input = Tensor::randn(0.0_f32, 1.0_f32, (256, 1, 768 * 2), &Device::Cpu)?;
        let input = &input.transpose(0, 1)?;
        let start = std::time::Instant::now();
        lstm.seq(&input)?;
        println!("candle: {:?}", start.elapsed().as_secs_f32());

        let mylstm = mylstm(768 * 2, 768 * 2, MyLSTMConfig::default(), vb.clone())?;
        let input = Tensor::randn(0.0_f32, 1.0_f32, (256, 1, 768 * 2), &Device::Cpu)?;

        let zero_state = mylstm.zero_state(1)?;

        let start = std::time::Instant::now();
        let output = mylstm.step(&input, &zero_state)?;
        //let output = mylstm.seq(&input)?;
        mylstm.states_to_tensor(&output)?;
        println!("mylstm: {:?}", start.elapsed().as_secs_f32());

        Ok(())
    }

    #[test]
    fn load_by_candle() -> Result<()> {
        let vb = VarBuilder::from_pth("lstm_test.pt", DType::F32, &Device::Cpu)?;
        let lstm = lstm(768, 768, LSTMConfig::default(), vb.clone())?;
        let input = vb.get((256, 1, 768), "input")?;
        let answer = vb.get((256, 1, 768), "output")?;
        let input = &input.transpose(0, 1)?;

        let start = std::time::Instant::now();
        let output = lstm.seq(&input)?;
        let output = lstm.states_to_tensor(&output)?;
        println!("candle: {:?}", start.elapsed().as_secs_f32());
        let output = output.transpose(0, 1)?;
        assert_tensor(&output, &answer, 3, 1e-5)?;

        let mylstm = mylstm(768, 768, MyLSTMConfig::default(), vb.clone())?;
        let input = vb.get((256, 1, 768), "input")?;
        let answer = vb.get((256, 1, 768), "output")?;

        let zero_state = mylstm.zero_state(1)?;

        let start = std::time::Instant::now();
        let output = mylstm.step(&input, &zero_state)?;
        //let output = mylstm.seq(&input)?;
        let output = mylstm.states_to_tensor(&output)?;
        println!("mylstm: {:?}", start.elapsed().as_secs_f32());
        assert_tensor(&output, &answer, 3, 1e-5)?;

        //let linear = linear(10, 10, vb)?;
        //linear.forward(&input)?;
        Ok(())
    }

    #[test]
    fn test_combine() -> Result<()> {
        let a = Tensor::arange(1.0_f32, 3.0, &Device::Cpu)?.reshape((1, 2))?;
        let b = Tensor::arange(3.0_f32, 5.0, &Device::Cpu)?.reshape((1, 2))?;
        println!("a: {:?}", a.to_vec2::<f32>()?);
        println!("b: {:?}", b.to_vec2::<f32>()?);

        let aa = Tensor::arange(5.0_f32, 7.0, &Device::Cpu)?;
        let bb = Tensor::arange(7.0_f32, 9.0, &Device::Cpu)?;
        println!("aa: {:?}", aa.to_vec1::<f32>()?);
        println!("bb: {:?}", bb.to_vec1::<f32>()?);

        let ia = Tensor::new(&[[10.0_f32]], &Device::Cpu)?;
        let ib = Tensor::new(&[[20.0_f32]], &Device::Cpu)?;

        let result_a = ia.matmul(&a)?.broadcast_add(&aa)?;
        let result_b = ib.matmul(&b)?.broadcast_add(&bb)?;
        let result = result_a.add(&result_b)?;
        println!("result: {:?}", result.to_vec2::<f32>()?);

        let start = std::time::Instant::now();
        for _i in 0..256 {
            let result_a = ia.matmul(&a)?.broadcast_add(&aa)?;
            let result_b = ib.matmul(&b)?.broadcast_add(&bb)?;
            result_a.add(&result_b)?;
        }
        let d1 = start.elapsed().as_secs_f32();

        let c = Tensor::cat(&[&a, &b], 0)?;
        println!("c: {:?}", c.to_vec2::<f32>()?);
        let bias = aa.add(&bb)?;
        println!("bias: {:?}", bias.to_vec1::<f32>()?);

        let i = Tensor::cat(&[&ia, &ib], D::Minus1)?;
        println!("i: {:?}", i.to_vec2::<f32>()?);
        let result_2 = i.matmul(&c)?.broadcast_add(&bias)?;
        println!("result_2: {:?}", result_2.to_vec2::<f32>()?);

        let start = std::time::Instant::now();
        for _i in 0..256 {
            let i = Tensor::cat(&[&ia, &ib], D::Minus1)?;
            i.matmul(&c)?.broadcast_add(&bias)?;
        }
        let d2 = start.elapsed().as_secs_f32();

        let c = Tensor::cat(&[&a, &b, &bias.unsqueeze(0)?], 0)?;
        println!("c: {:?}", c.to_vec2::<f32>()?);

        let i = Tensor::cat(&[&ia, &ib, &ia.ones_like()?], D::Minus1)?;
        let result_3 = i.matmul(&c)?;
        println!("result_3: {:?}", result_3.to_vec2::<f32>()?);

        let start = std::time::Instant::now();
        for _i in 0..256 {
            let i = Tensor::cat(&[&ia, &ib, &ia.ones_like()?], D::Minus1)?;
            i.matmul(&c)?;
        }
        let d3 = start.elapsed().as_secs_f32();

        println!("d1: {:?}", d1);
        println!("d2: {:?}", d2);
        println!("d3: {:?}", d3);
        println!("d2/d1: {:?}", d2 / d1);
        println!("d3/d1: {:?}", d3 / d1);

        Ok(())
    }

    #[test]
    fn simulate_input_layer() -> anyhow::Result<()> {
        let count = 256;
        let hi = Tensor::randn(0.0_f32, 1.0_f32, (768 * 4, 768), &Device::Cpu)?.t()?;
        let inputs = Tensor::randn(0.0_f32, 1.0_f32, (256, 1, 768), &Device::Cpu)?;

        let start = std::time::Instant::now();
        for i in 0..count {
            let input = inputs.get(i)?;
            let result = input.matmul(&hi)?;
            println!("result: {:?}", result.dims());
        }
        let d1 = start.elapsed().as_secs_f32();
        println!("d1: {:?}", d1);

        let inputs = inputs.transpose(0, 1)?.get(0)?;
        println!("inputs: {:?}", inputs.dims());
        let start = std::time::Instant::now();
        let result = inputs.matmul(&hi)?;
        println!("result: {:?}", result.dims());
        let d2 = start.elapsed().as_secs_f32();
        println!("d2: {:?}", d2);
        Ok(())
    }

    #[test]
    fn simulate_lstm() -> Result<()> {
        let count = 256;
        let hi = Tensor::randn(0.0_f32, 1.0_f32, (768 * 4, 768), &Device::Cpu)?.t()?;
        let hh = Tensor::randn(0.0_f32, 1.0_f32, (768 * 4, 768), &Device::Cpu)?.t()?;
        let bi = Tensor::randn(0.0_f32, 1.0_f32, 768 * 4, &Device::Cpu)?;
        let bh = Tensor::randn(0.0_f32, 1.0_f32, 768 * 4, &Device::Cpu)?;

        let inputs = Tensor::randn(0.0_f32, 1.0_f32, (256, 1, 768), &Device::Cpu)?;
        let hiddens = Tensor::randn(0.0_f32, 1.0_f32, (256, 1, 768), &Device::Cpu)?;

        let start = std::time::Instant::now();
        for i in 0..count {
            let input = inputs.get(i)?;
            let hidden = hiddens.get(i)?;
            let result_1 = input.matmul(&hi)?.broadcast_add(&bi)?;
            let result_2 = hidden.matmul(&hh)?.broadcast_add(&bh)?;
            let result = result_1.add(&result_2)?;

            let chunks = result.chunk(4, 1)?;
            let in_gate = candle_nn::ops::sigmoid(&chunks[0])?;
            let forget_gate = candle_nn::ops::sigmoid(&chunks[1])?;
            let cell_gate = chunks[2].tanh()?;
            let out_gate = candle_nn::ops::sigmoid(&chunks[3])?;

            //let next_c = ((forget_gate * &in_state.c)? + (in_gate * cell_gate)?)?;
            //let next_h = (out_gate * next_c.tanh()?)?;
        }
        let d1 = start.elapsed().as_secs_f32();
        println!("d1: {:?}", d1);

        let bias = bi.add(&bh)?;
        let start = std::time::Instant::now();
        for i in 0..count {
            let input = inputs.get(i)?;
            let hidden = hiddens.get(i)?;
            input
                .matmul(&hi)?
                .add(&hidden.matmul(&hh)?.broadcast_add(&bias)?)?;
        }
        let d2 = start.elapsed().as_secs_f32();
        println!("d2: {:?}", d2);

        let matrix = Tensor::cat(&[&hi, &hh], 0)?;
        let start = std::time::Instant::now();
        for i in 0..count {
            let input = inputs.get(i)?;
            let hidden = hiddens.get(i)?;
            let input = Tensor::cat(&[&input, &hidden], D::Minus1)?.contiguous()?;
            input.matmul(&matrix)?.broadcast_add(&bias)?;
        }
        let d3 = start.elapsed().as_secs_f32();
        println!("d3: {:?}", d3);

        let start = std::time::Instant::now();
        for i in 0..count {
            let input = inputs.get(i)?;
            let hidden = hiddens.get(i)?;
            let result_1 = input.matmul(&hi)?;
            let result_2 = hidden.matmul(&hh)?;
            result_1.add(&result_2)?;
        }
        let d4 = start.elapsed().as_secs_f32();
        println!("d4: {:?}", d4);
        Ok(())
    }
}
