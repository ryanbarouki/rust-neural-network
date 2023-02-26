use rand_distr::StandardNormal;
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp()) 
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn to_tuple(inp: &[usize]) -> (usize, usize) { 
    match inp {
        [a, b] => (*a, *b),
        _ => panic!(),
    }
}

fn zero_vec_like(arr: &[Array2<f64>]) -> Vec<Array2<f64>> {
    arr.iter().map(|x| Array2::zeros(to_tuple(x.shape())))
        .collect()
}

struct TrainingData {
    input: Array2<f64>,
    output: Array2<f64>,
}

#[derive(Debug)]
pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(sizes: &[usize]) -> Network {
        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();

        for i in 1..num_layers { 
            biases.push(Array::random((sizes[i], 1), StandardNormal));
            weights.push(Array::random((sizes[i], sizes[i-1]), StandardNormal));
        }

        Network {
            num_layers: num_layers, 
            sizes: sizes.to_owned(), 
            biases: biases, 
            weights: weights, 
        }
    }

    fn update_mini_batch(&mut self, 
                         training_data: &[TrainingData],
                         learning_rate: f64)
    {
        let mut nabla_b: Vec<Array2<f64>> = zero_vec_like(&self.biases);
        let mut nabla_w: Vec<Array2<f64>> = zero_vec_like(&self.weights);

        for TrainingData {input, output} in training_data {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(input, output);
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
        }

        let nbatch = training_data.len() as f64;

        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            *w -= &nw.mapv(|x| x*learning_rate / nbatch);
        }

        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
            *b -= &nb.mapv(|x| x*learning_rate / nbatch);
        }
    }

    pub fn backprop(&mut self, input: &Array2<f64>, output: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array2<f64>> = zero_vec_like(&self.biases);
        let mut nabla_w: Vec<Array2<f64>> = zero_vec_like(&self.weights);

        // feedforward

        let mut activations: Vec<Array2<f64>> = Vec::new();
        activations.push(input.clone());
        let mut zs: Vec<Array2<f64>> = Vec::new();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            println!("{}x{}", activations.last().unwrap().shape()[0], activations.last().unwrap().shape()[1]);
            let z = w.dot(activations.last().unwrap()) + b;
            zs.push(z.clone());
            activations.push(z.mapv(|x| sigmoid(x)));
        } 

        // backwards pass
        let mut delta = self.cost_func_derivative(activations.last().unwrap(), output) * 
            zs.last().unwrap().mapv(|x| sigmoid_derivative(x));
       
        *nabla_b.last_mut().unwrap() = delta.clone();
        *nabla_w.last_mut().unwrap() = delta.dot(&activations[activations.len() - 2].clone().reversed_axes());
        
        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = z.mapv(|x| sigmoid_derivative(x));
            delta = self.weights[self.weights.len()-l+1].clone().reversed_axes().dot(&delta) * sp;

            let len_nb = nabla_b.len();
            let len_nw = nabla_w.len();
            nabla_b[len_nb-l] = delta.clone();
            println!("{}x{}", delta.shape()[0], delta.shape()[1]);
            nabla_w[len_nw-l] = delta.dot(&activations[activations.len()-l-1].clone().reversed_axes());
        }

        (nabla_b, nabla_w)
    }

    fn cost_func_derivative(&mut self, output_activations: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
        output_activations - output
    }
}
