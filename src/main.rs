mod network;

use ndarray::{Array2, arr2};
use network::Network;

fn main() {
    
    // Just a quick test to see how backprop is behaving
    // Don't like lots of this but at least it's running
    let mut nn = Network::new(&[4usize, 20usize, 20usize, 4usize]);
    let input: Array2<f64> = arr2(&[[10.0], [13.0], [13.0], [30.0]]);
    let output: Array2<f64> = arr2(&[[10.0], [13.0], [13.0], [30.0]]);

    let (nabla_b, nabla_w) = nn.backprop(&input, &output);

    println!("{}", nabla_b.len());
    for n in nabla_b {
        println!("{}", n);
    }
}
