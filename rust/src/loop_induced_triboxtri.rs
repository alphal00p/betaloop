use crate::integrands::*;
use crate::utils;
use crate::utils::FloatLike;
use crate::Settings;
use havana::{ContinuousGrid, Grid, Sample};
use lorentz_vector::LorentzVector;
use num::Complex;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoopInducedTriBoxTriSettings {
    pub supegraph_yaml_file: String,
}

pub struct LoopInducedTriBoxTriIntegrand {
    pub settings: Settings,
    pub supergraph: SuperGraph,
    pub n_dim: usize,
}

#[allow(unused)]
impl LoopInducedTriBoxTriIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: LoopInducedTriBoxTriSettings,
    ) -> LoopInducedTriBoxTriIntegrand {
        /*
               output_LU_scalar betaLoop_triangleBoxTriangleBenchmark_scalar \
        --topology=[\
           ('q1', 0, 1), \
           ('2', 1, 3), ('7', 3, 4),('4', 4, 5), \
           ('3', 5, 6), ('6', 6, 7),('1', 7, 1), \
           ('5',3,7),('8',4,6),\
           ('q2', 5, 8) ] \
        --masses=[('2', 10.),('1', 10.),('5', 10.),('3', 20.),('4', 20.),('8', 20.),('7', 0.),('6', 0.),] \
        --lmb=('2', '4', '7') \
        --name="betaLoop_triangleBoxTriangleBenchmark_scalar" \
        --analytical_result=0.0e0 \
        --externals=('q1',) \
        --numerator='1'
               */

        let sg = SuperGraph::from_file(integrand_settings.supegraph_yaml_file.as_str()).unwrap();
        let n_dim = utils::get_n_dim_for_n_loop_momenta(&settings, sg.n_loop, true);
        LoopInducedTriBoxTriIntegrand {
            settings,
            supergraph: sg,
            n_dim: n_dim,
        }
    }

    fn evaluate_numerator<T: FloatLike>(&self, loop_momenta: &[LorentzVector<T>]) -> T {
        return T::from_f64(1.0).unwrap();
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            true,
        )
    }
}

#[allow(unused)]
impl HasIntegrand for LoopInducedTriBoxTriIntegrand {
    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(
            self.n_dim,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
        ))
    }

    fn get_n_dim(&self) -> usize {
        return self.n_dim;
    }

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let xs = match sample {
            Sample::ContinuousGrid(_w, v) => v,
            _ => panic!("Wrong sample type"),
        };
        let mut sample_xs = vec![self.settings.kinematics.e_cm];
        sample_xs.extend(xs);
        if self.settings.general.debug > 1 {
            println!(
                "Sampled x-space     : ( {} )",
                sample_xs
                    .iter()
                    .map(|&x| format!("{:.16}", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let (moms, jac) = self.parameterize(sample_xs.as_slice());
        let mut loop_momenta = vec![];
        for m in &moms {
            loop_momenta.push(LorentzVector {
                t: ((m[0] + m[1] + m[2]) * (m[0] + m[1] + m[2])).sqrt(),
                x: m[0],
                y: m[1],
                z: m[2],
            });
        }
        let mut itg_wgt = self.evaluate_numerator(loop_momenta.as_slice());
        if self.settings.general.debug > 1 {
            println!("Sampled loop momenta:");
            for (i, l) in loop_momenta.iter().enumerate() {
                println!(
                    "k{} = ( {:-23}, {:-23}, {:-23}, {:-23} )",
                    i,
                    format!("{:+.16e}", l.t),
                    format!("{:+.16e}", l.x),
                    format!("{:+.16e}", l.y),
                    format!("{:+.16e}", l.z)
                );
            }
            println!("Integrator weight : {:+.16e}", wgt);
            println!("Integrand weight  : {:+.16e}", itg_wgt);
            println!("Sampling jacobian : {:+.16e}", jac);
            println!("Final contribution: {:+.16e}", itg_wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * jac;
    }
}
