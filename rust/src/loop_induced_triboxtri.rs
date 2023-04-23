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
    pub n_dim: usize,
}

pub struct LoopInducedTriBoxTrIntegrand {
    pub settings: Settings,
    pub supergraph: SuperGraph,
    pub n_dim: usize,
}

#[allow(unused)]
impl LoopInducedTriBoxTrIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: LoopInducedTriBoxTriSettings,
    ) -> LoopInducedTriBoxTrIntegrand {
        LoopInducedTriBoxTrIntegrand {
            settings,
            supergraph: SuperGraph::default(),
            n_dim: integrand_settings.n_dim,
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
impl HasIntegrand for LoopInducedTriBoxTrIntegrand {
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
        let (moms, jac) = self.parameterize(xs);
        return Complex::new(jac, 0.);
    }
}
