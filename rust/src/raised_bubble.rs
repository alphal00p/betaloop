use core::f64;
use std::ops::{Add, Mul};

use havana::{ContinuousGrid, Grid, Sample};
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::{
    utils::{global_parameterize, FloatLike},
    HardCodedIntegrandSettings, HasIntegrand, Settings,
};

pub struct RaisedBubble {
    settings: Settings,
    external_vec: Mom<f64>,
    p0: f64,
    mass: f64,
}

impl RaisedBubble {
    pub fn new(settings: Settings) -> Self {
        let raised_bubble_settings = match &settings.hard_coded_integrand {
            HardCodedIntegrandSettings::RaisedBubble(settings) => settings,
            _ => unreachable!(),
        };

        let p0 = raised_bubble_settings.external[0];
        let external_vec = Mom {
            kx: raised_bubble_settings.external[1],
            ky: raised_bubble_settings.external[2],
            kz: raised_bubble_settings.external[3],
        };

        let mass = raised_bubble_settings.mass;

        Self {
            settings,
            external_vec,
            p0,
            mass,
        }
    }

    fn evaluate_impl<T: FloatLike + Into<f64>>(&self, loop_momentum: Mom<T>, jacobian: T) -> f64 {
        let emr12 = loop_momentum;
        let emr3 = loop_momentum + self.external_vec.convert();
        let mass_t_sq: T = Into::<T>::into(self.mass) * Into::<T>::into(self.mass);
        let energy12 = (emr12.squared() + mass_t_sq).sqrt();
        let energy3 = (emr3.squared() + mass_t_sq).sqrt();

        let p0 = Into::<T>::into(self.p0);

        let eta1 = energy12 + energy12;
        let eta2 = energy12 + energy3 - p0;
        let eta3 = energy12 + energy3 - p0;
        let eta4 = energy12 + energy3 + p0;
        let eta5 = energy12 + energy3 + p0;

        let eight = Into::<T>::into(8.);

        let inv_energy_product = (eight * energy12 * energy12 * energy3).inv();

        let term_1 = (eta2 * eta3).inv();
        let term_2 = (eta4 * eta5).inv();
        let term_3 = eta1.inv() * (eta2.inv() + eta3.inv() + eta4.inv() + eta5.inv());

        let pi = f64::consts::PI;

        let pi_factor = eight * Into::<T>::into(pi * pi * pi);
        let pysecdec_fudge_factor = Into::<T>::into(16. * pi * pi);

        let res = (inv_energy_product * (term_1 + term_2 + term_3) / pi_factor)
            * jacobian
            * pysecdec_fudge_factor;

        if self.settings.general.debug > 0 {
            println!("emr12: {:?}", emr12);
            println!("emr3: {:?}", emr3);
            println!("p0: {}", p0);
            println!("m^2: {}", mass_t_sq);
            println!("energy12: {}", energy12);
            println!("energy3: {}", energy3);
            println!("eta1: {}", eta1);
            println!("eta2: {}", eta2);
            println!("eta3: {}", eta3);
            println!("eta4: {}", eta4);
            println!("eta5: {}", eta5);
        }

        Into::<f64>::into(res)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaisedBubbleSettings {
    external: [f64; 4],
    mass: f64,
}

impl HasIntegrand for RaisedBubble {
    fn get_n_dim(&self) -> usize {
        3
    }

    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(self.get_n_dim(), 64, 1000))
    }

    fn evaluate_sample(
        &mut self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let sample = match sample {
            Sample::ContinuousGrid(_, xs) => xs,
            _ => unreachable!(),
        };

        let (loop_momenta, jacobian) =
            global_parameterize(sample, self.settings.kinematics.e_cm, &self.settings, false);

        let k = Mom {
            kx: loop_momenta[0][0],
            ky: loop_momenta[0][1],
            kz: loop_momenta[0][2],
        };

        Complex {
            re: self.evaluate_impl(k, jacobian),
            im: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct Mom<T> {
    kx: T,
    ky: T,
    kz: T,
}

impl<T> Add<Mom<T>> for Mom<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Mom<T>) -> Self::Output {
        Self::Output {
            kx: self.kx + rhs.kx,
            ky: self.ky + rhs.ky,
            kz: self.kz + rhs.kz,
        }
    }
}

impl Mom<f64> {
    fn convert<T: FloatLike>(&self) -> Mom<T> {
        Mom::<T> {
            kx: self.kx.into(),
            ky: self.ky.into(),
            kz: self.kz.into(),
        }
    }
}

impl<T> Mom<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Copy,
{
    fn squared(&self) -> T {
        self.kx * self.kx + self.ky * self.ky + self.kz * self.kz
    }
}
