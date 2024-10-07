use ::f128::f128;
use core::f64;
use std::ops::{Add, Mul};

use num::Complex;
use serde::{Deserialize, Serialize};
use symbolica::numerical_integration::{ContinuousGrid, Grid, Sample};

use crate::{
    utils::{global_parameterize, FloatLike},
    HardCodedIntegrandSettings, HasIntegrand, Settings,
};

pub struct RaisedBubble {
    settings: Settings,
    p0: f64,
    mass: f64,
}

impl RaisedBubble {
    pub fn new(settings: Settings) -> Self {
        let raised_bubble_settings = match &settings.hard_coded_integrand {
            HardCodedIntegrandSettings::RaisedBubble(settings) => settings,
            _ => unreachable!(),
        };

        let p0 = raised_bubble_settings.external_energy;
        assert!(
            p0 > 0.,
            "only positive energies are supported at the moment"
        );

        let mass = raised_bubble_settings.mass;

        Self { settings, p0, mass }
    }

    fn evaluate_impl<T: FloatLike + Into<f64>>(&self, loop_momentum: Mom<T>, jacobian: T) -> f64 {
        let pi = Into::<T>::into(f64::consts::PI);

        let mass_t_sq: T = Into::<T>::into(self.mass) * Into::<T>::into(self.mass);
        let energy = (loop_momentum.squared() + mass_t_sq).sqrt();

        let p0 = Into::<T>::into(self.p0);
        let two = Into::<T>::into(2.);

        let eta1 = energy * two;
        let eta2 = energy * two - p0;
        let eta3 = eta2;
        let eta4 = energy * two + p0;
        let eta5 = eta4;

        let eight = Into::<T>::into(8.);

        let inv_energy_product = (energy * energy * energy).inv();

        let term_1 = inv_energy_product * (eta2 * eta3).inv() * jacobian;
        let term_2 = inv_energy_product * (eta4 * eta5).inv() * jacobian;
        let term_3 = inv_energy_product * (eta1 * eta2).inv() * jacobian;
        let term_4 = inv_energy_product * (eta1 * eta3).inv() * jacobian;
        let term_5 = inv_energy_product * (eta1 * eta4).inv() * jacobian;
        let term_6 = inv_energy_product * (eta1 * eta5).inv() * jacobian;

        if self.settings.general.debug > 0 {
            println!("term_1: {}", term_1);
            println!("term_2: {}", term_2);
            println!("term_3: {}", term_3);
            println!("term_4: {}", term_4);
            println!("term_5: {}", term_5);
            println!("term_6: {}", term_6);
        }

        let bare_res = term_1 + term_2 + term_3 + term_4 + term_5 + term_6;

        let pi_factor = eight * Into::<T>::into(pi * pi * pi);
        let pysecdec_fudge_factor = Into::<T>::into(16.) * pi * pi;

        let mut ct = T::zero();
        let threshold = p0 * p0 / Into::<T>::into(4.) - mass_t_sq;

        if threshold > T::zero() {
            let radius = loop_momentum.hemispherical_norm();

            let r_plus = threshold.sqrt();
            let r_minus = -r_plus;

            if self.settings.general.debug > 0 {
                println!("radius: {}", radius);
                println!("rstar: {}", r_plus);
            }

            let energy_at_threshold = (r_plus * r_plus + mass_t_sq).sqrt();
            let energy_product_at_threshold =
                energy_at_threshold * energy_at_threshold * energy_at_threshold;

            let energy_derivative_plus = r_plus / energy_at_threshold;
            let energy_derivative_minus = r_minus / energy_at_threshold;

            let eta_derivative_plus = energy_derivative_plus * two;
            let eta_derivative_minus = energy_derivative_minus * two;

            let eta1_at_threshold = energy_at_threshold * two;

            let ct1 = (energy_product_at_threshold
                * (radius - r_plus)
                * (radius - r_plus)
                * eta_derivative_plus
                * eta_derivative_plus)
                .inv()
                * (r_plus / radius).powi(2)
                * jacobian;

            ct += ct1;

            if self.settings.general.debug > 0 {
                println!("ct1: {}", ct1);
            }

            let ct2 = (energy_product_at_threshold
                * (radius - r_minus)
                * (radius - r_minus)
                * eta_derivative_minus
                * eta_derivative_minus)
                .inv()
                * (r_minus / radius).powi(2)
                * jacobian;

            ct += ct2;

            //  ct += (energy_product_at_threshold
            //      * eta_derivative_minus
            //      * eta_derivative_minus
            //      * r_minus.powi(1))
            //  .inv()
            //      * pi
            //      * normalized_gaussian(radius)
            //      / (radius * radius)
            //      * (jacobian);

            if self.settings.general.debug > 0 {
                println!("ct2: {}", ct2);
            }

            let ct3 = (two * r_plus / energy_product_at_threshold
                + Into::<T>::into(-3.) * r_plus.powi(2) / (energy_at_threshold).powi(4)
                    * energy_derivative_plus)
                / eta_derivative_plus.powi(2)
                * (radius - r_plus).inv()
                / (radius * radius)
                * (-(radius - r_plus).powi(2)).exp()
                * jacobian;

            ct += ct3;

            if self.settings.general.debug > 0 {
                println!("ct3: {}", ct3);
                println!("ct1 + ct3: {}", ct1 + ct3);
                println!("term1 - ct1 - ct3: {}", term_1 - ct1 - ct3)
            }

            let ct4 = (two * r_minus / energy_product_at_threshold
                + Into::<T>::into(-3.) * r_minus.powi(2) / energy_at_threshold.powi(4)
                    * energy_derivative_minus)
                / (eta_derivative_minus).powi(2)
                * (radius - r_minus).inv()
                / (radius * radius)
                * (-(radius - r_minus).powi(2)).exp()
                * jacobian;

            ct += ct4;

            if self.settings.general.debug > 0 {
                println!("ct4: {}", ct4);
            }

            let ct5 = two
                * (energy_product_at_threshold
                    * eta1_at_threshold
                    * eta_derivative_plus
                    * (radius - r_plus))
                    .inv()
                * (r_plus / radius).powi(2)
                * (-(radius - r_plus).powi(2)).exp()
                * jacobian;

            ct += ct5;

            if self.settings.general.debug > 0 {
                println!("ct5: {}", ct5);
                println!("term3 + term4 - ct5: {}", term_3 + term_4 - ct5);
            }

            let ct6 = two
                * (energy_product_at_threshold
                    * eta1_at_threshold
                    * eta_derivative_minus
                    * (radius - r_minus))
                    .inv()
                * (r_minus / radius).powi(2)
                * (-(radius - r_minus).powi(2)).exp()
                * jacobian;
            ct += ct6;

            if self.settings.general.debug > 0 {
                println!("ct6: {}", ct6);
            }
        }

        let res = bare_res - ct;

        if self.settings.general.debug > 0 {
            println!("loop_momenta {:?}", loop_momentum);
            println!("p0: {}", p0);
            println!("m^2: {}", mass_t_sq);
            println!("energy: {}", energy);
            println!("eta1: {}", eta1);
            println!("eta2: {}", eta2);
            println!("eta3: {}", eta3);
            println!("eta4: {}", eta4);
            println!("eta5: {}", eta5);
            println!("bare_res: {}", bare_res);
            println!("ct: {}", ct);
            println!("diff: {}", res);
        }

        Into::<f64>::into(res / pi_factor * pysecdec_fudge_factor / eight)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaisedBubbleSettings {
    external_energy: f64,
    mass: f64,
}

impl HasIntegrand for RaisedBubble {
    fn get_n_dim(&self) -> usize {
        3
    }

    fn create_grid(&self) -> Grid<f64> {
        Grid::Continuous(ContinuousGrid::new(
            self.get_n_dim(),
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
            None,
            self.settings.integrator.train_on_avg,
        ))
    }

    fn evaluate_sample(
        &mut self,
        sample: &Sample<f64>,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let raw_sample = match sample {
            Sample::Continuous(_, xs) => xs,
            _ => unreachable!(),
        };

        let (loop_momenta, jacobian) = global_parameterize(
            raw_sample,
            self.settings.kinematics.e_cm,
            &self.settings,
            false,
        );

        let k = Mom {
            kx: loop_momenta[0][0],
            ky: loop_momenta[0][1],
            kz: loop_momenta[0][2],
        };

        if use_f128 {
            let k: Mom<f128> = k.convert();
            let jacobian = f128::new(jacobian);

            Complex {
                re: self.evaluate_impl(k, jacobian),
                im: 0.0,
            }
        } else {
            let res = Complex {
                re: self.evaluate_impl(k, jacobian),
                im: 0.0,
            };

            if res.re > 10e5 {
                self.evaluate_sample(sample, wgt, iter, true)
            } else {
                res
            }
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
    T: Add<T, Output = T>,
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

impl<T> Mul<T> for Mom<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            kx: self.kx * rhs,
            ky: self.ky * rhs,
            kz: self.kz * rhs,
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

impl<T: FloatLike> Mom<T> {
    fn norm(&self) -> T {
        self.squared().sqrt()
    }

    fn hemispherical_norm(&self) -> T {
        if self.kx.is_positive() {
            self.norm()
        } else {
            -self.norm()
        }
    }
}

fn normalized_gaussian<T: FloatLike>(x: T) -> T {
    let sqrt_pi = f64::consts::PI.sqrt();

    (-x * x).exp() / Into::<T>::into(sqrt_pi)
}
