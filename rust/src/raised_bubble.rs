use ::f128::f128;
use core::f64;
use std::ops::{Add, Div, Mul, Sub};

use num::Complex;
use serde::{Deserialize, Serialize};
use symbolica::numerical_integration::{ContinuousGrid, Grid, Sample};

use crate::{utils::FloatLike, HardCodedIntegrandSettings, HasIntegrand, Settings};
pub struct RaisedBubble {
    settings: Settings,
    p0: f64,
    mass_sq: f64,
    force_orientation: Option<usize>,
    threshold: Option<f64>,
}

impl RaisedBubble {
    pub fn new(settings: Settings) -> Self {
        let raised_bubble_settings = match &settings.hard_coded_integrand {
            HardCodedIntegrandSettings::RaisedBubble(settings) => settings,
            _ => unreachable!(),
        };

        let s = raised_bubble_settings.s;
        assert!(s > 0., "only positive s are is supported at the moment");

        let p0 = s.sqrt();

        let mass_sq = raised_bubble_settings.mass_sq;
        let force_orientation = raised_bubble_settings.force_orientation;
        if let Some(orientation) = force_orientation {
            assert!(
                orientation < 6,
                "there are only 6 orientations in this graph"
            )
        }

        let threshold_sq = p0 * p0 / 4. - mass_sq;

        let threshold = if threshold_sq > 0. {
            println!("evaluating raised bubble above threshold");
            Some(threshold_sq.sqrt())
        } else {
            println!("evaluating raised bubble below threshold");
            None
        };

        Self {
            settings,
            p0,
            mass_sq,
            force_orientation,
            threshold,
        }
    }

    fn evaluate_energy<T: FloatLike>(&self, r: Dual<T>) -> Dual<T> {
        (r * &r + &Into::<T>::into(self.mass_sq)).sqrt()
    }

    fn evaluate_eta_internal<T: FloatLike>(&self, r: Dual<T>) -> Dual<T> {
        let energy = self.evaluate_energy(r);
        energy + &energy
    }

    fn evaluate_eta_exist<T: FloatLike>(&self, r: Dual<T>) -> Dual<T> {
        let energy = self.evaluate_energy(r);
        energy + &energy - &Into::<T>::into(self.p0)
    }

    fn evaluate_eta_nonexist<T: FloatLike>(&self, r: Dual<T>) -> Dual<T> {
        let energy = self.evaluate_energy(r);
        energy + &energy + &Into::<T>::into(self.p0)
    }

    fn cff_orientation_1<T: FloatLike>(&self, r: T) -> T {
        if self.settings.general.debug > 0 {
            println!("evaluating cff orientation 1");
        }

        let dual_r = Dual::new(r);
        let energy = self.evaluate_energy(dual_r);
        let jacobian = dual_r * &dual_r;

        let eta_exist = self.evaluate_eta_exist(dual_r);

        let dual_res =
            jacobian * &(energy * &energy * &energy).inv() * &(eta_exist * &eta_exist).inv();

        if self.settings.general.debug > 0 {
            println!("bare result: {}", dual_res.real);
        }

        if let Some(threshold) = self.threshold {
            let (rplus, rminus) = (
                Dual::new(Into::<T>::into(threshold)),
                Dual::new(-Into::<T>::into(threshold)),
            );

            let (energy_plus, energy_minus) =
                (self.evaluate_energy(rplus), self.evaluate_energy(rminus));

            let (eta_exist_plus, eta_exist_minus) = (
                self.evaluate_eta_exist(rplus),
                self.evaluate_eta_exist(rminus),
            );

            let ct_plus_builder = (rplus * &rplus) / &(energy_plus * &energy_plus * &energy_plus);

            let ct_minus_builder =
                (rminus * &rminus) / &(energy_minus * &energy_minus * &energy_minus);

            let qudratic_ct_plus = ct_plus_builder.real
                / (eta_exist_plus.eps * eta_exist_plus.eps)
                / ((r - rplus.real) * (r - rplus.real));

            let qudratic_ct_minus = ct_minus_builder.real
                / (eta_exist_minus.eps * eta_exist_minus.eps)
                / ((r - rminus.real) * (r - rminus.real));

            let linear_ct_plus = -ct_plus_builder.eps
                / (eta_exist_plus.eps * eta_exist_plus.eps * (r - rplus.real))
                * unnormalized_gaussian(r - rplus.real);

            let linear_ct_minus = -ct_minus_builder.eps
                / (eta_exist_minus.eps * eta_exist_minus.eps * (r - rminus.real))
                * unnormalized_gaussian(r - rminus.real);

            let res = dual_res.real
                - qudratic_ct_plus
                - qudratic_ct_minus
                - linear_ct_plus
                - linear_ct_minus;

            if self.settings.general.debug > 0 {
                println!("quadratic_ct_plus: {}", qudratic_ct_plus);
                println!("quadratic_ct_minus: {}", qudratic_ct_minus);
                println!("linear_ct_plus: {}", linear_ct_plus);
                println!("linear_ct_minus: {}", linear_ct_minus);
                println!(
                    "bare - quadratic: {}",
                    dual_res.real - qudratic_ct_plus - qudratic_ct_minus
                );
                println!("orientation_result: {}", res);
            }

            res
        } else {
            dual_res.real
        }
    }

    fn cff_orientation_2<T: FloatLike>(&self, r: T) -> T {
        let dual_r = Dual::new(r);
        let energy = self.evaluate_energy(dual_r);
        let jacobian = dual_r * &dual_r;

        let eta_nonexist = self.evaluate_eta_nonexist(dual_r);

        let dual_res =
            jacobian * &(energy * &energy * &energy).inv() * &(eta_nonexist * &eta_nonexist).inv();

        dual_res.real
    }

    fn cff_orientation_3<T: FloatLike>(&self, r: T) -> T {
        let dual_r = Dual::new(r);
        let energy = self.evaluate_energy(dual_r);
        let jacobian = dual_r * &dual_r;

        let eta_exist = self.evaluate_eta_exist(dual_r);
        let eta_internal = self.evaluate_eta_internal(dual_r);

        let dual_res =
            jacobian * &(energy * &energy * &energy).inv() * &(eta_exist * &eta_internal).inv();

        if let Some(threshold) = self.threshold {
            let (rplus, rminus) = (
                Dual::new(Into::<T>::into(threshold)),
                Dual::new(Into::<T>::into(-threshold)),
            );

            let (energy_plus, energy_minus) =
                (self.evaluate_energy(rplus), self.evaluate_energy(rminus));

            let (eta_exist_plus, eta_exist_minus) = (
                self.evaluate_eta_exist(rplus),
                self.evaluate_eta_exist(rminus),
            );

            let (eta_internal_plus, eta_internal_minus) = (
                self.evaluate_eta_internal(rplus),
                self.evaluate_eta_internal(rminus),
            );

            let ct_plus_builder = (rplus * &rplus)
                / &(energy_plus * &energy_plus * &energy_plus * &eta_internal_plus);

            let linear_ct_plus = ct_plus_builder.real / ((r - rplus.real) * eta_exist_plus.eps)
                * unnormalized_gaussian(r - rplus.real);

            let ct_minus_builder = (rminus * &rminus)
                / &(energy_minus * &energy_minus * &energy_minus * &eta_internal_minus);

            let linear_ct_minus = ct_minus_builder.real / ((r - rminus.real) * eta_exist_minus.eps)
                * unnormalized_gaussian(r - rminus.real);

            dual_res.real - linear_ct_minus - linear_ct_plus
        } else {
            dual_res.real
        }
    }

    fn cff_orientation_4<T: FloatLike>(&self, r: T) -> T {
        self.cff_orientation_3(r)
    }

    fn cff_orientation_5<T: FloatLike>(&self, r: T) -> T {
        let dual_r = Dual::new(r);
        let energy = self.evaluate_energy(dual_r);
        let jacobian = dual_r * &dual_r;

        let eta_nonexist = self.evaluate_eta_nonexist(dual_r);
        let eta_internal = self.evaluate_eta_internal(dual_r);

        let dual_res =
            jacobian * &(energy * &energy * &energy).inv() * &(eta_nonexist * &eta_internal).inv();

        dual_res.real
    }

    fn cff_orientation_6<T: FloatLike>(&self, r: T) -> T {
        self.cff_orientation_5(r)
    }

    fn get_radius_from_sample(&self, sample: &Sample<f64>) -> (f64, f64) {
        let x = match sample {
            Sample::Continuous(_, xs) => xs[0],
            _ => unreachable!(),
        };

        let r = self.p0 * (1. / (1. - x) - 1. / x);
        let jac = self.p0 * (1. / (x * x) + 1. / ((1. - x) * (1. - x)));
        (r, jac)
    }

    fn evaluate_impl<T: FloatLike + Into<f64>>(&self, r: T) -> f64 {
        let eight = Into::<T>::into(8.);
        let pi = Into::<T>::into(f64::consts::PI);
        let pi_factor = eight * pi * pi * pi;
        let pysecdec_fudge_factor = -Into::<T>::into(16.) * pi * pi;
        let angular_integral = Into::<T>::into(2.) * pi;

        if self.settings.general.debug > 0 {
            println!("force_orientation: {:?}", self.force_orientation);
        }

        let res = match self.force_orientation {
            None => {
                self.cff_orientation_1(r)
                    + self.cff_orientation_2(r)
                    + self.cff_orientation_3(r)
                    + self.cff_orientation_4(r)
                    + self.cff_orientation_5(r)
                    + self.cff_orientation_6(r)
            }
            Some(i) => match i {
                0 => self.cff_orientation_1(r),
                1 => self.cff_orientation_2(r),
                2 => self.cff_orientation_3(r),
                3 => self.cff_orientation_4(r),
                4 => self.cff_orientation_5(r),
                5 => self.cff_orientation_6(r),
                _ => unreachable!(),
            },
        };

        let numerical_factor = pysecdec_fudge_factor * angular_integral / pi_factor / eight;
        if self.settings.general.debug > 0 {
            println!("orientation_sum: {}", res);
            println!("numerical factor: {}", numerical_factor);
        }

        Into::<f64>::into(res * numerical_factor)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaisedBubbleSettings {
    s: f64,
    mass_sq: f64,
    force_orientation: Option<usize>,
}

impl HasIntegrand for RaisedBubble {
    fn get_n_dim(&self) -> usize {
        1
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
        _wgt: f64,
        _iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let (r, jac) = self.get_radius_from_sample(sample);

        if self.settings.general.debug > 0 {
            println!("radius: {}", r);
            println!("jac: {}", jac);
        }

        if use_f128 {
            let r128 = f128::new(r);
            let res = self.evaluate_impl(r128) * jac;
            Complex { re: res, im: 0.0 }
        } else {
            match self.threshold {
                None => {
                    let res = self.evaluate_impl(r) * jac;
                    Complex { re: res, im: 0.0 }
                }
                Some(threshold) => {
                    let diff = (r.abs() - threshold).abs();

                    if self.settings.general.debug > 0 {
                        println!("distance_to_threshold: {}", diff);
                    }
                    if diff < self.p0 * 0.01 {
                        self.evaluate_sample(sample, _wgt, _iter, true)
                    } else {
                        let res = self.evaluate_impl(r) * jac;
                        Complex { re: res, im: 0.0 }
                    }
                }
            }
        }
    }
}

fn unnormalized_gaussian<T: FloatLike>(x: T) -> T {
    (-x * x).exp()
}

#[derive(Clone, Copy, Debug)]
struct Dual<T: FloatLike> {
    real: T,
    eps: T,
}

impl<T: FloatLike> Mul<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: self.real * rhs.real,
            eps: self.eps * rhs.real + self.real * rhs.eps,
        }
    }
}

impl<T: FloatLike> Add<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: self.real + rhs.real,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<T: FloatLike> Div<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: self.real / rhs.real,
            eps: (self.eps * rhs.real - self.real * rhs.eps) / (rhs.real * rhs.real),
        }
    }
}

impl<T: FloatLike> Sub<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: self.real - rhs.real,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<T: FloatLike> Mul<&T> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: &T) -> Self::Output {
        Self::Output {
            real: self.real * rhs,
            eps: self.eps * rhs,
        }
    }
}

impl<T: FloatLike> Add<&T> for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: &T) -> Self::Output {
        Self::Output {
            real: self.real + rhs,
            eps: self.eps,
        }
    }
}

impl<T: FloatLike> Div<&T> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: &T) -> Self::Output {
        Self::Output {
            real: self.real / rhs,
            eps: self.eps / rhs,
        }
    }
}

impl<T: FloatLike> Sub<&T> for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: &T) -> Self::Output {
        Self::Output {
            real: self.real - rhs,
            eps: self.eps,
        }
    }
}

impl<T: FloatLike> Dual<T> {
    fn new(real: T) -> Self {
        Self {
            real,
            eps: T::one(),
        }
    }

    fn sqrt(&self) -> Self {
        Self {
            real: self.real.sqrt(),
            eps: self.eps * Into::<T>::into(0.5) * self.real.sqrt().inv(),
        }
    }

    fn inv(&self) -> Self {
        Self {
            real: self.real.inv(),
            eps: -self.eps / self.real,
        }
    }
}
