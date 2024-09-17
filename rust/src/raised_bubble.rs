use core::f64;
use num::Complex;
use ref_ops::{RefAdd, RefDiv, RefMul, RefNeg, RefSub};
use rug::{ops::CompleteRound, Float};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};
use symbolica::numerical_integration::{ContinuousGrid, Grid, Sample};

use crate::{HardCodedIntegrandSettings, HasIntegrand, Settings};
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

    fn evaluate_energy<T: RaisedBubbleFloat>(&self, r: &Dual<T>) -> Dual<T> {
        (r * r + r.real.from_f64(self.mass_sq)).sqrt()
    }

    fn evaluate_eta_internal<T: RaisedBubbleFloat>(&self, r: &Dual<T>) -> Dual<T> {
        let energy = self.evaluate_energy(r);
        &energy + &energy
    }

    fn evaluate_eta_exist<T: RaisedBubbleFloat>(&self, r: &Dual<T>) -> Dual<T> {
        let energy = self.evaluate_energy(r);
        &energy + &energy - r.real.from_f64(self.p0)
    }

    fn evaluate_eta_nonexist<T: RaisedBubbleFloat>(&self, r: &Dual<T>) -> Dual<T> {
        let energy = self.evaluate_energy(r);
        &energy + &energy + r.real.from_f64(self.p0)
    }

    fn evaluate_second_eta_dir<T: RaisedBubbleFloat>(&self, r: &F<T>) -> F<T> {
        let dual_r = Dual::new(r.clone());
        let energy = self.evaluate_energy(&dual_r).real;
        r.from_f64(2.0) * (-r * r / (&energy * &energy * &energy) + energy.inv())
    }

    fn cff_orientation_1<T: RaisedBubbleFloat>(&self, r: &F<T>) -> Complex<f64> {
        if self.settings.general.debug > 0 {
            println!("evaluating cff orientation 1");
        }

        let dual_r = Dual::new(r.clone());
        let energy = self.evaluate_energy(&dual_r);
        let jacobian = &dual_r * &dual_r;

        let eta_exist = self.evaluate_eta_exist(&dual_r);

        let dual_res =
            jacobian * (&energy * &energy * &energy).inv() * (&eta_exist * &eta_exist).inv();

        if self.settings.general.debug > 0 {
            println!("bare result: {}", dual_res.real);
        }

        if let Some(threshold) = self.threshold {
            let (rplus, rminus) = (
                Dual::new(r.from_f64(threshold)),
                Dual::new(-r.from_f64(threshold)),
            );

            let (energy_plus, energy_minus) =
                (self.evaluate_energy(&rplus), self.evaluate_energy(&rminus));

            if self.settings.general.debug > 0 {
                println!("energy_plus: {}", energy_plus);
                println!("energy_minus: {}", energy_minus);
            }

            let (eta_exist_plus, eta_exist_minus) = (
                self.evaluate_eta_exist(&rplus),
                self.evaluate_eta_exist(&rminus),
            );

            let ct_plus_builder = (&rplus * &rplus) / (&energy_plus * &energy_plus * &energy_plus);

            let ct_minus_builder =
                (&rminus * &rminus) / (&energy_minus * &energy_minus * &energy_minus);

            let qudratic_ct_plus = &ct_plus_builder.real
                / (&eta_exist_plus.eps * &eta_exist_plus.eps)
                / ((r - &rplus.real) * (r - &rplus.real));

            let qudratic_ct_minus = &ct_minus_builder.real
                / (&eta_exist_minus.eps * &eta_exist_minus.eps)
                / ((r - &rminus.real) * (r - &rminus.real));

            let p0 = r.from_f64(self.p0);
            let second_eta_dir_plus = self.evaluate_second_eta_dir(&rplus.real);

            let linear_ct_plus = (&ct_plus_builder.eps * &eta_exist_plus.eps
                - &second_eta_dir_plus * &ct_plus_builder.real)
                / (&eta_exist_plus.eps
                    * &eta_exist_plus.eps
                    * &eta_exist_plus.eps
                    * (r - &rplus.real))
                * unnormalized_gaussian(&(r - &rplus.real), &p0);

            let second_eta_dir_minus = self.evaluate_second_eta_dir(&rminus.real);

            let linear_ct_minus = (&ct_minus_builder.eps
                - &second_eta_dir_minus * &ct_minus_builder.real / &eta_exist_minus.eps)
                / (&eta_exist_minus.eps * &eta_exist_minus.eps * (r - &rminus.real))
                * unnormalized_gaussian(&(r - &rminus.real), &p0);

            let real_res = &dual_res.real
                - &qudratic_ct_plus
                - &qudratic_ct_minus
                - &linear_ct_plus
                - &linear_ct_minus;

            if self.settings.general.debug > 0 {
                println!("ct_builder_plus: {}", ct_plus_builder);
                println!("ct_builder_minus: {}", ct_minus_builder);
                println!("quadratic_ct_plus: {}", qudratic_ct_plus);
                println!("quadratic_ct_minus: {}", qudratic_ct_minus);
                println!("linear_ct_plus: {}", linear_ct_plus);
                println!("linear_ct_minus: {}", linear_ct_minus);
                println!(
                    "bare - quadratic: {}",
                    dual_res.real - qudratic_ct_plus - qudratic_ct_minus
                );
                println!("orientation_result: {}", real_res);
            }

            let pi = r.from_f64(f64::consts::PI);

            let integrated_ct_plus = &pi
                * (&ct_plus_builder.eps
                    - &second_eta_dir_plus * &ct_plus_builder.real / &eta_exist_plus.eps)
                / (&eta_exist_plus.eps * &eta_exist_plus.eps)
                * rplus.real.signum();

            let integrated_ct_minus = &pi
                * (&ct_minus_builder.eps
                    - &second_eta_dir_minus * &ct_minus_builder.real / &eta_exist_minus.eps)
                / (&eta_exist_minus.eps * &eta_exist_minus.eps)
                * rminus.real.signum();

            if self.settings.general.debug > 0 {
                println!("integrated_ct+ orientation 1: {}", integrated_ct_plus);
                println!("integrated_ct- orientation 1: {}", integrated_ct_minus);
                println!("eta_dir_plus: {}", eta_exist_plus.eps);
                println!("eta_dir_min: {}", eta_exist_minus.eps);
                println!("other_parth: {}", ct_plus_builder.eps);
                println!("other_parth: {}", ct_minus_builder.eps);
            }

            Complex::new(
                real_res.into(),
                (integrated_ct_minus * normalized_gaussian(r, &p0)
                    + integrated_ct_plus * normalized_gaussian(r, &p0))
                .into(),
            )
        } else {
            Complex {
                re: dual_res.real.into(),
                im: r.zero().into(),
            }
        }
    }

    fn cff_orientation_2<T: RaisedBubbleFloat>(&self, r: &F<T>) -> Complex<f64> {
        let dual_r = Dual::new(r.clone());
        let energy = self.evaluate_energy(&dual_r);
        let jacobian = &dual_r * &dual_r;

        let eta_nonexist = self.evaluate_eta_nonexist(&dual_r);

        let dual_res =
            &jacobian * (&energy * &energy * &energy).inv() * (&eta_nonexist * &eta_nonexist).inv();

        let real_part = dual_res.real.into();
        Complex::new(real_part, 0.0)
    }

    fn cff_orientation_3<T: RaisedBubbleFloat>(&self, r: &F<T>) -> Complex<f64> {
        let dual_r = Dual::new(r.clone());
        let energy = self.evaluate_energy(&dual_r);
        let jacobian = &dual_r * &dual_r;

        let eta_exist = self.evaluate_eta_exist(&dual_r);
        let eta_internal = self.evaluate_eta_internal(&dual_r);

        let dual_res =
            jacobian * (&energy * &energy * &energy).inv() * (&eta_exist * &eta_internal).inv();

        if let Some(threshold) = self.threshold {
            let (rplus, rminus) = (
                Dual::new(r.from_f64(threshold)),
                Dual::new(r.from_f64(-threshold)),
            );

            let (energy_plus, energy_minus) =
                (self.evaluate_energy(&rplus), self.evaluate_energy(&rminus));

            let (eta_exist_plus, eta_exist_minus) = (
                self.evaluate_eta_exist(&rplus),
                self.evaluate_eta_exist(&rminus),
            );

            let (eta_internal_plus, eta_internal_minus) = (
                self.evaluate_eta_internal(&rplus),
                self.evaluate_eta_internal(&rminus),
            );

            let ct_plus_builder = (&rplus * &rplus)
                / (&energy_plus * &energy_plus * &energy_plus * &eta_internal_plus);

            let p0 = r.from_f64(self.p0);
            let linear_ct_plus = &ct_plus_builder.real / ((r - &rplus.real) * &eta_exist_plus.eps)
                * unnormalized_gaussian(&(r - &rplus.real), &p0);

            let ct_minus_builder = (&rminus * &rminus)
                / (&energy_minus * &energy_minus * &energy_minus * &eta_internal_minus);

            let linear_ct_minus = &ct_minus_builder.real
                / ((r - &rminus.real) * &eta_exist_minus.eps)
                * unnormalized_gaussian(&(r - &rminus.real), &p0);

            let pi = r.from_f64(f64::consts::PI);

            let integrated_ct_plus =
                &pi * &ct_plus_builder.real / (&eta_exist_plus.eps) * rplus.real.signum();

            let integrated_ct_minus =
                &pi * &ct_minus_builder.real / (&eta_exist_minus.eps) * rminus.real.signum();
            if self.settings.general.debug > 0 {
                println!(
                    "integrated_ct orienatation3 {}",
                    &integrated_ct_plus + &integrated_ct_minus
                );
            }

            Complex::new(
                (dual_res.real - linear_ct_minus - linear_ct_plus).into(),
                (integrated_ct_plus * normalized_gaussian(r, &p0)
                    + integrated_ct_minus * normalized_gaussian(r, &p0))
                .into(),
            )
        } else {
            Complex::new(dual_res.real.into(), 0.0)
        }
    }

    fn cff_orientation_4<T: RaisedBubbleFloat>(&self, r: &F<T>) -> Complex<f64> {
        self.cff_orientation_3(r)
    }

    fn cff_orientation_5<T: RaisedBubbleFloat>(&self, r: &F<T>) -> Complex<f64> {
        let dual_r = Dual::new(r.clone());
        let energy = self.evaluate_energy(&dual_r);
        let jacobian = &dual_r * &dual_r;

        let eta_nonexist = self.evaluate_eta_nonexist(&dual_r);
        let eta_internal = self.evaluate_eta_internal(&dual_r);

        let dual_res =
            jacobian * (&energy * &energy * &energy).inv() * &(eta_nonexist * &eta_internal).inv();

        Complex::new(dual_res.real.into(), 0.0)
    }

    fn cff_orientation_6<T: RaisedBubbleFloat>(&self, r: &F<T>) -> Complex<f64> {
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

    fn evaluate_impl<T: RaisedBubbleFloat + Into<f64>>(&self, r: &F<T>) -> Complex<f64> {
        let eight = r.from_f64(8.0);
        let pi = r.from_f64(f64::consts::PI);

        let pi_factor = &eight * &pi * &pi * &pi;

        let pysecdec_fudge_factor = -r.from_f64(16.) * &pi * &pi;
        let angular_integral = r.from_f64(2.) * &pi;

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

        let scaled_res: Complex<f64> = res * Into::<f64>::into(numerical_factor);
        Complex::new(
            Into::<f64>::into(scaled_res.re),
            Into::<f64>::into(scaled_res.im),
        )
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
            let r128 = F(ArbPrec::<113>::new(r));

            if self.settings.general.debug > 0 {
                println!("using f128");
                println!("radius: {}", r128);
            }

            self.evaluate_impl(&r128) * jac
        } else {
            match self.threshold {
                None => self.evaluate_impl(&F(r)) * jac,
                Some(threshold) => {
                    let diff = (r.abs() - threshold).abs();
                    if self.settings.general.debug > 0 {
                        println!("threhsold: {}", threshold);
                    }

                    if self.settings.general.debug > 0 {
                        println!("distance_to_threshold: {}", diff);
                    }
                    if diff < self.p0 * 0.01 {
                        if diff < self.p0 * 0.001 {
                            let r256 = F(ArbPrec::<256>::new(r));
                            return self.evaluate_impl(&r256) * jac;
                        }
                        self.evaluate_sample(sample, _wgt, _iter, true)
                    } else {
                        self.evaluate_impl(&F(r)) * jac
                    }
                }
            }
        }
    }
}

fn unnormalized_gaussian<T: RaisedBubbleFloat>(x: &F<T>, p0: &F<T>) -> F<T> {
    (-x * x / (p0 * p0)).exp()
}

fn normalized_gaussian<T: RaisedBubbleFloat>(x: &F<T>, sigma: &F<T>) -> F<T> {
    let sqrt_pi = x.from_f64(f64::consts::PI.sqrt());
    let inv_sigma_quad = (sigma * sigma).inv();
    let minus_x_quad = -x * x;

    &sqrt_pi.inv() * &(sigma.inv() * &(&minus_x_quad * &inv_sigma_quad).exp())
}

#[derive(Clone, Debug)]
struct Dual<T: RaisedBubbleFloat> {
    real: F<T>,
    eps: F<T>,
}

impl<T: RaisedBubbleFloat> Display for Dual<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("real: {}, eps: {}", self.real, self.eps))
    }
}

impl<T: RaisedBubbleFloat> Mul<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: &self.real * &rhs.real,
            eps: (&self.eps * &rhs.real) + (&self.real * &rhs.eps),
        }
    }
}

impl<T: RaisedBubbleFloat> Mul<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: &Dual<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T: RaisedBubbleFloat> Mul<Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: Dual<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T: RaisedBubbleFloat> Mul<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: Dual<T>) -> Self::Output {
        self * &rhs
    }
}

impl<T: RaisedBubbleFloat> Add<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: &self.real + &rhs.real,
            eps: &self.eps + &rhs.eps,
        }
    }
}

impl<T: RaisedBubbleFloat> Add<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: &Dual<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T: RaisedBubbleFloat> Div<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: &self.real / &rhs.real,
            eps: (&self.eps * &rhs.real - &self.real * &rhs.eps) / (&rhs.real * &rhs.real),
        }
    }
}

impl<T: RaisedBubbleFloat> Div<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: &Dual<T>) -> Self::Output {
        &self / rhs
    }
}

impl<T: RaisedBubbleFloat> Div<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: Dual<T>) -> Self::Output {
        self / &rhs
    }
}

impl<T: RaisedBubbleFloat> Div<Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: Dual<T>) -> Self::Output {
        &self / &rhs
    }
}

impl<T: RaisedBubbleFloat> Sub<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: &Dual<T>) -> Self::Output {
        Self::Output {
            real: &self.real - &rhs.real,
            eps: &self.eps - &rhs.eps,
        }
    }
}

impl<T: RaisedBubbleFloat> Sub<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: &Dual<T>) -> Self::Output {
        &self - rhs
    }
}

impl<T: RaisedBubbleFloat> Mul<&F<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: &F<T>) -> Self::Output {
        Self::Output {
            real: &self.real * rhs,
            eps: &self.eps * rhs,
        }
    }
}

impl<T: RaisedBubbleFloat> Mul<F<T>> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: F<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T: RaisedBubbleFloat> Add<&F<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: &F<T>) -> Self::Output {
        Self::Output {
            real: &self.real + rhs,
            eps: self.eps.clone(),
        }
    }
}

impl<T: RaisedBubbleFloat> Add<F<T>> for Dual<T> {
    type Output = Dual<T>;
    fn add(self, rhs: F<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T: RaisedBubbleFloat> Div<&F<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: &F<T>) -> Self::Output {
        Self::Output {
            real: &self.real / rhs,
            eps: &self.eps / rhs,
        }
    }
}

impl<T: RaisedBubbleFloat> Div<&F<T>> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: &F<T>) -> Self::Output {
        &self / rhs
    }
}

impl<T: RaisedBubbleFloat> Div<F<T>> for Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: F<T>) -> Self::Output {
        &self / &rhs
    }
}

impl<T: RaisedBubbleFloat> Div<F<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn div(self, rhs: F<T>) -> Self::Output {
        self / &rhs
    }
}

impl<T: RaisedBubbleFloat> Sub<&F<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: &F<T>) -> Self::Output {
        Self::Output {
            real: &self.real - rhs,
            eps: self.eps.clone(),
        }
    }
}

impl<T: RaisedBubbleFloat> Sub<F<T>> for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: F<T>) -> Self::Output {
        &self - &rhs
    }
}

impl<T: RaisedBubbleFloat> Dual<T> {
    fn new(real: F<T>) -> Self {
        Self {
            eps: real.one(),
            real,
        }
    }

    fn sqrt(&self) -> Self {
        Self {
            real: self.real.sqrt(),
            eps: &self.eps * &(&self.real.from_f64(0.5) * &self.real.sqrt().inv()),
        }
    }

    fn inv(&self) -> Self {
        Self {
            real: self.real.inv(),
            eps: -&self.eps / &self.real,
        }
    }
}

trait RaisedBubbleFloat:
    Into<f64>
    + Clone
    + for<'a> RefMul<&'a Self, Output = Self>
    + for<'a> RefAdd<&'a Self, Output = Self>
    + for<'a> RefDiv<&'a Self, Output = Self>
    + for<'a> RefSub<&'a Self, Output = Self>
    + for<'a> RefNeg<Output = Self>
    + Display
    + PartialOrd
    + PartialEq
{
    fn sqrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn one(&self) -> Self;
    fn zero(&self) -> Self;
    #[allow(clippy::wrong_self_convention)]
    fn from_f64(&self, x: f64) -> Self;
}

impl RaisedBubbleFloat for f64 {
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    fn from_f64(&self, x: f64) -> Self {
        x
    }

    fn one(&self) -> Self {
        1.0
    }

    fn zero(&self) -> Self {
        0.0
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq)]
struct F<T: RaisedBubbleFloat>(T);

impl<T: RaisedBubbleFloat> Mul<&F<T>> for &F<T> {
    type Output = F<T>;
    fn mul(self, rhs: &F<T>) -> Self::Output {
        F(self.0.ref_mul(&rhs.0))
    }
}

impl<T: RaisedBubbleFloat> Mul<&F<T>> for F<T> {
    type Output = F<T>;
    fn mul(self, rhs: &F<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T: RaisedBubbleFloat> Mul<F<T>> for F<T> {
    type Output = F<T>;
    fn mul(self, rhs: F<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T: RaisedBubbleFloat> Mul<F<T>> for &F<T> {
    type Output = F<T>;

    fn mul(self, rhs: F<T>) -> Self::Output {
        self * &rhs
    }
}

impl<T: RaisedBubbleFloat> Add<&F<T>> for &F<T> {
    type Output = F<T>;
    fn add(self, rhs: &F<T>) -> Self::Output {
        F(self.0.ref_add(&rhs.0))
    }
}

impl<T: RaisedBubbleFloat> Add<&F<T>> for F<T> {
    type Output = F<T>;
    fn add(self, rhs: &F<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T: RaisedBubbleFloat> Add<F<T>> for F<T> {
    type Output = F<T>;
    fn add(self, rhs: F<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T: RaisedBubbleFloat> Add<F<T>> for &F<T> {
    type Output = F<T>;
    fn add(self, rhs: F<T>) -> Self::Output {
        self + &rhs
    }
}

impl<T: RaisedBubbleFloat> Div<&F<T>> for &F<T> {
    type Output = F<T>;
    fn div(self, rhs: &F<T>) -> Self::Output {
        F(self.0.ref_div(&rhs.0))
    }
}

impl<T: RaisedBubbleFloat> Div<&F<T>> for F<T> {
    type Output = F<T>;
    fn div(self, rhs: &F<T>) -> Self::Output {
        &self / rhs
    }
}

impl<T: RaisedBubbleFloat> Div<F<T>> for F<T> {
    type Output = F<T>;
    fn div(self, rhs: F<T>) -> Self::Output {
        &self / &rhs
    }
}

impl<T: RaisedBubbleFloat> Div<F<T>> for &F<T> {
    type Output = F<T>;
    fn div(self, rhs: F<T>) -> Self::Output {
        self / &rhs
    }
}

impl<T: RaisedBubbleFloat> Sub<&F<T>> for &F<T> {
    type Output = F<T>;
    fn sub(self, rhs: &F<T>) -> Self::Output {
        F(self.0.ref_sub(&rhs.0))
    }
}

impl<T: RaisedBubbleFloat> Sub<&F<T>> for F<T> {
    type Output = F<T>;
    fn sub(self, rhs: &F<T>) -> Self::Output {
        &self - rhs
    }
}

impl<T: RaisedBubbleFloat> Sub<F<T>> for F<T> {
    type Output = F<T>;
    fn sub(self, rhs: F<T>) -> Self::Output {
        &self - &rhs
    }
}

impl<T: RaisedBubbleFloat> Sub<F<T>> for &F<T> {
    type Output = F<T>;
    fn sub(self, rhs: F<T>) -> Self::Output {
        self - &rhs
    }
}

impl<T: RaisedBubbleFloat> Neg for &F<T> {
    type Output = F<T>;
    fn neg(self) -> Self::Output {
        F(self.0.ref_neg())
    }
}

impl<T: RaisedBubbleFloat> Neg for F<T> {
    type Output = F<T>;
    fn neg(self) -> Self::Output {
        F(self.0.ref_neg())
    }
}

impl<T: RaisedBubbleFloat> F<T> {
    fn inv(&self) -> Self {
        self.one() / self
    }

    fn one(&self) -> Self {
        F(self.0.one())
    }

    fn zero(&self) -> Self {
        F(self.0.zero())
    }

    fn sqrt(&self) -> Self {
        F(self.0.sqrt())
    }

    fn exp(&self) -> Self {
        F(self.0.exp())
    }

    fn signum(&self) -> Self {
        if self > &self.zero() {
            self.one()
        } else if self < &self.zero() {
            -self.one()
        } else {
            self.zero()
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_f64(&self, x: f64) -> Self {
        F(self.0.from_f64(x))
    }
}

impl<T: RaisedBubbleFloat> From<F<T>> for f64 {
    fn from(value: F<T>) -> Self {
        value.0.into()
    }
}

impl<T: RaisedBubbleFloat> Display for F<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
struct ArbPrec<const N: u32> {
    val: Float,
}

impl<const N: u32> ArbPrec<N> {
    fn new(val: f64) -> Self {
        Self {
            val: Float::with_val(N, val),
        }
    }
}

impl<const N: u32> RaisedBubbleFloat for ArbPrec<N> {
    fn sqrt(&self) -> Self {
        Self {
            val: self.val.sqrt_ref().complete(N.into()),
        }
    }

    fn exp(&self) -> Self {
        Self {
            val: self.val.clone().exp(),
        }
    }

    fn one(&self) -> Self {
        Self {
            val: Float::with_val(self.val.prec(), 1.0),
        }
    }

    fn zero(&self) -> Self {
        Self {
            val: Float::new(self.val.prec()),
        }
    }

    fn from_f64(&self, x: f64) -> Self {
        Self {
            val: Float::with_val(N, x),
        }
    }
}

impl<const N: u32> From<ArbPrec<N>> for f64 {
    fn from(value: ArbPrec<N>) -> Self {
        value.val.to_f64()
    }
}

impl<const N: u32> Sub<&ArbPrec<N>> for &ArbPrec<N> {
    type Output = ArbPrec<N>;
    fn sub(self, rhs: &ArbPrec<N>) -> Self::Output {
        Self::Output {
            val: (&self.val - &rhs.val).complete(N.into()),
        }
    }
}

impl<const N: u32> Add<&ArbPrec<N>> for &ArbPrec<N> {
    type Output = ArbPrec<N>;
    fn add(self, rhs: &ArbPrec<N>) -> Self::Output {
        Self::Output {
            val: (&self.val + &rhs.val).complete(N.into()),
        }
    }
}

impl<const N: u32> Mul<&ArbPrec<N>> for &ArbPrec<N> {
    type Output = ArbPrec<N>;
    fn mul(self, rhs: &ArbPrec<N>) -> Self::Output {
        Self::Output {
            val: (&self.val * &rhs.val).complete(N.into()),
        }
    }
}

impl<const N: u32> Div<&ArbPrec<N>> for &ArbPrec<N> {
    type Output = ArbPrec<N>;
    fn div(self, rhs: &ArbPrec<N>) -> Self::Output {
        Self::Output {
            val: (&self.val / &rhs.val).complete(N.into()),
        }
    }
}

impl<const N: u32> Display for ArbPrec<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.val.fmt(f)
    }
}

impl<const N: u32> Neg for &ArbPrec<N> {
    type Output = ArbPrec<N>;

    fn neg(self) -> Self::Output {
        Self::Output {
            val: -self.val.clone(),
        }
    }
}
