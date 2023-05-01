use crate::integrands::*;
use crate::utils::FloatLike;
use havana::{ContinuousGrid, Grid, Sample};
use lorentz_vector::LorentzVector;
use num::Complex;
use serde::Deserialize;

const ROOT_FINDING_EPSILON: f64 = 1.0e-6;
const ROOT_FINDING_MAX_ITERATION: u32 = 20;

#[derive(Clone, Copy)]
struct Energy<T: FloatLike> {
    p: LorentzVector<T>,
    m: T,
    shift: LorentzVector<T>,
    label: usize,
}

impl<T: FloatLike> Energy<T> {
    fn new(p: LorentzVector<T>, m: T, shift: LorentzVector<T>, label: usize) -> Self {
        Energy { p, m, shift, label }
    }

    #[inline]
    fn evaluate(&self, k: &LorentzVector<T>) -> T {
        ((k + self.p + self.shift).spatial_squared() + self.m * self.m).sqrt()
    }
}

#[derive(Clone)]
struct ESurface<T: FloatLike> {
    energies: [Energy<T>; 2],
    pij0: T,
    label: usize,
}

#[allow(unused)]
impl<T: FloatLike> ESurface<T> {
    fn new(energies: [Energy<T>; 2], pij0: T, label: usize) -> Self {
        ESurface {
            energies,
            pij0,
            label,
        }
    }

    #[inline]
    fn evaluate(&self, k: &LorentzVector<T>) -> T {
        self.energies[0].evaluate(k) + self.energies[1].evaluate(k) - self.pij0
    }

    #[inline]
    fn fast_evaluate(&self, energy_vals: &[T]) -> T {
        energy_vals[self.energies[0].label] + energy_vals[self.energies[1].label] - self.pij0
    }

    #[inline]
    fn evaluate_gradient(&self, k: &LorentzVector<T>) -> LorentzVector<T> {
        (k + self.energies[0].p + self.energies[0].shift) / self.energies[0].evaluate(k)
            + (k + self.energies[1].p + self.energies[1].shift) / self.energies[1].evaluate(k)
    }

    #[inline]
    fn fast_evaluate_gradient(&self, k: &LorentzVector<T>, energy_vals: &[T]) -> LorentzVector<T> {
        (k + self.energies[0].p + self.energies[0].shift) / energy_vals[self.energies[0].label]
            + (k + self.energies[1].p + self.energies[1].shift)
                / energy_vals[self.energies[1].label]
    }

    #[inline]
    fn get_rstars(&self, phi: &T, theta: &T) -> (T, T) {
        let khat = k_hat(phi, theta);
        let f = |r: T| self.evaluate(&(khat * r));
        let fprime = |r: T| self.evaluate_gradient(&(khat * r)).spatial_dot(&khat);

        let mut rplus = self.pij0.abs();
        let mut rminus = -self.pij0.abs();

        let epsilon = T::from_f64(ROOT_FINDING_EPSILON).unwrap();

        for _ in 0..ROOT_FINDING_MAX_ITERATION {
            rplus = rplus - f(rplus) / fprime(rplus);
            if rplus.abs() < epsilon {
                break;
            }
        }

        for _ in 0..ROOT_FINDING_MAX_ITERATION {
            rminus = rminus - f(rminus) / fprime(rminus);
            if rminus.abs() < epsilon {
                break;
            }
        }

        (rplus, rminus)
    }
}

#[derive(Clone)]
struct IntegrandTerm<T: FloatLike> {
    term: Vec<ESurface<T>>,
}

impl<T: FloatLike> IntegrandTerm<T> {
    fn new(term: Vec<ESurface<T>>) -> Self {
        Self { term }
    }

    #[inline]
    fn evaluate(&self, k: &LorentzVector<T>) -> T {
        let mut res = T::one();
        for surface in self.term.iter() {
            res = res / surface.evaluate(k);
        }
        res
    }

    #[inline]
    fn fast_evaluate(&self, surface_vals: &[T]) -> T {
        let mut inv_res = T::one();
        for surface in self.term.iter() {
            inv_res *= surface_vals[surface.label];
        }
        inv_res.recip()
    }

    fn contains_surface(&self, label: usize) -> Option<usize> {
        for i in 0..self.term.len() {
            if self.term[i].label == label {
                return Some(i);
            }
        }

        None
    }
}

#[derive(Clone)]
struct Integrand<T: FloatLike> {
    terms: Vec<IntegrandTerm<T>>,
    prefactor: Vec<Energy<T>>,
    e_cm: T,
    surfaces: Vec<ESurface<T>>,
}

#[allow(unused)]
impl<T: FloatLike> Integrand<T> {
    fn new(
        terms: Vec<IntegrandTerm<T>>,
        prefactor: Vec<Energy<T>>,
        e_cm: T,
        surfaces: Vec<ESurface<T>>,
    ) -> Self {
        Self {
            terms,
            prefactor,
            e_cm,
            surfaces,
        }
    }

    fn new_triangle(
        p1: LorentzVector<T>,
        p2: LorentzVector<T>,
        masses: [T; 3],
        s: LorentzVector<T>,
    ) -> Self {
        let e_cm = (p1 + p2).square().sqrt();
        let energies = vec![
            Energy::new(LorentzVector::new(), masses[0], s, 0),
            Energy::new(p1, masses[1], s, 1),
            Energy::new(p1 + p2, masses[2], s, 2),
        ];

        let surfaces = vec![
            ESurface::new([energies[0].clone(), energies[1].clone()], p1.t, 0),
            ESurface::new([energies[1].clone(), energies[2].clone()], p2.t, 1),
            ESurface::new([energies[0].clone(), energies[2].clone()], p1.t + p2.t, 2),
            ESurface::new([energies[0].clone(), energies[1].clone()], -p1.t, 3),
            ESurface::new([energies[1].clone(), energies[2].clone()], -p2.t, 4),
            ESurface::new([energies[0].clone(), energies[2].clone()], -p1.t - p2.t, 5),
        ];

        let terms = vec![
            IntegrandTerm::new(vec![surfaces[3].clone(), surfaces[1].clone()]),
            IntegrandTerm::new(vec![surfaces[0].clone(), surfaces[2].clone()]),
            IntegrandTerm::new(vec![surfaces[1].clone(), surfaces[2].clone()]),
            IntegrandTerm::new(vec![surfaces[0].clone(), surfaces[4].clone()]),
            IntegrandTerm::new(vec![surfaces[3].clone(), surfaces[5].clone()]),
            IntegrandTerm::new(vec![surfaces[4].clone(), surfaces[5].clone()]),
        ];

        Self {
            terms,
            prefactor: energies,
            e_cm,
            surfaces,
        }
    }

    fn generate_counter_term(&self, counter_surface: ESurface<T>) -> CounterTerm<T> {
        let mut counter_term_terms = vec![];

        for term in self.terms.iter() {
            if let Some(i) = term.contains_surface(counter_surface.label) {
                let mut new_term = term.clone();
                new_term.term.remove(i);
                counter_term_terms.push(new_term);
            }
        }

        CounterTerm {
            surface: counter_surface,
            prefactor: self.prefactor.clone(),
            terms: counter_term_terms,
            sigma: self.e_cm,
        }
    }

    #[inline]
    fn evaluate(&self, k: &LorentzVector<T>) -> T {
        let mut prefactor = T::one();
        for energy in self.prefactor.iter() {
            prefactor = prefactor / (T::from_i64(2).unwrap() * energy.evaluate(k));
        }

        let mut terms = T::zero();
        for term in self.terms.iter() {
            terms += term.evaluate(k);
        }

        prefactor * terms
    }

    #[inline]
    fn fast_evaluate(&self, k: &LorentzVector<T>) -> T {
        let mut energy_vals = vec![T::zero(); self.prefactor.len()];
        let mut prefactor = T::one();

        for i in 0..energy_vals.len() {
            energy_vals[i] = self.prefactor[i].evaluate(&k);
            prefactor = prefactor / (T::from_i64(2).unwrap() * energy_vals[i]);
        }

        let mut surface_vals = vec![T::zero(); self.surfaces.len()];

        for i in 0..surface_vals.len() {
            surface_vals[i] = self.surfaces[i].fast_evaluate(&energy_vals);
        }

        let mut terms = T::zero();
        for term in self.terms.iter() {
            terms += term.fast_evaluate(&surface_vals);
        }

        prefactor * terms
    }

    #[inline]
    fn evaluate_spherical(&self, x_spherical: &[T]) -> T {
        let (k, jac) = spherical_to_momentum(x_spherical);
        self.evaluate(&k) * jac
    }

    #[inline]
    fn fast_evaluate_spherical(&self, x_spherical: &[T]) -> T {
        let (k, jac) = spherical_to_momentum(x_spherical);
        self.fast_evaluate(&k) * jac
    }

    #[inline]
    fn evaluate_hypercube(&self, z: &[T]) -> T {
        let (x_spherical, jac) = hypercube_to_hemispherical(z, self.e_cm);
        self.evaluate_spherical(&x_spherical) * jac
    }

    #[inline]
    fn fast_evaluate_hypercube(&self, z: &[T]) -> T {
        let (x_spherical, jac) = hypercube_to_hemispherical(z, self.e_cm);
        self.fast_evaluate_spherical(&x_spherical) * jac
    }
}

struct CounterTerm<T: FloatLike> {
    surface: ESurface<T>,
    prefactor: Vec<Energy<T>>,
    terms: Vec<IntegrandTerm<T>>,
    sigma: T,
}

impl<T: FloatLike> CounterTerm<T> {
    #[inline]
    fn evaluate_hemispherical(&self, x_spherical: &[T]) -> T {
        let (r_star_plus, r_star_minus) = self.surface.get_rstars(&x_spherical[1], &x_spherical[2]); // khat is computed twice
        let khat = k_hat(&x_spherical[1], &x_spherical[2]);

        //forward gradient from rootfinding
        let (grad_phi_plus, grad_phi_minus) = (
            self.surface
                .evaluate_gradient(&(khat * r_star_plus))
                .spatial_dot(&khat),
            self.surface
                .evaluate_gradient(&(khat * r_star_minus))
                .spatial_dot(&khat),
        );

        let (mut prefactor_plus, mut prefactor_minus) = (T::one(), T::one());

        for energy in self.prefactor.iter() {
            prefactor_plus =
                prefactor_plus / (T::from_i64(2).unwrap() * energy.evaluate(&(khat * r_star_plus)));
            prefactor_minus = prefactor_minus
                / (T::from_i64(2).unwrap() * energy.evaluate(&(khat * r_star_minus)));
        }

        let (mut terms_plus, mut terms_minus) = (T::zero(), T::zero());

        for term in self.terms.iter() {
            terms_plus += term.evaluate(&(khat * r_star_plus));
            terms_minus += term.evaluate(&(khat * r_star_minus));
        }

        let a_ij = T::one()
            - (self.surface.pij0.recip()
                * (self.surface.energies[0].p - self.surface.energies[1].p).spatial_dot(&khat))
            .powi(2);

        let (gaussian_plus, gaussian_minus) = (
            (-self.sigma.powi(-2) * a_ij.powi(-2) * (x_spherical[0] - r_star_plus).powi(2)).exp(),
            (-self.sigma.powi(-2) * a_ij.powi(-2) * (x_spherical[0] - r_star_minus).powi(2)).exp(),
        );

        let counter_term_plus = x_spherical[2].sin()
            * r_star_plus.powi(2)
            * (x_spherical[0] - r_star_plus).recip()
            * grad_phi_plus.recip()
            * prefactor_plus
            * gaussian_plus
            * terms_plus;

        let counter_term_minus = x_spherical[2].sin()
            * r_star_minus.powi(2)
            * (x_spherical[0] - r_star_minus).recip()
            * grad_phi_minus.recip()
            * prefactor_minus
            * gaussian_minus
            * terms_minus;

        counter_term_plus + counter_term_minus
    }
}

struct RegulatedIntegrand<T: FloatLike> {
    integrand: Integrand<T>,
    counter_terms: Vec<CounterTerm<T>>,
}

#[allow(unused)]
impl<T: FloatLike> RegulatedIntegrand<T> {
    fn new(integrand: Integrand<T>, counter_terms: Vec<CounterTerm<T>>) -> Self {
        Self {
            integrand,
            counter_terms,
        }
    }

    #[inline]
    fn evaluate_hemispherical(&self, x_spherical: &[T]) -> T {
        let mut res = self.integrand.fast_evaluate_spherical(&x_spherical);

        for counter_term in self.counter_terms.iter() {
            res -= counter_term.evaluate_hemispherical(&x_spherical);
        }

        res
    }

    #[inline]
    fn evaluate_hypercube(&self, z: &[T], b: T) -> T {
        let (x_spherical, jac) = hypercube_to_hemispherical(z, b);
        -self.evaluate_hemispherical(&x_spherical)
            * jac
            * T::from_f64(2.0 / std::f64::consts::PI).unwrap()
    }
}

#[inline]
fn _hypercube_to_spherical<T: FloatLike>(z: &[T], e_cm: T) -> ([T; 3], T) {
    let r = e_cm * z[0] / (T::one() - z[0]);
    let phi = T::from_f64(2.0 * std::f64::consts::PI).unwrap() * z[1];
    let theta = T::from_f64(std::f64::consts::PI).unwrap() * z[2];

    let conformal_jac = T::from_f64(2.0 * std::f64::consts::PI * std::f64::consts::PI).unwrap()
        * e_cm
        * ((T::one() - z[0]).recip() + z[0] * (T::one() - z[0]).powi(-2));

    return ([r, phi, theta], conformal_jac);
}

#[inline]
fn hypercube_to_hemispherical<T: FloatLike>(z: &[T], b: T) -> ([T; 3], T) {
    let q = T::from_f64(2.0).unwrap() * z[0] - T::one();
    let r = b * q / (T::one() - q * q);
    let phi = T::from_f64(std::f64::consts::PI).unwrap() * z[1];
    let theta = T::from_f64(std::f64::consts::PI).unwrap() * z[2];

    let conformal_jac = T::from_f64(std::f64::consts::PI * std::f64::consts::PI).unwrap()
        * b
        * T::from_f64(2.0).unwrap()
        * ((T::one() - q * q).recip()
            + T::from_f64(2.0).unwrap() * q * q * (T::one() - q * q).powi(-2));

    ([r, phi, theta], conformal_jac.abs())
}

#[inline]
fn spherical_to_momentum<T: FloatLike>(k_s: &[T]) -> (LorentzVector<T>, T) {
    let momentum = LorentzVector::from_args(
        T::zero(),
        k_s[0] * k_s[1].cos() * k_s[2].sin(),
        k_s[0] * k_s[1].sin() * k_s[2].sin(),
        k_s[0] * k_s[2].cos(),
    );

    let jac = k_s[0] * k_s[0] * k_s[2].sin();
    return (momentum, jac);
}

#[inline]
fn k_hat<T: FloatLike>(phi: &T, theta: &T) -> LorentzVector<T> {
    LorentzVector::from_args(
        T::zero(),
        phi.cos() * theta.sin(),
        phi.sin() * theta.sin(),
        theta.cos(),
    )
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriangleSubtractionSettings {
    pub p1: [f64; 4],
    pub p2: [f64; 4],
    pub center: [f64; 4],
    pub masses: [f64; 3],
}

pub struct TriangleSubtractionIntegrand {
    _integrand_settings: TriangleSubtractionSettings,
    integrand_f64: RegulatedIntegrand<f64>,
    // integrand_f128: RegulatedIntegrand<f128>, todo add f128 support
    sqrt_s: f64,
}

#[allow(unused)]
impl TriangleSubtractionIntegrand {
    pub fn new(integrand_settings: TriangleSubtractionSettings) -> Self {
        let p1 = LorentzVector::from_args(
            (integrand_settings.p1[0]),
            (integrand_settings.p1[1]),
            (integrand_settings.p1[2]),
            (integrand_settings.p1[3]),
        );

        let p2 = LorentzVector::from_args(
            (integrand_settings.p2[0]),
            (integrand_settings.p2[1]),
            (integrand_settings.p2[2]),
            (integrand_settings.p2[3]),
        );

        let sqrt_s = (p1 + p2).square().sqrt();
        let shift = LorentzVector::from_args(
            (integrand_settings.center[0]),
            (integrand_settings.center[1]),
            (integrand_settings.center[2]),
            (integrand_settings.center[3]),
        );

        let masses = [
            (integrand_settings.masses[0]),
            (integrand_settings.masses[1]),
            (integrand_settings.masses[2]),
        ];

        let integrand = Integrand::new_triangle(p1, p2, masses, shift);

        let energies = vec![
            Energy::new(LorentzVector::new(), masses[0], shift, 0),
            Energy::new(p1, masses[1], shift, 1),
            Energy::new(p1 + p2, masses[2], shift, 2),
        ];

        let surfaces = [
            ESurface::new([energies[0].clone(), energies[1].clone()], p1.t, 0),
            ESurface::new([energies[1].clone(), energies[2].clone()], p2.t, 1),
            ESurface::new([energies[0].clone(), energies[2].clone()], p1.t + p2.t, 2),
        ];
        let mut counter_terms = vec![];

        for surface in surfaces.iter() {
            counter_terms.push(integrand.generate_counter_term(surface.clone()));
        }

        let regulated_integrand = RegulatedIntegrand::new(integrand, counter_terms);

        Self {
            _integrand_settings: integrand_settings.clone(),
            integrand_f64: regulated_integrand,
            sqrt_s,
        }
    }
}

#[allow(unused)]
impl HasIntegrand for TriangleSubtractionIntegrand {
    fn get_n_dim(&self) -> usize {
        3
    }

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        if let Sample::ContinuousGrid(cont_weight, cs) = sample {
            let res = self.integrand_f64.evaluate_hypercube(&cs, self.sqrt_s);
            return Complex::new(res, 0.0);
        } else {
            unreachable!()
        }
    }

    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(3, 10, 1000))
    }
}
