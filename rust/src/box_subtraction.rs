use crate::integrands::*;
use crate::utils::FloatLike;
use lorentz_vector::LorentzVector;
use num::Complex;
use serde::Deserialize;
use std::fs;
use symbolica::numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, Sample};
use yaml_rust::{Yaml, YamlLoader};

const ROOT_FINDING_EPSILON: f64 = 1.0e-6;
const ROOT_FINDING_MAX_ITERATION: u32 = 20;

const DELTA: f64 = 1.0;
const RESCALE: f64 = 1.0;

#[derive(Copy, Clone)]
struct Energy<T: FloatLike> {
    p: LorentzVector<T>,
    m: T,
    label: usize,
}

impl<T: FloatLike> Energy<T> {
    fn new(p: LorentzVector<T>, m: T, label: usize) -> Self {
        Energy { p, m, label }
    }

    #[inline]
    fn evaluate(&self, k: &LorentzVector<T>, shift: &LorentzVector<T>) -> T {
        ((k + shift + self.p).spatial_squared() + self.m * self.m).sqrt()
    }
}

#[derive(Copy, Clone)]
struct ESurface<T: FloatLike> {
    energies: [Energy<T>; 2],
    pij0: T,
    label: usize,
}

impl<T: FloatLike> ESurface<T> {
    fn new(energies: [Energy<T>; 2], label: usize) -> Self {
        let pij0 = energies[0].p.t - energies[1].p.t;
        ESurface {
            energies,
            pij0,
            label,
        }
    }

    fn exists(&self) -> bool {
        (self.energies[0].p - self.energies[1].p).square()
            >= (self.energies[0].m + self.energies[1].m).powi(2)
            && self.pij0 >= T::zero()
    }

    fn simple_eq(&self, other: Self) -> bool {
        self.energies[0].label == other.energies[0].label
            && self.energies[1].label == other.energies[1].label
            && self.pij0 == other.pij0
    }

    fn contains_point(&self, k: &LorentzVector<T>, shift: &LorentzVector<T>) -> bool {
        self.energies[0].evaluate(&k, &shift) + self.energies[1].evaluate(&k, &shift) - self.pij0
            < T::zero()
    }

    #[inline]
    fn fast_evaluate(&self, energy_vals: &[T]) -> T {
        energy_vals[self.energies[0].label] + energy_vals[self.energies[1].label] - self.pij0
    }

    #[inline]
    fn get_rstar_and_gradient(
        &self,
        khat: &LorentzVector<T>,
        shift: &LorentzVector<T>,
    ) -> (T, T, T, T) {
        let mut rplus = self.pij0.abs();
        let p0_dot_khat = (self.energies[0].p + shift).spatial_dot(&khat);
        let p1_dot_khat = (self.energies[1].p + shift).spatial_dot(&khat);

        let epsilon = T::from_f64(ROOT_FINDING_EPSILON).unwrap();
        let mut fprimeplus = T::one();

        for _ in 0..ROOT_FINDING_MAX_ITERATION {
            let kstar = khat * rplus;
            let energy1 = self.energies[0].evaluate(&kstar, shift);
            let energy2 = self.energies[1].evaluate(&kstar, shift);

            let f = energy1 + energy2 - self.pij0;
            fprimeplus = (rplus + p0_dot_khat) / energy1 + (rplus + p1_dot_khat) / energy2;

            rplus = rplus - f / fprimeplus;
            if f.abs() < epsilon {
                break;
            }
        }

        let mut rminus = rplus - self.pij0;
        let mut fprimeminus = T::one();

        for _ in 0..ROOT_FINDING_MAX_ITERATION {
            let kstar = khat * rminus;
            let energy1 = self.energies[0].evaluate(&kstar, shift);
            let energy2 = self.energies[1].evaluate(&kstar, shift);

            let f = energy1 + energy2 - self.pij0;
            fprimeminus = (rminus + p0_dot_khat) / energy1 + (rminus + p1_dot_khat) / energy2;

            rminus = rminus - f / fprimeminus;
            if f.abs() < epsilon {
                break;
            }
        }

        (rplus, fprimeplus, rminus, fprimeminus)
    }
}

#[derive(Clone)]
struct ESurfaceProduct<T: FloatLike> {
    factors: Vec<ESurface<T>>,
}

impl<T: FloatLike> ESurfaceProduct<T> {
    #[inline]
    fn fast_evaluate(&self, surface_vals: &[T]) -> T {
        let mut inv_res = T::one();
        for factor in self.factors.iter() {
            inv_res *= surface_vals[factor.label];
        }

        inv_res.recip()
    }

    fn contains(&self, surface: &ESurface<T>) -> Option<usize> {
        for i in 0..self.factors.len() {
            if surface.label == self.factors[i].label {
                return Some(i);
            }
        }
        None
    }
}

struct CrossFreeFamilyIntegrand<T: FloatLike> {
    terms: Vec<ESurfaceProduct<T>>,
    esurfaces: Vec<ESurface<T>>,
    energies: Vec<Energy<T>>,
    e_cm: T,
}

impl<T: FloatLike> CrossFreeFamilyIntegrand<T> {
    fn _fast_evaluate(&self, k: &LorentzVector<T>) -> T {
        let mut energy_vals = vec![];
        let mut inv_prefactor = T::one();

        for energy in self.energies.iter() {
            let energy_val = energy.evaluate(k, &LorentzVector::new());
            inv_prefactor *= T::from_f64(2.0).unwrap() * energy_val;
            energy_vals.push(energy_val);
        }

        let mut surface_vals = vec![];

        for surface in self.esurfaces.iter() {
            surface_vals.push(surface.fast_evaluate(&energy_vals))
        }

        let mut res = T::zero();
        for term in self.terms.iter() {
            res += term.fast_evaluate(&surface_vals);
        }

        res * inv_prefactor.recip()
    }

    fn construct_channel(
        &self,
        groups: Vec<Vec<ESurface<T>>>,
        removed_group: Vec<ESurface<T>>,
        selected_group: Vec<ESurface<T>>,
        center: LorentzVector<T>,
    ) -> CrossFreeFamilyChannel<T> {
        let mut channel_terms = vec![];
        let mut remove_numerator = vec![];

        for surface in removed_group.iter() {
            remove_numerator.push((*surface, 2));
        }

        for term in self.terms.iter() {
            let mut new_term = ESurfaceRatio {
                numerator: remove_numerator.clone(),
                denominator: term.clone(),
            };

            new_term.simplify();
            channel_terms.push(new_term);
        }

        for surface in selected_group.iter() {
            if !surface.contains_point(&center, &LorentzVector::new()) {
                println!("warning! center not in group")
            }
        }

        CrossFreeFamilyChannel {
            terms: channel_terms,
            groups,
            energies: self.energies.clone(),
            selected_group,
            center,
            all_surfaces: self.esurfaces.clone(),
            e_cm: self.e_cm,
        }
    }
}

#[derive(Clone)]
struct ESurfaceRatio<T: FloatLike> {
    numerator: Vec<(ESurface<T>, i32)>,
    denominator: ESurfaceProduct<T>,
}

impl<T: FloatLike> ESurfaceRatio<T> {
    fn fast_evaluate(&self, surface_vals: &[T]) -> T {
        let denom = self.denominator.fast_evaluate(surface_vals);
        let mut num = T::one();

        for surface in self.numerator.iter() {
            num *= surface_vals[surface.0.label].powi(surface.1);
        }

        return num * denom;
    }

    fn simplify(&mut self) {
        for numerator in self.numerator.iter_mut() {
            if let Some(i) = self.denominator.contains(&numerator.0) {
                self.denominator.factors.remove(i);
                (*numerator).1 -= 1;
            }
        }
    }
}

#[derive(Clone)]
struct CrossFreeFamilyChannel<T: FloatLike> {
    terms: Vec<ESurfaceRatio<T>>,
    all_surfaces: Vec<ESurface<T>>,
    groups: Vec<Vec<ESurface<T>>>,
    selected_group: Vec<ESurface<T>>,
    energies: Vec<Energy<T>>,
    center: LorentzVector<T>,
    e_cm: T,
}

impl<T: FloatLike> CrossFreeFamilyChannel<T> {
    fn fast_evaluate(&self, k: &LorentzVector<T>) -> T {
        let mut energy_vals = vec![];
        let mut inv_energy_prefactor = T::one();

        for energy in self.energies.iter() {
            let new_energy = energy.evaluate(&k, &self.center);
            inv_energy_prefactor *= T::from_f64(2.0).unwrap() * new_energy;
            energy_vals.push(new_energy);
        }

        let mut surface_vals = vec![];

        for surface in self.all_surfaces.iter() {
            surface_vals.push(surface.fast_evaluate(&energy_vals));
        }

        let mut inv_multichanneling_prefactor = T::zero();

        for group in self.groups.iter() {
            let mut group_prod = T::one();
            for surface in group.iter() {
                group_prod *= surface_vals[surface.label].powi(2);
            }

            inv_multichanneling_prefactor += group_prod;
        }

        let prefactor = (inv_energy_prefactor * inv_multichanneling_prefactor).recip();

        let mut term_sum = T::zero();

        for term in self.terms.iter() {
            term_sum += term.fast_evaluate(&surface_vals);
        }

        prefactor * term_sum
    }

    fn counter_term(&self, counter_surface: ESurface<T>) -> CrossFreeFamilyCounterTerm<T> {
        let mut new_terms = vec![];

        for old_term in self.terms.iter() {
            if let Some(j) = old_term.denominator.contains(&counter_surface) {
                let mut new_term = old_term.clone();
                new_term.denominator.factors.remove(j);
                new_terms.push(new_term);
            }
        }

        let delta = T::from_f64(DELTA).unwrap();

        CrossFreeFamilyCounterTerm {
            terms: new_terms,
            counter_surface,
            energies: self.energies.clone(),
            center: self.center,
            delta,
            e_cm: self.e_cm,
            all_surfaces: self.all_surfaces.clone(),
            groups: self.groups.clone(),
        }
    }

    fn regulate(&self) -> RegulatedChannel<T> {
        let mut counter_terms = vec![];
        for surface in self.selected_group.iter() {
            counter_terms.push(self.counter_term(surface.clone()));
        }

        RegulatedChannel {
            unregulated_channel: self.clone(),
            counter_terms,
            _center: self.center,
            e_cm: self.e_cm,
        }
    }
}

struct CrossFreeFamilyCounterTerm<T: FloatLike> {
    terms: Vec<ESurfaceRatio<T>>,
    all_surfaces: Vec<ESurface<T>>,
    counter_surface: ESurface<T>,
    energies: Vec<Energy<T>>,
    center: LorentzVector<T>,
    delta: T,
    groups: Vec<Vec<ESurface<T>>>,
    e_cm: T,
}

impl<T: FloatLike> CrossFreeFamilyCounterTerm<T> {
    #[inline]
    fn evaluate(&self, r: T, khat: &LorentzVector<T>) -> T {
        let (r_star_plus, grad_plus, r_star_minus, grad_minus) = self
            .counter_surface
            .get_rstar_and_gradient(&khat, &self.center);

        let mut ct = T::zero();

        if (r - r_star_plus).abs() < self.delta * self.e_cm {
            let kstar = khat * r_star_plus;

            let mut energy_vals = vec![];
            let mut inv_energy_perfactor = T::one();

            for energy in self.energies.iter() {
                let new_energy = energy.evaluate(&kstar, &self.center);
                inv_energy_perfactor *= T::from_f64(2.0).unwrap() * new_energy;
                energy_vals.push(new_energy);
            }

            let mut surface_vals = vec![];

            for surface in self.all_surfaces.iter() {
                surface_vals.push(surface.fast_evaluate(&energy_vals));
            }

            let mut inv_multichannel_prefactor = T::zero();
            for group in self.groups.iter() {
                let mut group_prod = T::one();
                for surface in group.iter() {
                    group_prod *= surface_vals[surface.label].powi(2);
                }
                inv_multichannel_prefactor += group_prod;
            }

            let mut term_sum = T::zero();
            for term in self.terms.iter() {
                term_sum += term.fast_evaluate(&surface_vals);
            }

            let sin_theta = (khat.x * khat.x + khat.y * khat.y).sqrt();
            let jac = r_star_plus * r_star_plus * sin_theta;
            let exp = (-(self.e_cm * T::from_f64(RESCALE).unwrap()).powi(-2)
                * (r - r_star_plus).powi(2))
            .exp();

            ct +=
                ((r - r_star_plus) * inv_energy_perfactor * inv_multichannel_prefactor * grad_plus)
                    .inv()
                    * jac
                    * exp
                    * term_sum;
        }

        if (r - r_star_minus).abs() < self.delta * self.e_cm {
            let kstar = khat * r_star_minus;

            let mut energy_vals = vec![];
            let mut inv_energy_perfactor = T::one();

            for energy in self.energies.iter() {
                let new_energy = energy.evaluate(&kstar, &self.center);
                inv_energy_perfactor *= T::from_f64(2.0).unwrap() * new_energy;
                energy_vals.push(new_energy);
            }

            let mut surface_vals = vec![];

            for surface in self.all_surfaces.iter() {
                surface_vals.push(surface.fast_evaluate(&energy_vals));
            }

            let mut inv_multichannel_prefactor = T::zero();
            for group in self.groups.iter() {
                let mut group_prod = T::one();
                for surface in group.iter() {
                    group_prod *= surface_vals[surface.label].powi(2);
                }
                inv_multichannel_prefactor += group_prod;
            }

            let mut term_sum = T::zero();
            for term in self.terms.iter() {
                term_sum += term.fast_evaluate(&surface_vals);
            }

            let sin_theta = (khat.x * khat.x + khat.y * khat.y).sqrt();
            let jac = r_star_minus * r_star_minus * sin_theta;
            let exp = (-(self.e_cm * T::from_f64(RESCALE).unwrap()).powi(-2)
                * (r - r_star_minus).powi(2))
            .exp();

            ct += ((r - r_star_minus)
                * inv_energy_perfactor
                * inv_multichannel_prefactor
                * grad_minus)
                .recip()
                * jac
                * exp
                * term_sum;
        }
        ct
    }
}

struct RegulatedChannel<T: FloatLike> {
    unregulated_channel: CrossFreeFamilyChannel<T>,
    counter_terms: Vec<CrossFreeFamilyCounterTerm<T>>,
    _center: LorentzVector<T>,
    e_cm: T,
}

impl<T: FloatLike> RegulatedChannel<T> {
    #[inline]
    fn evaluate_hypercube(&self, z: &[T]) -> T {
        let (x_spherical, conformal_jac) = hypercube_to_hemispherical(z, self.e_cm);
        let mut counter_term_sum = T::zero();

        for counter_term in self.counter_terms.iter() {
            counter_term_sum += counter_term
                .evaluate(x_spherical[0], &k_hat(&x_spherical[1], &x_spherical[2]))
                * conformal_jac;
        }

        let (k, spherical_jac) = spherical_to_cartesian(&x_spherical);
        let unregulated_integrand =
            self.unregulated_channel.fast_evaluate(&k) * spherical_jac * conformal_jac;

        unregulated_integrand - counter_term_sum
    }
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

#[inline]
fn _hypercube_to_spherical<T: FloatLike>(z: &[T]) -> ([T; 3], T) {
    let r = T::from_f64(100.0).unwrap() * z[0] / (T::one() - z[0]);
    let phi = T::from_f64(2.0 * std::f64::consts::PI).unwrap() * z[1];
    let theta = T::from_f64(std::f64::consts::PI).unwrap() * z[2];

    let jac = T::from_f64(100.0).unwrap()
        * (T::one() / (T::one() - z[0]) + z[0] / (T::one() - z[0]).powi(2))
        * T::from_f64(2.0 * std::f64::consts::PI.powi(2)).unwrap();
    ([r, phi, theta], jac)
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
fn spherical_to_cartesian<T: FloatLike>(x_s: &[T]) -> (LorentzVector<T>, T) {
    let x = x_s[0] * x_s[1].cos() * x_s[2].sin();
    let y = x_s[0] * x_s[1].sin() * x_s[2].sin();
    let z = x_s[0] * x_s[2].cos();

    let jac = x_s[0] * x_s[0] * x_s[2].sin();

    return (LorentzVector::from_args(T::zero(), x, y, z), jac);
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BoxSubtractionSettings {
    box_yaml_path: String,
    // use hardcoded Box4E, so we don't need extra code to find overlap structure
    //this can be changed later
}

pub struct BoxSubtractionIntegrand {
    _integrand_settings: BoxSubtractionSettings,
    integrand_f64: [RegulatedChannel<f64>; 4],
    // integrand_f128: [RegulatedChanel<f128>; 4],
}

impl BoxSubtractionIntegrand {
    pub fn new(integrand_settings: BoxSubtractionSettings) -> Self {
        let integrand_yaml_file = fs::read_to_string(&integrand_settings.box_yaml_path)
            .expect("failed to load integrand");
        let integrand_yaml =
            &YamlLoader::load_from_str(&integrand_yaml_file).expect("failed to load yaml")[0];

        let mut terms_vec = vec![];

        for i in 1.. {
            match &integrand_yaml[i] {
                Yaml::BadValue => break,
                _ => terms_vec.push(&integrand_yaml[i]),
            }
        }

        // Box4E
        let p1 = LorentzVector::from_args(14.0, -6.6, -40.0, 0.0);
        let p2 = LorentzVector::from_args(-43.0, 15.2, 33.0, 0.0);
        let p3 = LorentzVector::from_args(-17.9, -50.0, 11.8, 0.0);

        let p = [LorentzVector::new(), p1, p1 + p2, p1 + p2 + p3];
        let energies = vec![
            Energy::new(p[0], 0.0, 0),
            Energy::new(p[1], 0.0, 1),
            Energy::new(p[2], 0.0, 2),
            Energy::new(p[3], 0.0, 3),
        ];

        let mut surface_list = vec![];
        let mut cff_terms = vec![];
        for term in terms_vec.iter() {
            let mut orientation = vec![];

            for i in 1.. {
                match &term["Orientation"][i] {
                    Yaml::BadValue => break,
                    _ => orientation.push(&term["Orientation"][i]),
                }
            }

            let orientations: Vec<i64> = orientation
                .into_iter()
                .map(|z| z.clone().into_i64().unwrap())
                .collect();

            let subterms = &term["Terms"];

            for i in 0.. {
                let esurfs = &subterms[i]["Esurfs"];
                match esurfs {
                    Yaml::BadValue => break,
                    _ => {
                        let mut surfaces = vec![];
                        for j in 0..3 {
                            let ose1 = esurfs[j]["OSE"][0].clone().into_i64().unwrap() as usize - 1;
                            let ose2 = esurfs[j]["OSE"][1].clone().into_i64().unwrap() as usize - 1;

                            let orientation_tuple = (orientations[ose1], orientations[ose2]);

                            let surface_energies;

                            if orientation_tuple == (1, -1) {
                                surface_energies = [energies[ose1], energies[ose2]];
                            } else {
                                surface_energies = [energies[ose2], energies[ose1]];
                            }

                            let mut surface = ESurface::new(surface_energies, 0);

                            let mut original_surface = true;
                            for existing_surface in surface_list.iter() {
                                if surface.simple_eq(*existing_surface) {
                                    surface.label = existing_surface.label;
                                    original_surface = false;
                                }
                            }

                            if original_surface {
                                surface.label = surface_list.len();
                                surface_list.push(surface);
                            }

                            surfaces.push(surface);
                        }
                        cff_terms.push(ESurfaceProduct { factors: surfaces })
                    }
                }
            }
        }

        let mut existing_surfaces = vec![];
        for surface in surface_list.iter() {
            if surface.exists() {
                existing_surfaces.push(surface);
            }
        }

        let mut groups = vec![];

        // hardcode the max overlap structure

        groups.push(vec![surface_list[6], surface_list[7]]);
        groups.push(vec![surface_list[6], surface_list[10]]);
        groups.push(vec![surface_list[7], surface_list[11]]);
        groups.push(vec![surface_list[10], surface_list[11]]);

        let raw_integrand = CrossFreeFamilyIntegrand {
            terms: cff_terms,
            esurfaces: surface_list,
            energies: energies.clone(),
            e_cm: (p1 + p2).square().sqrt(),
        };

        let channel1 = raw_integrand.construct_channel(
            groups.clone(),
            groups[0].clone(),
            groups[3].clone(),
            -energies[0].p,
        );

        let channel2 = raw_integrand.construct_channel(
            groups.clone(),
            groups[1].clone(),
            groups[2].clone(),
            -energies[3].p,
        );

        let channel3 = raw_integrand.construct_channel(
            groups.clone(),
            groups[2].clone(),
            groups[1].clone(),
            -energies[2].p,
        );

        let channel4 = raw_integrand.construct_channel(
            groups.clone(),
            groups[3].clone(),
            groups[0].clone(),
            -energies[1].p,
        );

        let regulated_channels = [
            channel1.regulate(),
            channel2.regulate(),
            channel3.regulate(),
            channel4.regulate(),
        ];

        BoxSubtractionIntegrand {
            _integrand_settings: integrand_settings.clone(),
            integrand_f64: regulated_channels,
        }
    }
}

#[allow(unused)]
impl HasIntegrand for BoxSubtractionIntegrand {
    fn get_n_dim(&self) -> usize {
        3
    }

    fn create_grid(&self) -> Grid<f64> {
        Grid::Discrete(DiscreteGrid::new(
            vec![
                Some(Grid::Continuous(ContinuousGrid::new(
                    3, 10, 1000, None, false,
                ))),
                Some(Grid::Continuous(ContinuousGrid::new(
                    3, 10, 1000, None, false,
                ))),
                Some(Grid::Continuous(ContinuousGrid::new(
                    3, 10, 1000, None, false,
                ))),
                Some(Grid::Continuous(ContinuousGrid::new(
                    3, 10, 1000, None, false,
                ))),
            ],
            0.01,
            false,
        ))
    }

    fn evaluate_sample(
        &mut self,
        sample: &Sample<f64>,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        if let Sample::Discrete(weight, xs, cont_sample) = &sample {
            if let Sample::Continuous(_cont_weight, cs) = &**cont_sample.as_ref().unwrap() {
                let res =
                    -self.integrand_f64[*xs].evaluate_hypercube(cs) * 2.0 / std::f64::consts::PI;
                return Complex::new(res, 0.0);
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }
}
