use crate::{utils, Settings};

use crate::loop_induced_triboxtri::{LoopInducedTriBoxTriIntegrand, LoopInducedTriBoxTriSettings};
use crate::utils::FloatLike;
use enum_dispatch::enum_dispatch;
use havana::Sample;
use havana::{ContinuousGrid, Grid};
use lorentz_vector::LorentzVector;
use num::Complex;
use serde::Deserialize;
use std::fmt::{Display, Formatter};

#[derive(Debug, Copy, Clone, PartialEq, Deserialize)]
pub enum HardCodedIntegrand {
    #[serde(rename = "unit_surface")]
    UnitSurface,
    #[serde(rename = "unit_volume")]
    UnitVolume,
    #[serde(rename = "loop_induced_TriBoxTri")]
    LoopInducedTriBoxTri,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
#[serde(tag = "type")]
pub enum HardCodedIntegrandSettings {
    #[serde(rename = "unit_surface")]
    UnitSurface(UnitSurfaceSettings),
    #[serde(rename = "unit_volume")]
    UnitVolume(UnitVolumeSettings),
    #[serde(rename = "loop_induced_TriBoxTri")]
    LoopInducedTriBoxTri(LoopInducedTriBoxTriSettings),
}

impl Display for HardCodedIntegrandSettings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HardCodedIntegrandSettings::UnitSurface(_) => write!(f, "unit_surface"),
            HardCodedIntegrandSettings::UnitVolume(_) => write!(f, "unit_volume"),
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(_) => {
                write!(f, "loop_induced_TriBoxTri")
            }
        }
    }
}

impl Default for HardCodedIntegrandSettings {
    fn default() -> HardCodedIntegrandSettings {
        HardCodedIntegrandSettings::UnitSurface(UnitSurfaceSettings { n_dim: 8 })
    }
}

pub struct Esurface {
    pub edge_ids: Vec<usize>,
    pub shift: Vec<usize>,
}
#[allow(unused)]
impl Esurface {
    pub fn new(edge_ids: Vec<usize>, shift: Vec<usize>) -> Esurface {
        Esurface { edge_ids, shift }
    }
}
pub struct CFFDenom {
    pub e_surfaces: Vec<Esurface>,
}
#[allow(unused)]
impl CFFDenom {
    pub fn new(e_surfaces: Vec<Esurface>) -> CFFDenom {
        CFFDenom { e_surfaces }
    }
}

pub struct CFFFactor {
    pub denoms: Vec<CFFDenom>,
}
#[allow(unused)]
impl CFFFactor {
    pub fn new(denoms: Vec<CFFDenom>) -> CFFFactor {
        CFFFactor { denoms }
    }
}
pub struct CFFTerm {
    pub orientation: Vec<bool>,
    pub factors: Vec<CFFFactor>,
}
#[allow(unused)]
impl CFFTerm {
    pub fn new(orientation: Vec<bool>, factors: Vec<CFFFactor>) -> CFFTerm {
        CFFTerm {
            orientation,
            factors,
        }
    }
}
pub struct CFFExpression {
    pub terms: Vec<CFFTerm>,
}
#[allow(unused)]
impl CFFExpression {
    pub fn new(terms: Vec<CFFTerm>) -> CFFExpression {
        CFFExpression { terms }
    }
}
pub struct Amplitude {
    pub edges: Vec<Edge>,
    pub thresholds: Vec<Esurface>,
    pub cff_expression: CFFExpression,
    pub n_loop: usize,
}
#[allow(unused)]
impl Amplitude {
    pub fn new(
        edges: Vec<Edge>,
        cff_expression: CFFExpression,
        thresholds: Vec<Esurface>,
        n_loop: usize,
    ) -> Amplitude {
        Amplitude {
            edges,
            cff_expression,
            thresholds,
            n_loop,
        }
    }
}
pub struct Cut {
    pub cut_edge_ids: Vec<usize>,
    pub left_amplitude: Amplitude,
    pub right_amplitude: Amplitude,
}
#[allow(unused)]
impl Cut {
    pub fn new(
        cut_edge_ids: Vec<usize>,
        left_amplitude: Amplitude,
        right_amplitude: Amplitude,
    ) -> Cut {
        Cut {
            cut_edge_ids,
            left_amplitude,
            right_amplitude,
        }
    }
}
pub struct Edge {
    pub mass: f64,
    pub signature: (Vec<isize>, Vec<isize>),
    pub id: usize,
}
#[allow(unused)]
impl Edge {
    pub fn new(mass: f64, signature: (Vec<isize>, Vec<isize>), id: usize) -> Edge {
        Edge {
            mass,
            signature,
            id,
        }
    }
}
pub struct SuperGraph {
    pub edges: Vec<Edge>,
    pub cuts: Vec<Cut>,
    pub n_loop: usize,
}
#[allow(unused)]
impl SuperGraph {
    pub fn new(edges: Vec<Edge>, cuts: Vec<Cut>, n_loop: usize) -> SuperGraph {
        SuperGraph {
            edges,
            cuts,
            n_loop,
        }
    }

    pub fn default() -> SuperGraph {
        SuperGraph {
            edges: Vec::new(),
            cuts: Vec::new(),
            n_loop: 0,
        }
    }
}

#[enum_dispatch]
pub trait HasIntegrand {
    fn create_grid(&self) -> Grid;

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64>;

    fn get_n_dim(&self) -> usize;
}

#[enum_dispatch(HasIntegrand)]
pub enum Integrand {
    UnitSurface(UnitSurfaceIntegrand),
    UnitVolume(UnitVolumeIntegrand),
    LoopInducedTriBoxTri(LoopInducedTriBoxTriIntegrand),
}

pub fn integrand_factory(settings: &Settings) -> Integrand {
    match settings.hard_coded_integrand.clone() {
        HardCodedIntegrandSettings::UnitSurface(integrand_settings) => Integrand::UnitSurface(
            UnitSurfaceIntegrand::new(settings.clone(), integrand_settings),
        ),
        HardCodedIntegrandSettings::UnitVolume(integrand_settings) => Integrand::UnitVolume(
            UnitVolumeIntegrand::new(settings.clone(), integrand_settings),
        ),
        HardCodedIntegrandSettings::LoopInducedTriBoxTri(integrand_settings) => {
            Integrand::LoopInducedTriBoxTri(LoopInducedTriBoxTriIntegrand::new(
                settings.clone(),
                integrand_settings,
            ))
        }
    }
}

pub fn compute_surface_and_volume(n_dim: usize, radius: f64) -> (f64, f64) {
    let mut surface = 2.0;
    let mut volume = 1.0;
    for i in 1..n_dim + 1 {
        (surface, volume) = (2. * std::f64::consts::PI * volume, surface / (i as f64));
    }
    (
        surface * radius.powi(n_dim as i32),
        volume * radius.powi(n_dim as i32),
    )
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UnitSurfaceSettings {
    pub n_dim: usize,
}

pub struct UnitSurfaceIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub supergraph: SuperGraph,
    pub surface: f64,
}

#[allow(unused)]
impl UnitSurfaceIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: UnitSurfaceSettings,
    ) -> UnitSurfaceIntegrand {
        if (integrand_settings.n_dim + 1) % 3 != 0 {
            panic!("n_dim must be a multiple of 3 for unit surface integrand");
        }
        let surface =
            compute_surface_and_volume(integrand_settings.n_dim, settings.kinematics.e_cm).0;
        UnitSurfaceIntegrand {
            settings,
            n_dim: integrand_settings.n_dim,
            supergraph: SuperGraph::default(),
            surface: surface,
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
impl HasIntegrand for UnitSurfaceIntegrand {
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
        // Normalize the integral
        itg_wgt /= self.surface;
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
            println!("Final contribution: {:+.16e}", itg_wgt * wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * wgt * jac;
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UnitVolumeSettings {
    pub n_dim: usize,
}

pub struct UnitVolumeIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub supergraph: SuperGraph,
    pub volume: f64,
}

#[allow(unused)]
impl UnitVolumeIntegrand {
    pub fn new(settings: Settings, integrand_settings: UnitVolumeSettings) -> UnitVolumeIntegrand {
        if (integrand_settings.n_dim % 3) != 0 {
            panic!("n_dim must be a multiple of 3 for unit volume integrand");
        }
        let volume =
            compute_surface_and_volume(integrand_settings.n_dim, settings.kinematics.e_cm).1;
        UnitVolumeIntegrand {
            settings,
            n_dim: integrand_settings.n_dim,
            supergraph: SuperGraph::default(),
            volume: volume,
        }
    }

    fn evaluate_numerator<T: FloatLike>(&self, loop_momenta: &[LorentzVector<T>]) -> T {
        if loop_momenta
            .iter()
            .map(|l| l.spatial_squared())
            .sum::<T>()
            .sqrt()
            > Into::<T>::into(self.settings.kinematics.e_cm)
        {
            return T::from_f64(0.0).unwrap();
        } else {
            return T::from_f64(1.0).unwrap();
        }
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            false,
        )
    }
}

#[allow(unused)]
impl HasIntegrand for UnitVolumeIntegrand {
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
        let (moms, jac) = self.parameterize(xs);
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
        // Normalize the integral
        itg_wgt /= self.volume;
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
            println!("Final contribution: {:+.16e}", itg_wgt * wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * wgt * jac;
    }
}
