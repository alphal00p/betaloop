use crate::box_subtraction::{BoxSubtractionIntegrand, BoxSubtractionSettings};
use crate::{utils, Settings};

use crate::loop_induced_triboxtri::{LoopInducedTriBoxTriIntegrand, LoopInducedTriBoxTriSettings};
use crate::triangle_subtraction::{TriangleSubtractionIntegrand, TriangleSubtractionSettings};
use crate::utils::FloatLike;
use color_eyre::{Help, Report};
use enum_dispatch::enum_dispatch;
use eyre::WrapErr;
use havana::Sample;
use havana::{ContinuousGrid, Grid};
use lorentz_vector::LorentzVector;
use num::Complex;
use serde::Deserialize;
use std::fmt::{Display, Formatter};
use std::fs::File;

#[derive(Debug, Copy, Clone, PartialEq, Deserialize)]
pub enum HardCodedIntegrand {
    #[serde(rename = "unit_surface")]
    UnitSurface,
    #[serde(rename = "unit_volume")]
    UnitVolume,
    #[serde(rename = "loop_induced_TriBoxTri")]
    LoopInducedTriBoxTri,
    #[serde(rename = "triangle_subtraction")]
    TriangleSubtraction,
    #[serde(rename = "box_subtraction")]
    BoxSubtraction,
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
    #[serde(rename = "triangle_subtraction")]
    TriangleSubtraction(TriangleSubtractionSettings),
    #[serde(rename = "box_subtraction")]
    BoxSubtraction(BoxSubtractionSettings),
}

impl Display for HardCodedIntegrandSettings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HardCodedIntegrandSettings::UnitSurface(_) => write!(f, "unit_surface"),
            HardCodedIntegrandSettings::UnitVolume(_) => write!(f, "unit_volume"),
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(_) => {
                write!(f, "loop_induced_TriBoxTri")
            }
            HardCodedIntegrandSettings::TriangleSubtraction(_) => {
                write!(f, "triangle_subtraction")
            }
            HardCodedIntegrandSettings::BoxSubtraction(_) => {
                write!(f, "box_subtraction")
            }
        }
    }
}

impl Default for HardCodedIntegrandSettings {
    fn default() -> HardCodedIntegrandSettings {
        HardCodedIntegrandSettings::UnitSurface(UnitSurfaceSettings { n_3d_momenta: 1 })
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Esurface {
    pub edge_ids: Vec<usize>,
    pub shift: Vec<isize>,
    pub id: usize,
}
#[allow(unused)]
impl Esurface {
    pub fn new(edge_ids: Vec<usize>, shift: Vec<isize>, id: usize) -> Esurface {
        Esurface {
            edge_ids,
            shift,
            id,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CFFFactor {
    // Denominator consists of a list of E-surface IDs appearing in the denominator
    pub denominator: Vec<usize>,
    pub factors: Vec<CFFFactor>,
}
#[allow(unused)]
impl CFFFactor {
    pub fn new(denominator: Vec<usize>, factors: Vec<CFFFactor>) -> CFFFactor {
        CFFFactor {
            denominator,
            factors,
        }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CFFTerm {
    pub orientation: Vec<(usize, isize)>,
    pub factors: Vec<CFFFactor>,
}
#[allow(unused)]
impl CFFTerm {
    pub fn new(orientation: Vec<(usize, isize)>, factors: Vec<CFFFactor>) -> CFFTerm {
        CFFTerm {
            orientation,
            factors,
        }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CFFExpression {
    pub terms: Vec<CFFTerm>,
    pub e_surfaces: Vec<Esurface>,
}
#[allow(unused)]
impl CFFExpression {
    pub fn new(terms: Vec<CFFTerm>, e_surfaces: Vec<Esurface>) -> CFFExpression {
        CFFExpression { terms, e_surfaces }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Amplitude {
    pub edges: Vec<Edge>,
    pub lmb_edges: Vec<Edge>,
    pub external_edge_id_and_flip: Vec<(usize, isize)>,
    // Thresholds given as list of E-surface ids in the cff_expression
    pub thresholds: Vec<usize>,
    pub cff_expression: CFFExpression,
    pub n_loop: usize,
}
#[allow(unused)]
impl Amplitude {
    pub fn new(
        edges: Vec<Edge>,
        lmb_edges: Vec<Edge>,
        external_edge_id_and_flip: Vec<(usize, isize)>,
        cff_expression: CFFExpression,
        thresholds: Vec<usize>,
        n_loop: usize,
    ) -> Amplitude {
        Amplitude {
            edges,
            lmb_edges,
            external_edge_id_and_flip,
            cff_expression,
            thresholds,
            n_loop,
        }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Cut {
    pub cut_edge_ids_and_flip: Vec<(usize, isize)>,
    pub left_amplitude: Amplitude,
    pub right_amplitude: Amplitude,
}
#[allow(unused)]
impl Cut {
    pub fn new(
        cut_edge_ids_and_flip: Vec<(usize, isize)>,
        left_amplitude: Amplitude,
        right_amplitude: Amplitude,
    ) -> Cut {
        Cut {
            cut_edge_ids_and_flip,
            left_amplitude,
            right_amplitude,
        }
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Edge {
    pub mass: f64,
    pub signature: (Vec<isize>, Vec<isize>),
    pub id: usize,
    pub power: usize,
}
#[allow(unused)]
impl Edge {
    pub fn new(mass: f64, signature: (Vec<isize>, Vec<isize>), power: usize, id: usize) -> Edge {
        Edge {
            mass,
            signature,
            id,
            power,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
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

    pub fn from_file(filename: &str) -> Result<SuperGraph, Report> {
        let f = File::open(filename)
            .wrap_err_with(|| format!("Could not open supergraph file {}", filename))
            .suggestion("Does the path exist?")?;
        serde_yaml::from_reader(f)
            .wrap_err("Could not parse supergraph file")
            .suggestion("Is it a correct YAML file")
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
    TriangleSubtraction(TriangleSubtractionIntegrand),
    BoxSubtraction(BoxSubtractionIntegrand),
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
        HardCodedIntegrandSettings::TriangleSubtraction(integrand_settings) => {
            Integrand::TriangleSubtraction(TriangleSubtractionIntegrand::new(integrand_settings))
        }
        HardCodedIntegrandSettings::BoxSubtraction(integrand_settings) => {
            Integrand::BoxSubtraction(BoxSubtractionIntegrand::new(integrand_settings))
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UnitSurfaceSettings {
    pub n_3d_momenta: usize,
}

pub struct UnitSurfaceIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub n_3d_momenta: usize,
    pub supergraph: SuperGraph,
    pub surface: f64,
}

#[allow(unused)]
impl UnitSurfaceIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: UnitSurfaceSettings,
    ) -> UnitSurfaceIntegrand {
        let n_dim =
            utils::get_n_dim_for_n_loop_momenta(&settings, integrand_settings.n_3d_momenta, true);
        let surface = utils::compute_surface_and_volume(
            integrand_settings.n_3d_momenta * 3 - 1,
            settings.kinematics.e_cm,
        )
        .0;
        UnitSurfaceIntegrand {
            settings,
            n_3d_momenta: integrand_settings.n_3d_momenta,
            n_dim: n_dim,
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
            println!("Final contribution: {:+.16e}", itg_wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * jac;
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UnitVolumeSettings {
    pub n_3d_momenta: usize,
}

pub struct UnitVolumeIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub n_3d_momenta: usize,
    pub supergraph: SuperGraph,
    pub volume: f64,
}

#[allow(unused)]
impl UnitVolumeIntegrand {
    pub fn new(settings: Settings, integrand_settings: UnitVolumeSettings) -> UnitVolumeIntegrand {
        let n_dim =
            utils::get_n_dim_for_n_loop_momenta(&settings, integrand_settings.n_3d_momenta, false);
        let volume = utils::compute_surface_and_volume(
            integrand_settings.n_3d_momenta * 3,
            settings.kinematics.e_cm,
        )
        .1;
        UnitVolumeIntegrand {
            settings,
            n_3d_momenta: integrand_settings.n_3d_momenta,
            n_dim: n_dim,
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
                t: 0.,
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
            println!("Final contribution: {:+.16e}", itg_wgt * jac);
        }
        return Complex::new(itg_wgt, 0.) * jac;
    }
}
