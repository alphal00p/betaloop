mod utils;

use clap::{App, Arg, ArgMatches, SubCommand};
use color_eyre::{Help, Report};
use colored::Colorize;
use enum_dispatch::enum_dispatch;
use eyre::WrapErr;
use f128::f128;
use havana::{AverageAndErrorAccumulator, Sample};
use havana::{ContinuousGrid, Grid};
use lorentz_vector::{Field, LorentzVector, RealNumberLike};
use num::Complex;
use num_traits::{Float, FloatConst, FromPrimitive, Num, Signed, ToPrimitive};
use rayon::prelude::*;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str::FromStr;
use tabled::{Style, Table, Tabled};
use utils::Signum;

use git_version::git_version;
use serde::{Deserialize, Serialize};
const GIT_VERSION: &str = git_version!();

pub const MAX_CORES: usize = 1000;

#[cfg(not(feature = "higher_loops"))]
pub const MAX_LOOP: usize = 3;
#[cfg(feature = "higher_loops")]
pub const MAX_LOOP: usize = 6;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum ParameterizationMode {
    #[serde(rename = "cartesian")]
    Cartesian,
    #[serde(rename = "spherical")]
    Spherical,
    #[serde(rename = "hyperspherical")]
    HyperSpherical,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum ParameterizationMapping {
    #[serde(rename = "log")]
    Log,
    #[serde(rename = "linear")]
    Linear,
}

pub trait FloatConvertFrom<U> {
    fn convert_from(x: &U) -> Self;
}

impl FloatConvertFrom<f64> for f64 {
    fn convert_from(x: &f64) -> f64 {
        *x
    }
}

impl FloatConvertFrom<f128> for f64 {
    fn convert_from(x: &f128) -> f64 {
        (*x).to_f64().unwrap()
    }
}

impl FloatConvertFrom<f128> for f128 {
    fn convert_from(x: &f128) -> f128 {
        *x
    }
}

impl FloatConvertFrom<f64> for f128 {
    fn convert_from(x: &f64) -> f128 {
        f128::from_f64(*x).unwrap()
    }
}

pub trait FloatLike:
    From<f64>
    + FloatConvertFrom<f64>
    + FloatConvertFrom<f128>
    + Num
    + FromPrimitive
    + Float
    + Field
    + RealNumberLike
    + Signed
    + FloatConst
    + std::fmt::LowerExp
    + 'static
    + Signum
{
}

impl FloatLike for f64 {}
impl FloatLike for f128 {}

#[derive(Debug, Copy, Clone, PartialEq, Deserialize)]
pub enum HardCodedIntegrand {
    #[serde(rename = "unit")]
    Unit,
    #[serde(rename = "loop_induced_TriBoxTri")]
    LoopInducedTriBoxTri,
}

impl Display for HardCodedIntegrand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            HardCodedIntegrand::Unit => write!(f, "unit"),
            HardCodedIntegrand::LoopInducedTriBoxTri => write!(f, "loop_induced_TriBoxTri"),
        }
    }
}

impl Default for HardCodedIntegrand {
    fn default() -> HardCodedIntegrand {
        HardCodedIntegrand::Unit
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GeneralSettings {
    pub debug: usize,
    pub hard_coded_integrand: HardCodedIntegrand,
}

#[derive(Debug, Copy, Clone, PartialEq, Deserialize)]
pub enum IntegratedPhase {
    #[serde(rename = "real")]
    Real,
    #[serde(rename = "imag")]
    Imag,
    #[serde(rename = "both")]
    Both,
}

impl Default for IntegratedPhase {
    fn default() -> IntegratedPhase {
        IntegratedPhase::Real
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct KinematicsSettings {
    pub e_cm: f64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct IntegratorSettings {
    pub n_bins: usize,
    pub min_samples_for_update: usize,
    pub n_start: usize,
    pub n_increase: usize,
    pub n_max: usize,
    pub integrated_phase: IntegratedPhase,
    pub learning_rate: f64,
    pub train_on_avg: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ParameterizationSettings {
    pub mode: ParameterizationMode,
    pub mapping: ParameterizationMapping,
    pub b: f64,
    pub input_rescaling: Vec<Vec<(f64, f64)>>,
    pub shifts: Vec<(f64, f64, f64, f64)>,
}

impl Default for ParameterizationMapping {
    fn default() -> ParameterizationMapping {
        ParameterizationMapping::Log
    }
}

impl Default for ParameterizationMode {
    fn default() -> ParameterizationMode {
        ParameterizationMode::Spherical
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Settings {
    #[serde(rename = "General")]
    pub general: GeneralSettings,
    #[serde(rename = "Kinematics")]
    pub kinematics: KinematicsSettings,
    #[serde(rename = "Parameterization")]
    pub parameterization: ParameterizationSettings,
    #[serde(rename = "Integrator")]
    pub integrator: IntegratorSettings,
}

impl Settings {
    pub fn from_file(filename: &str) -> Result<Settings, Report> {
        let f = File::open(filename)
            .wrap_err_with(|| format!("Could not open settings file {}", filename))
            .suggestion("Does the path exist?")?;
        serde_yaml::from_reader(f)
            .wrap_err("Could not parse settings file")
            .suggestion("Is it a correct yaml file")
    }
}
pub struct Esurface {
    pub edge_ids: Vec<usize>,
    pub shift: Vec<usize>,
}
impl Esurface {
    pub fn new(edge_ids: Vec<usize>, shift: Vec<usize>) -> Esurface {
        Esurface { edge_ids, shift }
    }
}
pub struct cFFDenom {
    pub e_surfaces: Vec<Esurface>,
}
impl cFFDenom {
    pub fn new(e_surfaces: Vec<Esurface>) -> cFFDenom {
        cFFDenom { e_surfaces }
    }
}

pub struct cFFFactor {
    pub denoms: Vec<cFFDenom>,
}
impl cFFFactor {
    pub fn new(denoms: Vec<cFFDenom>) -> cFFFactor {
        cFFFactor { denoms }
    }
}
pub struct cFFTerm {
    pub orientation: Vec<bool>,
    pub factors: Vec<cFFFactor>,
}
impl cFFTerm {
    pub fn new(orientation: Vec<bool>, factors: Vec<cFFFactor>) -> cFFTerm {
        cFFTerm {
            orientation,
            factors,
        }
    }
}
pub struct cFFExpression {
    pub terms: Vec<cFFTerm>,
}
impl cFFExpression {
    pub fn new(terms: Vec<cFFTerm>) -> cFFExpression {
        cFFExpression { terms }
    }
}
pub struct Amplitude {
    pub edges: Vec<Edge>,
    pub thresholds: Vec<Esurface>,
    pub cFF_expression: cFFExpression,
    pub n_loop: usize,
}

impl Amplitude {
    pub fn new(
        edges: Vec<Edge>,
        cFF_expression: cFFExpression,
        thresholds: Vec<Esurface>,
        n_loop: usize,
    ) -> Amplitude {
        Amplitude {
            edges,
            cFF_expression,
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
    fn evaluate_numerator<T: FloatLike>(
        &self,
        supergraph: &SuperGraph,
        loop_momenta: &[LorentzVector<T>],
    ) -> T;

    fn create_grid(&self) -> Grid;

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T);

    fn evaluate_sample(&self, sample: &Sample, wgt: f64, iter: usize) -> Complex<f64>;
}

#[enum_dispatch(HasIntegrand)]
enum Integrand {
    Unit(UnitIntegrand),
    LoopInducedTriBoxTri(LoopInducedTriBoxTrIntegrand),
}

pub struct UnitIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
    pub supergraph: SuperGraph,
}

impl UnitIntegrand {
    pub fn new(settings: Settings, n_dim: usize) -> UnitIntegrand {
        UnitIntegrand {
            settings,
            n_dim,
            supergraph: SuperGraph::default(),
        }
    }
}

impl HasIntegrand for UnitIntegrand {
    fn evaluate_numerator<T: FloatLike>(
        &self,
        supergraph: &SuperGraph,
        loop_momenta: &[LorentzVector<T>],
    ) -> T {
        return T::from_f64(1.0).unwrap();
    }

    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(
            self.n_dim,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
        ))
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            true,
        )
    }

    fn evaluate_sample(&self, sample: &Sample, wgt: f64, iter: usize) -> Complex<f64> {
        let xs = match sample {
            Sample::ContinuousGrid(_w, v) => v,
            _ => panic!("Wrong sample type"),
        };
        let mut sample_xs = vec![self.settings.kinematics.e_cm];
        sample_xs.extend(xs);
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
        let wgt = self.evaluate_numerator(&self.supergraph, loop_momenta.as_slice());
        return Complex::new(wgt, 0.) * jac;
    }
}

pub struct LoopInducedTriBoxTrIntegrand {
    pub settings: Settings,
    pub n_dim: usize,
}

impl LoopInducedTriBoxTrIntegrand {
    pub fn new(settings: Settings) -> LoopInducedTriBoxTrIntegrand {
        LoopInducedTriBoxTrIntegrand { settings, n_dim: 8 }
    }
}

impl HasIntegrand for LoopInducedTriBoxTrIntegrand {
    fn evaluate_numerator<T: FloatLike>(
        &self,
        supergraph: &SuperGraph,
        loop_momenta: &[LorentzVector<T>],
    ) -> T {
        return T::from_f64(1.0).unwrap();
    }

    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(
            self.n_dim,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
        ))
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            true,
        )
    }

    fn evaluate_sample(&self, sample: &Sample, wgt: f64, iter: usize) -> Complex<f64> {
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

#[derive(Serialize, Deserialize)]
pub struct IntegrationResult {
    pub neval: i64,
    pub fail: i32,
    pub result: Vec<f64>,
    pub error: Vec<f64>,
    pub prob: Vec<f64>,
}

#[derive(Tabled)]
struct IntegralResult {
    id: String,
    n_samples: String,
    #[tabled(rename = "n_samples[%]")]
    n_samples_perc: String,
    #[tabled(rename = "<I>")]
    integral: String,
    #[tabled(rename = "sqrt(σ)")]
    variance: String,
    err: String,
    #[tabled(err = "err[%]")]
    err_perc: String,
    #[tabled(err = "PDF")]
    pdf: String,
}

fn havana_integrate<F>(
    settings: &Settings,
    user_data_generator: F,
    target: Option<Complex<f64>>,
) -> IntegrationResult
where
    F: Fn(&Settings) -> UserData,
{
    let mut num_points = 0;
    const N_INTEGRAND_ACCUMULATORS: usize = 2;

    let mut samples = vec![Sample::new(); settings.integrator.n_start];
    let mut f_evals = vec![vec![0.; N_INTEGRAND_ACCUMULATORS]; settings.integrator.n_start];
    let mut integral = AverageAndErrorAccumulator::new();
    let mut all_integrals = vec![AverageAndErrorAccumulator::new(); N_INTEGRAND_ACCUMULATORS];

    let mut rng = rand::thread_rng();

    let user_data = user_data_generator(settings);

    let mut grid = user_data.integrand[0].create_grid();

    let mut iter = 1;

    let cores = user_data.integrand.len();

    while num_points < settings.integrator.n_max {
        let cur_points = settings.integrator.n_start + settings.integrator.n_increase * (iter - 1);
        samples.resize(cur_points, Sample::new());
        f_evals.resize(cur_points, vec![0.; N_INTEGRAND_ACCUMULATORS]);

        for sample in &mut samples[..cur_points] {
            grid.sample(&mut rng, sample);
        }

        // the number of points per core for all cores but the last, which may have fewer
        let nvec_per_core = (cur_points - 1) / cores + 1;

        user_data.integrand[..cores]
            .into_par_iter()
            .zip(f_evals.par_chunks_mut(nvec_per_core))
            .zip(samples.par_chunks(nvec_per_core))
            .for_each(|((integrand_f, ff), xi)| {
                for (f_evals_i, s) in ff.iter_mut().zip(xi.iter()) {
                    let fc = integrand_f.evaluate_sample(s, s.get_weight(), iter);
                    f_evals_i[0] = fc.re;
                    f_evals_i[1] = fc.im;
                }
            });

        for (s, f) in samples[..cur_points].iter().zip(&f_evals[..cur_points]) {
            let sel_f = match settings.integrator.integrated_phase {
                IntegratedPhase::Real => &f[0],
                IntegratedPhase::Imag => &f[1],
                IntegratedPhase::Both => unimplemented!(),
            };
            grid.add_training_sample(s, *sel_f).unwrap();
            integral.add_sample(*sel_f * s.get_weight(), Some(s));
        }

        grid.update(
            settings.integrator.learning_rate,
            settings.integrator.n_bins,
            settings.integrator.train_on_avg,
        );
        integral.update_iter();

        for i_integrand in 0..N_INTEGRAND_ACCUMULATORS {
            for (s, f) in samples[..cur_points].iter().zip(&f_evals[..cur_points]) {
                all_integrals[i_integrand].add_sample(f[i_integrand] * s.get_weight(), Some(s));
            }
            all_integrals[i_integrand].update_iter()
        }

        if settings.general.debug > 1 {
            if let havana::Grid::DiscreteGrid(g) = &grid {
                g.discrete_dimensions[0]
                    .plot(&format!("grid_disc_it_{}.svg", iter))
                    .unwrap();
            }
            let mut tabled_data = vec![];

            tabled_data.push(IntegralResult {
                id: format!("Sum@it#{}", integral.cur_iter),
                n_samples: format!("{}", integral.processed_samples),
                n_samples_perc: format!("{:.3e}%", 100.),
                integral: format!("{:.8e}", integral.avg),
                variance: format!(
                    "{:.8e}",
                    integral.err * ((integral.processed_samples - 1).max(0) as f64).sqrt()
                ),
                err: format!("{:.8e}", integral.err),
                err_perc: format!(
                    "{:.3e}%",
                    (integral.err / (integral.avg.abs()).max(1.0e-99)).abs() * 100.
                ),
                pdf: String::from_str("N/A").unwrap(),
            });
            if let havana::Grid::DiscreteGrid(g) = &grid {
                for (i, b) in g.discrete_dimensions[0].bin_accumulator.iter().enumerate() {
                    tabled_data.push(IntegralResult {
                        id: format!("chann#{}", i),
                        n_samples: format!("{}", b.processed_samples),
                        n_samples_perc: format!(
                            "{:.3e}%",
                            ((b.processed_samples as f64)
                                / (integral.processed_samples.max(1) as f64))
                                * 100.
                        ),
                        integral: format!("{:.8e}", b.avg),
                        variance: format!(
                            "{:.8e}",
                            b.err * ((b.processed_samples - 1).max(0) as f64).sqrt()
                        ),
                        err: format!("{:.8e}", b.err),
                        err_perc: format!(
                            "{:.3e}%",
                            (b.err / (b.avg.abs()).max(1.0e-99)).abs() * 100.
                        ),
                        pdf: format!(
                            "{:.8e}",
                            if i > 1 {
                                g.discrete_dimensions[0].cdf[i]
                                    - g.discrete_dimensions[0].cdf[i - 1]
                            } else {
                                g.discrete_dimensions[0].cdf[i]
                            }
                        ),
                    });
                }
            }
            let mut f = BufWriter::new(
                File::create(&format!("results_it_{}.txt", iter))
                    .expect("Could not create results file"),
            );
            writeln!(
                f,
                "{}",
                Table::new(tabled_data).with(Style::psql()).to_string()
            )
            .unwrap();
        }

        iter += 1;
        num_points += cur_points;

        println!(
            "| Iteration #{:-4}: n_pts={:-10}, n_tot={:-10}",
            iter, cur_points, num_points
        );

        fn print_integral_result(
            itg: &AverageAndErrorAccumulator,
            i_itg: usize,
            trgt: Option<f64>,
        ) {
            println!(
                "|  itg #{} re: {} {} {:.2} χ² {} {}",
                i_itg,
                utils::format_uncertainty(itg.avg, itg.err),
                if itg.avg != 0. {
                    format!("({:.2}%)", (itg.err / itg.avg).abs() * 100.)
                } else {
                    format!("")
                },
                itg.chi_sq,
                if i_itg == 0 {
                    if let Some(t) = trgt {
                        format!(
                            "Δ={:.2}σ, Δ={:.2}%",
                            (t - itg.avg).abs() / itg.err,
                            (t - itg.avg).abs() / t.abs() * 100.
                        )
                    } else {
                        format!("")
                    }
                } else {
                    format!("")
                },
                if itg.avg.abs() != 0. {
                    format!(
                        "mwi: {:.2}",
                        itg.max_eval_negative.abs().max(itg.max_eval_positive.abs())
                            / (itg.avg.abs() * (itg.processed_samples as f64))
                    )
                } else {
                    format!("")
                }
            );
        }

        for i_integrand in 0..(N_INTEGRAND_ACCUMULATORS / 2) {
            print_integral_result(
                &all_integrals[2 * i_integrand],
                i_integrand + 1,
                if i_integrand == 0 {
                    target.map(|o| o.re).or(None)
                } else {
                    None
                },
            );
            print_integral_result(
                &all_integrals[2 * i_integrand + 1],
                i_integrand + 1,
                if i_integrand == 0 {
                    target.map(|o| o.im).or(None)
                } else {
                    None
                },
            );
        }
    }

    IntegrationResult {
        neval: integral.processed_samples as i64,
        fail: integral.num_zero_evals as i32,
        result: vec![integral.avg],
        error: vec![integral.err],
        prob: vec![integral.chi_sq],
    }
}

struct UserData {
    integrand: Vec<Integrand>,
}
fn main() -> Result<(), Report> {
    let matches = App::new("betaLoop")
        .version("0.1")
        .about("New breed of Local Unitarity implementation")
        .arg(
            Arg::with_name("cores")
                .short("c")
                .long("cores")
                .value_name("NUMCORES")
                .help("Set the number of cores"),
        )
        .arg(
            Arg::with_name("config")
                .short("f")
                .long("config")
                .value_name("CONFIG_FILE")
                .default_value("../betaloop_config.yaml")
                .help("Set the configuration file"),
        )
        .arg(
            Arg::with_name("target")
                .long("target")
                .multiple(true)
                .allow_hyphen_values(true)
                .value_name("TARGET")
                .help("Specify the integration target a <real> <imag>"),
        )
        .arg(
            Arg::with_name("debug")
                .long("debug")
                .value_name("LEVEL")
                .help("Set the debug level. Higher means more verbose."),
        )
        .arg(
            Arg::with_name("n_start")
                .long("n_start")
                .value_name("N_START")
                .help("Number of starting samples for Vegas"),
        )
        .get_matches();

    let mut settings = Settings::from_file(matches.value_of("config").unwrap())?;

    if let Some(x) = matches.value_of("debug") {
        settings.general.debug = usize::from_str(x).unwrap();
    }
    if let Some(x) = matches.value_of("n_start") {
        settings.integrator.n_start = usize::from_str(x).unwrap();
    }

    let mut cores = 1;
    if let Some(x) = matches.value_of("cores") {
        cores = usize::from_str(x).unwrap();
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(cores)
        .build_global()
        .unwrap();

    let num_integrands = cores;

    let mut target = None;
    if let Some(t) = matches.values_of("target") {
        let tt: Vec<_> = t
            .map(|x| f64::from_str(x.trim_end_matches(',')).unwrap())
            .collect();
        if tt.len() != 2 {
            panic!("Expected two numbers for target");
        }
        target = Some(Complex::new(tt[0], tt[1]));
    }

    let user_data_generator = |settings: &Settings| UserData {
        integrand: (0..num_integrands)
            .map(|_i| match settings.general.hard_coded_integrand {
                HardCodedIntegrand::Unit => {
                    Integrand::Unit(UnitIntegrand::new(settings.clone(), 8))
                }
                HardCodedIntegrand::LoopInducedTriBoxTri => Integrand::LoopInducedTriBoxTri(
                    LoopInducedTriBoxTrIntegrand::new(settings.clone()),
                ),
            })
            .collect(),
    };

    if settings.general.debug > 1 {
        println!("Debug mode enabled");
    }
    println!(
        r#"
    ______      _        _                       
    | ___ \    | |      | |                      
    | |_/ / ___| |_ __ _| |     ___   ___  _ __  
    | ___ \/ _ \ __/ _` | |    / _ \ / _ \| '_ \ 
    | |_/ /  __/ || (_| | |___| (_) | (_) | |_) |
    \____/ \___|\__\__,_\_____/\___/ \___/| .__/ 
                                          | |    
    {:-24}              |_|    
    "#,
        GIT_VERSION
    );
    println!();
    println!(
        "Beta loop is at your service and starts integrating '{}' ...",
        settings.general.hard_coded_integrand
    );
    println!();
    havana_integrate(&settings, user_data_generator, target);

    Ok(())
}
