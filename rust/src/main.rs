mod box_subtraction;
mod bubble;
mod h_function_test;
mod integrands;
mod loop_induced_triboxtri;
mod observables;
mod raised_bubble;
mod tbbt;
mod tests;
mod triangle_subtraction;
mod triboxtri;
mod triboxtri_cff;
mod triboxtri_cff_as;
mod triboxtri_cff_sectored;
mod utils;

use clap::{App, Arg, SubCommand};
use color_eyre::{Help, Report};
use colored::Colorize;
use eyre::WrapErr;
use integrands::*;
use lorentz_vector::LorentzVector;
use num::Complex;
use num_traits::ToPrimitive;
use observables::ObservableSettings;
use observables::PhaseSpaceSelectorSettings;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str::FromStr;
use std::time::Instant;
use symbolica::numerical_integration::Grid;
use symbolica::numerical_integration::{Sample, StatisticsAccumulator};
use tabled::{Style, Table, Tabled};

use git_version::git_version;
use serde::{Deserialize, Serialize};
const GIT_VERSION: &str = git_version!();

pub const MAX_CORES: usize = 1000;

#[cfg(not(feature = "higher_loops"))]
pub const MAX_LOOP: usize = 3;
#[cfg(feature = "higher_loops")]
pub const MAX_LOOP: usize = 6;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub enum HFunction {
    #[default]
    #[serde(rename = "poly_exponential")]
    PolyExponential,
    #[serde(rename = "exponential")]
    Exponential,
    #[serde(rename = "poly_left_right_exponential")]
    PolyLeftRightExponential,
    #[serde(rename = "exponential_ct")]
    ExponentialCT,
}

const fn _default_true() -> bool {
    true
}
const fn _default_false() -> bool {
    false
}
const fn _default_usize_null() -> Option<usize> {
    None
}
fn _default_input_rescaling() -> Vec<Vec<(f64, f64)>> {
    vec![vec![(0.0, 1.0); 3]; 15]
}
fn _default_shifts() -> Vec<(f64, f64, f64, f64)> {
    vec![(1.0, 0.0, 0.0, 0.0); 15]
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HFunctionSettings {
    pub function: HFunction,
    pub sigma: f64,
    #[serde(default = "_default_true")]
    pub enabled_dampening: bool,
    #[serde(default = "_default_usize_null")]
    pub power: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum ParameterizationMode {
    #[serde(rename = "cartesian")]
    Cartesian,
    #[serde(rename = "spherical")]
    Spherical,
    #[serde(rename = "hyperspherical")]
    HyperSpherical,
    #[serde(rename = "hyperspherical_flat")]
    HyperSphericalFlat,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum ParameterizationMapping {
    #[serde(rename = "log")]
    Log,
    #[serde(rename = "linear")]
    Linear,
}

#[derive(Debug, Default, Clone, Deserialize, PartialEq)]
pub enum CTVariable {
    #[default]
    #[serde(rename = "R")]
    Radius,
    #[serde(rename = "logR")]
    LogRadius,
}

#[derive(Debug, Default, Clone, Deserialize, PartialEq)]
pub enum NumeratorType {
    #[default]
    #[serde(rename = "one")]
    One,
    #[serde(rename = "spatial_exponential_dummy")]
    SpatialExponentialDummy,
    #[serde(rename = "physical")]
    Physical,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GeneralSettings {
    pub debug: usize,
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
    pub show_max_wgt_info: bool,
    pub use_weighted_average: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ParameterizationSettings {
    pub mode: ParameterizationMode,
    pub mapping: ParameterizationMapping,
    pub b: f64,
    #[serde(default = "_default_input_rescaling")]
    pub input_rescaling: Vec<Vec<(f64, f64)>>,
    #[serde(default = "_default_shifts")]
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
    #[serde(rename = "HardCodedIntegrand")]
    pub hard_coded_integrand: HardCodedIntegrandSettings,
    #[serde(rename = "Kinematics")]
    pub kinematics: KinematicsSettings,
    #[serde(rename = "Parameterization")]
    pub parameterization: ParameterizationSettings,
    #[serde(rename = "Integrator")]
    pub integrator: IntegratorSettings,
    #[serde(rename = "Observables")]
    pub observables: Vec<ObservableSettings>,
    #[serde(rename = "Selectors")]
    pub selectors: Vec<PhaseSpaceSelectorSettings>,
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

pub fn havana_integrate<F>(
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
    let mut integral = StatisticsAccumulator::new();
    let mut all_integrals = vec![StatisticsAccumulator::new(); N_INTEGRAND_ACCUMULATORS];

    let mut rng = rand::thread_rng();

    let mut user_data = user_data_generator(settings);

    let mut grid = user_data.integrand[0].create_grid();

    let grid_str = match &grid {
        Grid::Discrete(g) => format!(
            "top-level discrete {}-dimensional grid",
            format!("{}", g.bins.len()).bold().blue()
        ),
        Grid::Continuous(g) => {
            format!(
                "top-level continuous {}-dimensional grid",
                format!("{}", g.continuous_dimensions.len()).bold().blue()
            )
        }
    };

    let mut iter = 0;

    let cores = user_data.integrand.len();

    let t_start = Instant::now();
    println!(
        "Beta loop now integrates '{}' over a {} ...",
        format!("{}", settings.hard_coded_integrand).green(),
        grid_str
    );
    println!();
    while num_points < settings.integrator.n_max {
        let cur_points = settings.integrator.n_start + settings.integrator.n_increase * iter;
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
                    let fc = integrand_f.evaluate_sample(s, s.get_weight(), iter, false);
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
            settings.integrator.learning_rate,
        );
        integral.update_iter(settings.integrator.use_weighted_average);

        for i_integrand in 0..N_INTEGRAND_ACCUMULATORS {
            for (s, f) in samples[..cur_points].iter().zip(&f_evals[..cur_points]) {
                all_integrals[i_integrand].add_sample(f[i_integrand] * s.get_weight(), Some(s));
            }
            all_integrals[i_integrand].update_iter(settings.integrator.use_weighted_average);
        }

        if settings.general.debug > 1 {
            if let Grid::Discrete(g) = &grid {
                // g.bins[0]
                //     .plot(&format!("grid_disc_it_{}.svg", iter))
                //     .unwrap();

                println!("plotting grids is not supported on this branch");
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
            if let Grid::Discrete(g) = &grid {
                for (i, b) in g.bins.iter().enumerate() {
                    tabled_data.push(IntegralResult {
                        id: format!("chann#{}", i),
                        n_samples: format!("{}", b.accumulator.processed_samples),
                        n_samples_perc: format!(
                            "{:.3e}%",
                            ((b.accumulator.processed_samples as f64)
                                / (integral.processed_samples.max(1) as f64))
                                * 100.
                        ),
                        integral: format!("{:.8e}", b.accumulator.avg),
                        variance: format!(
                            "{:.8e}",
                            b.accumulator.err
                                * ((b.accumulator.processed_samples - 1).max(0) as f64).sqrt()
                        ),
                        err: format!("{:.8e}", b.accumulator.err),
                        err_perc: format!(
                            "{:.3e}%",
                            (b.accumulator.err / (b.accumulator.avg.abs()).max(1.0e-99)).abs()
                                * 100.
                        ),
                        pdf: format!("{:.8e}", if i > 1 { g.bins[i].pdf } else { g.bins[i].pdf }),
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
            "/  [ {} ] {}: n_pts={:-6.0}K {} {} /sample/core ",
            format!(
                "{:^7}",
                utils::format_wdhms(t_start.elapsed().as_secs() as usize)
            )
            .bold(),
            format!("Iteration #{:-4}", iter).bold().green(),
            cur_points as f64 / 1000.,
            if num_points >= 10_000_000 {
                format!("n_tot={:-7.0}M", num_points as f64 / 1_000_000.)
                    .bold()
                    .green()
            } else {
                format!("n_tot={:-7.0}K", num_points as f64 / 1000.)
                    .bold()
                    .green()
            },
            format!(
                "{:-17.3} ms",
                (((t_start.elapsed().as_secs() as f64) * 1000.) / (num_points as f64))
                    * (cores as f64)
            )
            .bold()
            .blue()
        );

        fn print_integral_result(
            itg: &StatisticsAccumulator<f64>,
            i_itg: usize,
            i_iter: usize,
            tag: &str,
            trgt: Option<f64>,
        ) {
            println!(
                "|  itg #{:-3} {}: {} {} {} {} {}",
                format!("{:<3}", i_itg),
                format!("{:-2}", tag).blue().bold(),
                format!("{:-19}", utils::format_uncertainty(itg.avg, itg.err))
                    .blue()
                    .bold(),
                if itg.avg != 0. {
                    if (itg.err / itg.avg).abs() > 0.01 {
                        format!(
                            "{:-8}",
                            format!("{:.3}%", (itg.err / itg.avg).abs() * 100.).red()
                        )
                    } else {
                        format!(
                            "{:-8}",
                            format!("{:.3}%", (itg.err / itg.avg).abs() * 100.).green()
                        )
                    }
                } else {
                    format!("{:-8}", "")
                },
                if itg.chi_sq / (i_iter as f64) > 5. {
                    format!("{:-6.3} χ²/dof", itg.chi_sq / (i_iter as f64)).red()
                } else {
                    format!("{:-6.3} χ²/dof", itg.chi_sq / (i_iter as f64)).normal()
                },
                if i_itg == 1 {
                    if let Some(t) = trgt {
                        if (t - itg.avg).abs() / itg.err > 5.
                            || (t.abs() != 0. && (t - itg.avg).abs() / t.abs() > 0.01)
                        {
                            format!(
                                "Δ={:-7.3}σ, Δ={:-7.3}%",
                                (t - itg.avg).abs() / itg.err,
                                if t.abs() > 0. {
                                    (t - itg.avg).abs() / t.abs() * 100.
                                } else {
                                    0.
                                }
                            )
                            .red()
                        } else {
                            format!(
                                "Δ={:-7.3}σ, Δ={:-7.3}%",
                                (t - itg.avg).abs() / itg.err,
                                if t.abs() > 0. {
                                    (t - itg.avg).abs() / t.abs() * 100.
                                } else {
                                    0.
                                }
                            )
                            .green()
                        }
                    } else {
                        format!("{}", "").normal()
                    }
                } else {
                    format!("{}", "").normal()
                },
                if itg.avg.abs() != 0. {
                    let mwi = itg.max_eval_negative.abs().max(itg.max_eval_positive.abs())
                        / (itg.avg.abs() * (itg.processed_samples as f64));
                    if mwi > 1. {
                        format!("  mwi: {:-5.3}", mwi).red()
                    } else {
                        format!("  mwi: {:-5.3}", mwi).normal()
                    }
                } else {
                    format!("  mwi: {:-5.3}", 0.).normal()
                }
            );
        }

        for i_integrand in 0..(N_INTEGRAND_ACCUMULATORS / 2) {
            print_integral_result(
                &all_integrals[2 * i_integrand],
                i_integrand + 1,
                iter,
                "re",
                if i_integrand == 0 {
                    target.map(|o| o.re).or(None)
                } else {
                    None
                },
            );
            print_integral_result(
                &all_integrals[2 * i_integrand + 1],
                i_integrand + 1,
                iter,
                "im",
                if i_integrand == 0 {
                    target.map(|o| o.im).or(None)
                } else {
                    None
                },
            );
        }
        if settings.integrator.show_max_wgt_info {
            println!("|  -------------------------------------------------------------------------------------------");
            println!(
                "|  {:<16} | {:<23} | {}",
                "Integrand", "Max Eval", "Max Eval xs",
            );
            for i_integrand in 0..(N_INTEGRAND_ACCUMULATORS / 2) {
                for part in 0..=1 {
                    for sgn in 0..=1 {
                        if (if sgn == 0 {
                            all_integrals[2 * i_integrand + part].max_eval_positive
                        } else {
                            all_integrals[2 * i_integrand + part].max_eval_negative
                        }) == 0.
                        {
                            continue;
                        }

                        println!(
                            "|  {:<20} | {:<23} | {}",
                            format!(
                                "itg #{:-3} {} [{}] ",
                                format!("{:<3}", i_integrand + 1),
                                format!("{:<2}", if part == 0 { "re" } else { "im" }).blue(),
                                format!("{:<1}", if sgn == 0 { "+" } else { "-" }).blue()
                            ),
                            format!(
                                "{:+.16e}",
                                if sgn == 0 {
                                    all_integrals[2 * i_integrand + part].max_eval_positive
                                } else {
                                    all_integrals[2 * i_integrand + part].max_eval_negative
                                }
                            ),
                            format!(
                                "( {} )",
                                if let Some(sample) = if sgn == 0 {
                                    &all_integrals[2 * i_integrand + part].max_eval_positive_xs
                                } else {
                                    &all_integrals[2 * i_integrand + part].max_eval_negative_xs
                                } {
                                    match sample {
                                        Sample::Continuous(_w, v) => v
                                            .iter()
                                            .map(|&x| format!("{:.16}", x))
                                            .collect::<Vec<_>>()
                                            .join(", "),
                                        _ => "N/A".to_string(),
                                    }
                                } else {
                                    format!("{}", "N/A")
                                }
                            )
                        );
                    }
                }
            }
        }
        // now merge all statistics and observables into the first
        let (first, others) = user_data.integrand[..cores].split_at_mut(1);
        for other_itg in others {
            first[0].merge_results(other_itg, iter);
        }

        // now write the observables to disk
        if let Some(itg) = user_data.integrand[..cores].first_mut() {
            itg.update_results(iter);
        }
        println!("");
    }

    IntegrationResult {
        neval: integral.processed_samples as i64,
        fail: integral.num_zero_evaluations as i32,
        result: all_integrals.iter().map(|res| res.avg).collect::<Vec<_>>(),
        error: all_integrals.iter().map(|res| res.err).collect::<Vec<_>>(),
        prob: all_integrals
            .iter()
            .map(|res| res.chi_sq)
            .collect::<Vec<_>>(),
    }
}

pub fn inspect(
    settings: &Settings,
    integrand: &mut Integrand,
    mut pt: Vec<f64>,
    mut force_radius: bool,
    is_momentum_space: bool,
    use_f128: bool,
) -> Complex<f64> {
    if integrand.get_n_dim() == pt.len() - 1 {
        force_radius = true;
    }

    let xs_f128 = if is_momentum_space {
        let (xs, inv_jac) = utils::global_inv_parameterize::<f128::f128>(
            &pt.chunks_exact_mut(3)
                .map(|x| LorentzVector::from_args(0., x[0], x[1], x[2]).cast())
                .collect::<Vec<LorentzVector<f128::f128>>>(),
            (settings.kinematics.e_cm * settings.kinematics.e_cm).into(),
            settings,
            force_radius,
        );
        if settings.general.debug > 1 {
            println!(
                "f128 sampling jacobian for this point = {:+.32e}",
                f128::f128::ONE / inv_jac
            );
        };
        xs
    } else {
        pt.iter().map(|x| f128::f128::from(*x)).collect::<Vec<_>>()
    };
    let xs_f64 = xs_f128
        .iter()
        .map(|x| f128::f128::to_f64(x).unwrap())
        .collect::<Vec<_>>();

    let eval = integrand.evaluate_sample(
        &Sample::Continuous(
            1.,
            if force_radius {
                xs_f64.clone()[1..].iter().map(|x| *x).collect::<Vec<_>>()
            } else {
                xs_f64.clone()
            },
        ),
        1.,
        1,
        use_f128,
    );
    println!();
    println!(
        "For input point xs: \n\n{}\n\nThe evaluation of integrand '{}' is:\n\n{}",
        format!(
            "( {} )",
            xs_f64
                .iter()
                .map(|&x| format!("{:.16}", x))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .blue(),
        format!("{}", settings.hard_coded_integrand).green(),
        format!("( {:+.16e}, {:+.16e} i)", eval.re, eval.im).blue(),
    );
    println!();

    eval
}
pub struct UserData {
    integrand: Vec<Integrand>,
}

fn print_banner() {
    println!(
        "{}{}{}",
        format!(
            "{}",
            r#"
    ______      _        _                       
    | ___ \    | |      | |                      
    | |_/ / ___| |_ __ _| |     ___   ___  _ __  
    | ___ \/ _ \ __/ _` | |    / _ \ / _ \| '_ \ 
    | |_/ /  __/ || (_| | |___| (_) | (_) | |_) |
    \____/ \___|\__\__,_\_____/\___/ \___/| .__/ 
                                            | |    
    "#
        )
        .bold()
        .blue(),
        format!("{:-26}", GIT_VERSION).green(),
        format!("{}", r#"              |_|    "#).bold().blue(),
    );
    println!();
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
                .short("t")
                .long("target")
                .multiple(true)
                .allow_hyphen_values(true)
                .value_name("TARGET")
                .help("Specify the integration target a <real> <imag>"),
        )
        .arg(
            Arg::with_name("debug")
                .short("d")
                .long("debug")
                .value_name("LEVEL")
                .help("Set the debug level. Higher means more verbose."),
        )
        .arg(
            Arg::with_name("n_start")
                .long("n_start")
                .value_name("N_START")
                .help("Number of starting samples for the integrator"),
        )
        .arg(
            Arg::with_name("n_max")
                .long("n_max")
                .value_name("N_MAX")
                .help("Max number of starting samples to consider for integration"),
        )
        .arg(
            Arg::with_name("n_increase")
                .long("n_increase")
                .value_name("N_INCREASE")
                .help("Increase of number of sample points for each successive iteration"),
        )
        .subcommand(
            SubCommand::with_name("inspect")
                .about("Inspect a single input point")
                .arg(
                    Arg::with_name("point")
                        .short("p")
                        .required(true)
                        .min_values(1)
                        .allow_hyphen_values(true),
                )
                .arg(
                    Arg::with_name("use_f128")
                        .short("f128")
                        .long("use_f128")
                        .help("Use f128 evaluation"),
                )
                .arg(
                    Arg::with_name("force_radius")
                        .long("force_radius")
                        .help("force radius in parameterisation"),
                )
                .arg(
                    Arg::with_name("momentum_space")
                        .short("m")
                        .long("momentum_space")
                        .help("Set if the point is specified in momentum space"),
                )
                .arg(
                    Arg::with_name("debug")
                        .short("d")
                        .long("debug")
                        .value_name("LEVEL")
                        .help("Set the debug level. Higher means more verbose."),
                ),
        )
        .subcommand(
            SubCommand::with_name("bench")
                .about("Benchmark timing for individual evaluations of the integrand")
                .arg(
                    Arg::with_name("samples")
                        .required(true)
                        .long("samples")
                        .short("s")
                        .value_name("SAMPLES")
                        .help("Number of samples for benchmark"),
                ),
        )
        .get_matches();

    let mut settings: Settings = Settings::from_file(matches.value_of("config").unwrap())?;

    print_banner();
    if settings.general.debug > 0 {
        println!(
            "{}",
            format!("Debug mode enabled at level {}", settings.general.debug).red()
        );
        println!();
    }

    if let Some(x) = matches.value_of("debug") {
        settings.general.debug = usize::from_str(x).unwrap();
    }
    if let Some(x) = matches.value_of("n_start") {
        settings.integrator.n_start = usize::from_str(x).unwrap();
    }
    if let Some(x) = matches.value_of("n_max") {
        settings.integrator.n_max = usize::from_str(x).unwrap();
    }
    if let Some(x) = matches.value_of("n_increase") {
        settings.integrator.n_increase = usize::from_str(x).unwrap();
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

    if let Some(matches) = matches.subcommand_matches("inspect") {
        if let Some(x) = matches.value_of("debug") {
            settings.general.debug = usize::from_str(x).unwrap();
        }

        let mut integrand = integrand_factory(&settings);

        let pt = matches
            .values_of("point")
            .unwrap()
            .map(|x| f64::from_str(x.trim_end_matches(',')).unwrap())
            .collect::<Vec<_>>();
        let force_radius = matches.is_present("force_radius");

        let _result = inspect(
            &settings,
            &mut integrand,
            pt.clone(),
            force_radius,
            matches.is_present("momentum_space"),
            matches.is_present("use_f128"),
        );
    } else if let Some(matches) = matches.subcommand_matches("bench") {
        let n_samples: usize = matches.value_of("samples").unwrap().parse().unwrap();
        println!();
        println!(
            "Benchmarking runtime of integrand '{}' over {} samples...",
            format!("{}", settings.hard_coded_integrand).green(),
            format!("{}", n_samples).blue()
        );
        let mut integrand = integrand_factory(&settings);
        let now = Instant::now();
        for _i in 1..n_samples {
            integrand.evaluate_sample(
                &Sample::Continuous(
                    1.,
                    (0..integrand.get_n_dim())
                        .map(|_i| rand::random::<f64>())
                        .collect(),
                ),
                1.,
                1,
                false,
            );
        }
        let total_time = now.elapsed().as_secs_f64();
        println!();
        println!(
            "> Total time: {} s for {} samples, {} ms per sample",
            format!("{:.1}", total_time).blue(),
            format!("{}", n_samples).blue(),
            format!("{:.5}", total_time * 1000. / (n_samples as f64)).green(),
        );
        println!();
    } else {
        let user_data_generator = |settings: &Settings| UserData {
            integrand: (0..num_integrands)
                .map(|_i| integrand_factory(settings))
                .collect(),
        };
        let result = havana_integrate(&settings, user_data_generator, target);

        println!("");
        println!(
            "{}",
            format!(
                "Havana integration completed after {} sample evaluations.",
                format!("{:.2}M", (result.neval as f64) / (1000000. as f64))
                    .bold()
                    .blue()
            )
            .bold()
            .green()
        );
        println!("");
    }
    Ok(())
}
