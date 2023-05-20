#![allow(dead_code)]
use crate::*;
use crate::{
    h_function_test::HFunctionTestSettings, integrands::UnitVolumeSettings,
    loop_induced_triboxtri::LoopInducedTriBoxTriSettings,
};

const CENTRAL_VALUE_TOLERANCE: f64 = 2.0e-2;
const INSPECT_TOLERANCE: f64 = 1.0e-15;
const DIFF_TARGET_TO_ERROR_MUST_BE_LESS_THAN: f64 = 3.;
const BASE_N_START_SAMPLE: usize = 100_000;

const N_CORES_FOR_INTEGRATION_IN_TESTS: usize = 16;

fn load_default_settings() -> Settings {
    Settings::from_file("./default_tests_config.yaml").unwrap()
}

fn approx_eq(res: f64, target: f64, tolerance: f64) -> bool {
    if target == 0.0 {
        return res.abs() < tolerance;
    } else {
        // println!("A1 {:.16e} vs {:.16e}", res, target);
        // println!(
        //     "A2 {:.16e} vs {:.16e}",
        //     ((res - target) / target).abs(),
        //     tolerance
        // );
        ((res - target) / target).abs() < tolerance
    }
}

fn validate_error(error: f64, target_diff: f64) -> bool {
    if target_diff == 0.0 {
        return true;
    } else {
        // println!("B1 {:.16e} vs {:.16e}", error, target_diff);
        // println!(
        //     "B2 {:.16e} vs {:.16e}",
        //     (target_diff / error).abs(),
        //     DIFF_TARGET_TO_ERROR_MUST_BE_LESS_THAN
        // );
        (error / target_diff).abs() < DIFF_TARGET_TO_ERROR_MUST_BE_LESS_THAN
    }
}

fn compare_integration(
    settings: &mut Settings,
    phase: IntegratedPhase,
    target: Complex<f64>,
) -> bool {
    // Allow this to fail as it may be called more than once
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_CORES_FOR_INTEGRATION_IN_TESTS)
        .build_global()
        .unwrap_or_else(|_| {});

    let user_data_generator = |settings: &Settings| UserData {
        integrand: (0..N_CORES_FOR_INTEGRATION_IN_TESTS)
            .map(|_i| crate::integrand_factory(settings))
            .collect(),
    };
    match phase {
        IntegratedPhase::Both => {
            (*settings).integrator.integrated_phase = IntegratedPhase::Real;
            let res = havana_integrate(&settings, user_data_generator, Some(target));
            if !approx_eq(res.result[0], target.re, CENTRAL_VALUE_TOLERANCE)
                || !validate_error(res.error[0], target.re - res.result[0])
            {
                println!(
                    "Incorrect real part of result: {:-19} vs {:.16e}",
                    format!(
                        "{:-19}",
                        utils::format_uncertainty(res.result[0], res.error[0])
                    )
                    .red()
                    .bold(),
                    target.re
                );
                return false;
            }
            (*settings).integrator.integrated_phase = IntegratedPhase::Imag;
            let res = havana_integrate(&settings, user_data_generator, Some(target));
            if !approx_eq(res.result[1], target.im, CENTRAL_VALUE_TOLERANCE)
                || !validate_error(res.error[1], target.re - res.result[1])
            {
                println!(
                    "Incorrect imag part of result: {:-19} vs {:.16e}",
                    format!(
                        "{:-19}",
                        utils::format_uncertainty(res.result[1], res.error[1])
                    )
                    .red()
                    .bold(),
                    target.im
                );
                return false;
            }
        }
        IntegratedPhase::Real => {
            (*settings).integrator.integrated_phase = IntegratedPhase::Real;
            let res = havana_integrate(&settings, user_data_generator, Some(target));
            if !approx_eq(res.result[0], target.re, CENTRAL_VALUE_TOLERANCE)
                || !validate_error(res.error[0], target.im - res.result[0])
            {
                println!(
                    "Incorrect real part of result: {:-19} vs {:.16e}",
                    format!(
                        "{:-19}",
                        utils::format_uncertainty(res.result[0], res.error[0])
                    )
                    .red()
                    .bold(),
                    target.re
                );
                return false;
            }
        }
        IntegratedPhase::Imag => {
            (*settings).integrator.integrated_phase = IntegratedPhase::Imag;
            let res = havana_integrate(&settings, user_data_generator, Some(target));
            if !approx_eq(res.result[1], target.im, CENTRAL_VALUE_TOLERANCE)
                || !validate_error(res.error[1], target.im - res.result[1])
            {
                println!(
                    "Incorrect imag part of result: {:-19} vs {:.16e}",
                    format!(
                        "{:-19}",
                        utils::format_uncertainty(res.result[1], res.error[1])
                    )
                    .red()
                    .bold(),
                    target.im
                );
                return false;
            }
        }
    }
    true
}

fn compare_inspect(
    settings: &mut Settings,
    pt: Vec<f64>,
    is_momentum_space: bool,
    target: Complex<f64>,
) -> bool {
    let integrand = integrand_factory(&settings);
    let res = inspect(&settings, &integrand, pt, false, is_momentum_space, true);
    if !approx_eq(res.re, target.re, INSPECT_TOLERANCE)
        || !approx_eq(res.im, target.im, INSPECT_TOLERANCE)
    {
        println!(
            "Incorrect result from inspect: {}\n                            vs {}",
            format!("{:+16e} + i {:+16e}", res.re, res.im).red().bold(),
            format!("{:.16e} + i {:+16e}", target.re, target.im)
                .red()
                .bold()
        );
        return false;
    }
    return true;
}

fn get_loop_induced_triboxtri_integrand() -> LoopInducedTriBoxTriSettings {
    let parsed_itg = serde_yaml::from_str("
    type: loop_induced_TriBoxTri
    supergraph_yaml_file: ./data/loop_induced_TriBoxTri.yaml
    q: [60.0, 0.0, 0.0, 0.0]
    h_function:
      function: poly_left_right_exponential # Options are poly_exponential, poly_left_right_exponential, exponential
      sigma: 20.
      power: 10 # Typically best to set to 3*n_loops+1
    threshold_CT_settings:
      enabled: true
      include_integrated_ct: true
      compute_only_im_squared: false
      im_squared_through_local_ct_only: false
      include_amplitude_level_cts: true
      variable: R # for the local CT: options in this case are R and logR
      local_ct_sliver_width: null # Specify an f64 for a finite sliver width, otherwise leave Option to null
      integrated_ct_sliver_width: null # Specify an f64 for a finite sliver width, otherwise leave Option to null
      parameterization_center: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      local_ct_h_function:
        function: exponential_ct # Options in this case is just exponential_ct
        sigma: 15.
        enabled_dampening: true
      integrated_ct_h_function:
        function: poly_left_right_exponential # Options are poly_exponential, poly_left_right_exponential, exponential
        sigma: 5.
        power: 0
").unwrap();
    match parsed_itg {
        HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg) => itg,
        _ => panic!("Wrong type of integrand"),
    }
}

fn get_h_function_test_integrand() -> HFunctionTestSettings {
    let parsed_itg = serde_yaml::from_str(
        "
    type: h_function_test
    h_function:
        function: poly_left_right_exponential # Options are poly_exponential, exponential
        sigma: 0.01
        power: 12
",
    )
    .unwrap();
    match parsed_itg {
        HardCodedIntegrandSettings::HFunctionTest(itg) => itg,
        _ => panic!("Wrong type of integrand"),
    }
}

fn get_unit_volume_integrand() -> UnitVolumeSettings {
    let parsed_itg = serde_yaml::from_str(
        "
    type: unit_volume
    n_3d_momenta: 11
",
    )
    .unwrap();
    match parsed_itg {
        HardCodedIntegrandSettings::UnitVolume(itg) => itg,
        _ => panic!("Wrong type of integrand"),
    }
}

#[cfg(test)]
mod tests_integral {
    use super::*;

    #[test]
    fn unit_volume_11_momenta_hyperspherical_flat() {
        let mut settings = load_default_settings();
        let mut itg = get_unit_volume_integrand();
        settings.integrator.n_start = 5 * BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 1.;

        itg.n_3d_momenta = 11;
        settings.parameterization.mode = ParameterizationMode::HyperSphericalFlat;
        settings.hard_coded_integrand = HardCodedIntegrandSettings::UnitVolume(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Real,
            Complex::new(1.0, 0.0)
        ));
    }

    #[test]
    fn unit_volume_3_momenta_hyperspherical() {
        let mut settings = load_default_settings();
        let mut itg = get_unit_volume_integrand();
        settings.integrator.n_start = 5 * BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 1.;

        itg.n_3d_momenta = 3;
        settings.parameterization.mode = ParameterizationMode::HyperSpherical;
        settings.hard_coded_integrand = HardCodedIntegrandSettings::UnitVolume(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Real,
            Complex::new(1.0, 0.0)
        ));
    }

    #[test]
    fn unit_volume_3_momenta_spherical() {
        let mut settings = load_default_settings();
        let mut itg = get_unit_volume_integrand();
        settings.integrator.n_start = 5 * BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 1.;

        itg.n_3d_momenta = 3;
        settings.parameterization.mode = ParameterizationMode::Spherical;
        settings.hard_coded_integrand = HardCodedIntegrandSettings::UnitVolume(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Real,
            Complex::new(1.0, 0.0)
        ));
    }

    #[test]
    fn poly_left_right_exponential_h_function() {
        let mut settings = load_default_settings();
        let mut itg = get_h_function_test_integrand();
        settings.integrator.n_start = 5 * BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 1.;

        itg.h_function = HFunctionSettings {
            function: HFunction::PolyLeftRightExponential,
            sigma: 0.01,
            power: Some(12),
            enabled_dampening: true,
        };
        settings.hard_coded_integrand = HardCodedIntegrandSettings::HFunctionTest(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Real,
            Complex::new(1.0, 0.0)
        ));
    }

    #[test]
    fn loop_induced_triboxtri_euclidean() {
        let mut settings = load_default_settings();
        let mut itg = get_loop_induced_triboxtri_integrand();
        itg.supergraph_yaml_file = "./data/loop_induced_TriBoxTri.yaml".to_string();
        itg.q = [10.0, 0.0, 0.0, 0.0];
        settings.integrator.n_start = BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 10.;

        // cargo run --release -- --n_start 100000 --n_increase 0 --n_max 1000000 --config ../betaloop_config.yaml -d 0 -c 16 --target 6.4630e-14
        settings.hard_coded_integrand =
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Real,
            Complex::new(5.5881e-13, 0.0)
        ));
    }

    #[test]
    fn loop_induced_triboxtri_physical() {
        let mut settings = load_default_settings();
        let mut itg = get_loop_induced_triboxtri_integrand();
        itg.supergraph_yaml_file = "./data/loop_induced_TriBoxTri.yaml".to_string();
        itg.q = [60.0, 0.0, 0.0, 0.0];
        settings.integrator.n_start = BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 60.;

        itg.threshold_ct_settings.include_integrated_ct = true;
        itg.threshold_ct_settings.compute_only_im_squared = false;
        itg.threshold_ct_settings.im_squared_through_local_ct_only = false;
        itg.threshold_ct_settings.include_amplitude_level_cts = true;

        // cargo run --release -- --n_start 100000 --n_increase 0 --n_max 1000000 --config ../betaloop_config.yaml -d 0 -c 16 --target 6.4630e-14, 4.2985e-14
        settings.hard_coded_integrand =
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Both,
            Complex::new(6.4630e-14, 4.2985e-14)
        ));
    }

    #[test]
    fn loop_induced_triboxtri_physical_im_squared_trick() {
        let mut settings = load_default_settings();
        let mut itg = get_loop_induced_triboxtri_integrand();
        itg.supergraph_yaml_file = "./data/loop_induced_TriBoxTri.yaml".to_string();
        itg.q = [60.0, 0.0, 0.0, 0.0];

        itg.threshold_ct_settings.include_integrated_ct = false;
        itg.threshold_ct_settings.compute_only_im_squared = false;
        itg.threshold_ct_settings.im_squared_through_local_ct_only = true;
        itg.threshold_ct_settings.include_amplitude_level_cts = true;

        settings.integrator.n_start = BASE_N_START_SAMPLE;
        settings.integrator.n_max = 10 * BASE_N_START_SAMPLE;
        settings.integrator.n_increase = 0;
        settings.integrator.n_increase = 0;
        settings.kinematics.e_cm = 60.;

        // cargo run --release -- --n_start 100000 --n_increase 0 --n_max 1000000 --config ../betaloop_config.yaml -d 0 -c 16 --target 6.4630e-14, 0.0
        settings.hard_coded_integrand =
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg.clone());
        assert!(compare_integration(
            &mut settings,
            IntegratedPhase::Real,
            Complex::new(6.4630e-14, 0.0)
        ));
    }
}

#[cfg(test)]
mod tests_inspect {
    use super::*;

    // Amazingly enough, a simple failing test induces a segfault on MacOS... :/
    // #[test]
    // fn this_test_will_not_pass() {
    //     assert!(false);
    // }

    #[test]
    fn inspect_loop_induced_triboxtri() {
        let mut settings = load_default_settings();
        let mut itg = get_loop_induced_triboxtri_integrand();
        itg.supergraph_yaml_file = "./data/loop_induced_TriBoxTri.yaml".to_string();

        // ./target/release/betaloop --n_start 1000000 --config ../betaloop_config.yaml -d 4 -c 1 inspect --use_f128 -m -p 0. 0. 4.7140452079 0. 0. 3.7267799624 0. 3. 4.
        settings.hard_coded_integrand =
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg.clone());
        assert!(compare_inspect(
            &mut settings,
            vec![0., 0., 4.7140452079, 0., 0., 3.7267799624, 0., 3., 4.],
            true,
            Complex::new(-2.2473010401646284e-12, 7.6830999412124933e-14)
        ));

        itg.threshold_ct_settings.im_squared_through_local_ct_only = true;
        settings.hard_coded_integrand =
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg.clone());
        assert!(compare_inspect(
            &mut settings,
            vec![0., 0., 4.7140452079, 0., 0., 3.7267799624, 0., 3., 4.],
            true,
            Complex::new(1.5743284974316047e-11, 7.683099941212493e-14)
        ));

        itg.threshold_ct_settings.parameterization_center =
            vec![[1.0, -2.0, 3.0], [5.0, -2.0, 7.0]];
        settings.hard_coded_integrand =
            HardCodedIntegrandSettings::LoopInducedTriBoxTri(itg.clone());
        assert!(compare_inspect(
            &mut settings,
            vec![0., 0., 4.7140452079, 0., 0., 3.7267799624, 0., 3., 4.],
            true,
            Complex::new(7.350891519922086e-12, 1.2720710886848654e-13)
        ));
    }
}
